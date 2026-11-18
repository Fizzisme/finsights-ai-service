"""Chat sessions: several turns that remember each other, stored in finsights_ai (Node 1).

The caller is identified by the `X-User-Id` header, set by the gateway after authentication;
this service does not authenticate and must not be exposed directly. Sessions of another user
answer 404, so their existence is not revealed.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from typing import Annotated, Any

from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models.chat_models import BaseChatModel
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes.chat import configured_model
from app.api.routes.runs import NEWS_TOOLS
from app.core.settings import get_settings
from app.graph.history import KEEP_RECENT_TURNS, build_digest
from app.graph.nodes.agent import AgentRunError, RunResult
from app.graph.workflow import run_agent
from app.schemas.runs import ToolCallTrace
from app.schemas.sessions import (
    MessageRequest,
    SessionDetail,
    SessionMessage,
    SessionSummary,
    TurnResponse,
)
from app.storage.mysql import chat_store

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)

UNAVAILABLE = "Chat history is temporarily unavailable."


def current_user(x_user_id: Annotated[str | None, Header()] = None) -> str:
    try:
        return str(uuid.UUID(x_user_id or ""))
    except ValueError:
        raise HTTPException(
            status_code=401, detail="Missing or invalid X-User-Id header."
        ) from None


User = Annotated[str, Depends(current_user)]


async def _session(user: str, session_uid: str) -> dict[str, Any]:
    try:
        uuid.UUID(session_uid)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found.") from None
    try:
        session = await chat_store.get_session(user, session_uid)
    except SQLAlchemyError, RuntimeError:
        logger.exception("loading chat session failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _failure(exc: Exception) -> tuple[int, str, str]:
    """(HTTP status, user-facing detail, stored error code) of a failed agent run."""
    if isinstance(exc, (TimeoutError, OpenAITimeoutError, AnthropicTimeoutError)):
        return 504, "The agent run timed out.", "timeout"
    if isinstance(exc, (ModelCallLimitExceededError, AgentRunError)):
        return 502, "The agent did not reach an answer.", "no_answer"
    # Provider errors may include credentials or submitted text: never echo them.
    return 502, "The model provider request failed.", "provider_error"


@router.post("", response_model=SessionSummary, status_code=201)
async def create_session(user: User) -> Any:
    try:
        return await chat_store.create_session(user)
    except SQLAlchemyError, RuntimeError:
        logger.exception("creating chat session failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    user: User, limit: Annotated[int, Query(ge=1, le=100)] = 50
) -> Any:
    try:
        return await chat_store.list_sessions(user, limit)
    except SQLAlchemyError, RuntimeError:
        logger.exception("listing chat sessions failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None


@router.get("/{session_uid}", response_model=SessionDetail)
async def get_session(user: User, session_uid: str) -> Any:
    session = await _session(user, session_uid)
    try:
        rows = await chat_store.list_messages(session["id"])
    except SQLAlchemyError, RuntimeError:
        logger.exception("loading chat messages failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    messages = [
        SessionMessage(
            turn=r["turn"],
            speaker=r["speaker"],
            text=r["message_text"],
            status=r["message_status"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
    return SessionDetail(**session, messages=messages)


@router.delete("/{session_uid}", status_code=204)
async def delete_session(user: User, session_uid: str) -> Response:
    await _session(user, session_uid)
    try:
        await chat_store.delete_session(user, session_uid)
    except SQLAlchemyError, RuntimeError:
        logger.exception("deleting chat session failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    return Response(status_code=204)


@router.post("/{session_uid}/messages", response_model=TurnResponse)
async def send_message(
    user: User,
    session_uid: str,
    request: MessageRequest,
    model: Annotated[BaseChatModel, Depends(configured_model)],
) -> TurnResponse:
    session = await _session(user, session_uid)
    try:
        history = await chat_store.recent_digests(session["id"], KEEP_RECENT_TURNS)
    except SQLAlchemyError, RuntimeError:
        logger.exception("loading chat history failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None

    settings = get_settings()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    started = time.monotonic()
    try:
        async with asyncio.timeout(settings.agent_timeout_seconds):
            result: RunResult = await run_agent(
                model,
                NEWS_TOOLS,
                request.question,
                max_steps=settings.agent_max_steps,
                max_retries=settings.verifier_max_retries,
                max_tool_calls=settings.agent_max_tool_calls,
                history=history,
            )
        if not result.answer.strip():
            raise AgentRunError("empty answer")
    except (
        TimeoutError,
        OpenAITimeoutError,
        AnthropicTimeoutError,
        ModelCallLimitExceededError,
        AgentRunError,
        OpenAIAPIError,
        AnthropicAPIError,
    ) as exc:
        status, detail, code = _failure(exc)
        # Keep the failed turn so the user sees it and later turns know it failed.
        try:
            await _save(
                session,
                question=request.question,
                answer=detail,
                status=chat_store.FAILED,
                model_name=model_name,
                error_code=code,
                ai_trace={
                    "digest": build_digest(request.question, detail, verified=False),
                    "latency_ms": int((time.monotonic() - started) * 1000),
                },
            )
        except HTTPException:
            pass  # a concurrent turn won; report the run's own failure
        raise HTTPException(status_code=status, detail=detail) from None

    turn = await _save(
        session,
        question=request.question,
        answer=result.answer,
        status=chat_store.COMPLETE,
        model_name=model_name,
        prompt_tokens=result.input_tokens,
        completion_tokens=result.output_tokens,
        ai_trace={
            "digest": build_digest(
                request.question, result.answer, verified=result.verified
            ),
            "tool_calls": [asdict(c) for c in result.tool_calls],
            "verified": result.verified,
            "unsupported": result.unsupported,
            "latency_ms": int((time.monotonic() - started) * 1000),
        },
    )
    return TurnResponse(
        session_uid=session["session_uid"],
        turn=turn,
        answer=result.answer,
        tool_calls=[
            ToolCallTrace(name=c.name, args=c.args, ok=c.ok) for c in result.tool_calls
        ],
        verified=result.verified,
        unsupported=result.unsupported,
    )


async def _save(session: dict[str, Any], **turn: Any) -> int | None:
    """Store a turn. Losing history must not lose the answer: failures only log."""
    try:
        return await chat_store.save_turn(session, **turn)
    except chat_store.TurnConflict:
        raise HTTPException(
            status_code=409,
            detail="Another message of this session was saved at the same time.",
        ) from None
    except SQLAlchemyError, RuntimeError:
        logger.exception("saving chat turn failed")
        return None
