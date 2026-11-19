"""Chat sessions: several turns that remember each other, stored in finsights_ai (Node 1).

The caller is identified by the `X-User-Id` header, set by the gateway after authentication;
this service does not authenticate and must not be exposed directly. Sessions of another user
answer 404, so their existence is not revealed.
"""

import asyncio
import logging
import math
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Annotated, Any

from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models.chat_models import BaseChatModel
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes.chat import configured_model
from app.api.routes.runs import NEWS_TOOLS
from app.core.settings import get_settings
from app.graph import compaction
from app.graph.history import MAX_UNFOLDED_TURNS, Context, Digest, build_digest, to_fold
from app.graph.nodes.agent import AgentRunError, RunResult
from app.graph.workflow import EventSink, TokenTally, run_agent
from app.integrations.llm.config import ModelConfigError
from app.integrations.llm.factory import get_chat_model
from app.schemas.runs import ToolCallTrace
from app.schemas.sessions import (
    MessageRequest,
    SessionDetail,
    SessionMessage,
    SessionSummary,
    TurnResponse,
)
from app.storage.mysql import chat_store, usage_store

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


def compactor_model(profile: str) -> BaseChatModel | None:
    """The cheap tier of the request's profile folds history; None disables folding."""
    try:
        return get_chat_model(profile, "cheap")
    except ModelConfigError:
        return None


async def _compact(
    compactor: BaseChatModel | None,
    context: Context | None,
    history: list[Digest],
    on_event: EventSink | None = None,
) -> tuple[Context | None, dict[str, Any] | None]:
    """(new checkpoint or None, trace or None). Any failure means truncation instead."""
    fold = to_fold(context, history)
    if not fold or compactor is None:
        return None, None
    if on_event is not None:
        await on_event("compacting", {})
    try:
        async with asyncio.timeout(get_settings().llm_timeout_seconds):
            result = await compaction.fold_turns(
                compactor, context, fold, today=date.today()
            )
    except (
        TimeoutError,
        OpenAITimeoutError,
        AnthropicTimeoutError,
        OpenAIAPIError,
        AnthropicAPIError,
    ) as exc:
        logger.warning("compaction failed: %s", type(exc).__name__)
        return None, {"folded_turns": [d["turn"] for d in fold], "ok": False}
    return result.context, compaction.trace(result)


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


def quota_limits() -> dict[str, int]:
    settings = get_settings()
    return {
        usage_store.FIVE_HOUR: settings.quota_five_hour_tokens,
        usage_store.WEEKLY: settings.quota_weekly_tokens,
    }


async def within_quota(user: User) -> None:
    """Refuse a turn while a token quota window of the user is used up (429)."""
    limits = quota_limits()
    if not any(limits.values()):
        return
    try:
        row = await usage_store.get_usage(user)
    except SQLAlchemyError, RuntimeError:
        logger.exception("loading token usage failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    now = usage_store.utc_now()
    spent = [
        (window.resets_at, name)
        for name, window in usage_store.windows(row, now, limits).items()
        if window.exhausted and window.resets_at
    ]
    if not spent:
        return
    resets_at, name = max(spent)  # both used up: the later reset is the one that binds
    raise HTTPException(
        status_code=429,
        detail={
            "message": "Token quota used up.",
            "limit": name,
            # Naive UTC, written as GET /usage writes it.
            "resets_at": resets_at.isoformat() + "Z",
        },
        headers={"Retry-After": str(math.ceil((resets_at - now).total_seconds()))},
    )


@dataclass
class Conversation:
    """A session and what its earlier turns left for the next one."""

    user: str
    session: dict[str, Any]
    context: Context | None
    history: list[Digest]


async def conversation(
    user: User,
    session_uid: str,
    _quota: Annotated[None, Depends(within_quota)],
) -> Conversation:
    """Loaded before the turn starts, so a stream never opens for a missing session or a
    used-up quota."""
    session = await _session(user, session_uid)
    try:
        context = await chat_store.latest_context(session["id"])
        through = context["through_turn"] if context else 0
        history = await chat_store.digests_after(
            session["id"], through, MAX_UNFOLDED_TURNS
        )
    except SQLAlchemyError, RuntimeError:
        logger.exception("loading chat history failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    return Conversation(user, session, context, history)


ChatModel = Annotated[BaseChatModel, Depends(configured_model)]
Compactor = Annotated[BaseChatModel | None, Depends(compactor_model)]


@router.post("/{session_uid}/messages", response_model=TurnResponse)
async def send_message(
    request: MessageRequest,
    chat: Annotated[Conversation, Depends(conversation)],
    model: ChatModel,
    compactor: Compactor,
) -> TurnResponse:
    return await _turn(chat, request.question, model, compactor)


@router.post("/{session_uid}/messages/stream", response_class=EventSourceResponse)
async def stream_message(
    request: MessageRequest,
    chat: Annotated[Conversation, Depends(conversation)],
    model: ChatModel,
    compactor: Compactor,
) -> AsyncIterable[ServerSentEvent]:
    """The same turn as `send_message`, with its progress as Server-Sent Events.

    Events: `compacting`, `tool_start`, `tool_end`, `verifying`, `retrying`, then either
    `answer` (a TurnResponse, sent only after the verifier) or `error` ({status, detail}).
    A client that disconnects cancels the turn; nothing is saved.
    """
    events: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

    async def emit(name: str, data: dict[str, Any]) -> None:
        await events.put(ServerSentEvent(data=data, event=name))

    async def run() -> None:
        try:
            response = await _turn(chat, request.question, model, compactor, emit)
            await events.put(ServerSentEvent(data=response, event="answer"))
        except HTTPException as exc:
            await emit("error", {"status": exc.status_code, "detail": exc.detail})
        except Exception:
            logger.exception("streamed chat turn failed")
            await emit("error", {"status": 500, "detail": "Internal server error."})
        finally:
            await events.put(None)

    task = asyncio.create_task(run())
    try:
        while (event := await events.get()) is not None:
            yield event
    finally:
        task.cancel()  # no-op once the turn has finished


async def _turn(
    chat: Conversation,
    question: str,
    model: BaseChatModel,
    compactor: BaseChatModel | None,
    on_event: EventSink | None = None,
) -> TurnResponse:
    """Run, save and return one turn; failures raise HTTPException."""
    session, context, history = chat.session, chat.context, chat.history
    settings = get_settings()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    started = time.monotonic()
    started_at = usage_store.utc_now()  # a new quota window opens at the turn's start
    tally = TokenTally()
    # Fold old turns before the run, so this turn already reads the smaller history.
    checkpoint, fold_trace = await _compact(compactor, context, history, on_event)
    context = checkpoint or context
    extra: dict[str, Any] = {}
    if fold_trace:
        extra["compaction"] = fold_trace
    if checkpoint:
        extra["context"] = checkpoint  # later turns resume from here
    try:
        async with asyncio.timeout(settings.agent_timeout_seconds):
            result: RunResult = await run_agent(
                model,
                NEWS_TOOLS,
                question,
                max_steps=settings.agent_max_steps,
                max_retries=settings.verifier_max_retries,
                max_tool_calls=settings.agent_max_tool_calls,
                history=history,
                context=context,
                on_event=on_event,
                tally=tally,
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
                question=question,
                answer=detail,
                status=chat_store.FAILED,
                model_name=model_name,
                error_code=code,
                prompt_tokens=tally.input or None,
                completion_tokens=tally.output or None,
                ai_trace={
                    "digest": build_digest(question, detail, verified=False),
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    **extra,
                },
            )
        except HTTPException:
            pass  # a concurrent turn won; report the run's own failure
        raise HTTPException(status_code=status, detail=detail) from None
    finally:
        # Every run is charged, failed or cancelled (client gone) included: tokens were spent.
        await _charge(chat.user, tally, started_at)

    turn = await _save(
        session,
        question=question,
        answer=result.answer,
        status=chat_store.COMPLETE,
        model_name=model_name,
        prompt_tokens=result.input_tokens,
        completion_tokens=result.output_tokens,
        ai_trace={
            "digest": build_digest(question, result.answer, verified=result.verified),
            "tool_calls": [asdict(c) for c in result.tool_calls],
            "verified": result.verified,
            "unsupported": result.unsupported,
            "latency_ms": int((time.monotonic() - started) * 1000),
            # No column for it in finsights_ai: prompt_tokens includes these cache hits.
            "cache_read_tokens": result.cache_read_tokens,
            **extra,
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


async def _charge(user: str, tally: TokenTally, at: datetime) -> None:
    """Count a run against the user's token quota. Losing the count must not lose the
    answer: failures only log."""
    tokens = usage_store.billable_tokens(tally.input, tally.output, tally.cache_read)
    if not tokens:
        return
    try:
        await usage_store.charge(user, tokens, at)
    except SQLAlchemyError, RuntimeError:
        logger.exception("charging token usage failed")


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
