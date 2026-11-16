"""Stateless runs: health and one agent run."""

import asyncio
import logging
from typing import Annotated

from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from fastapi import APIRouter, Depends, HTTPException
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.language_models.chat_models import BaseChatModel
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError

from app.api.routes.chat import configured_model
from app.core.settings import get_settings
from app.graph.nodes.agent import AgentRunError
from app.graph.workflow import run_agent
from app.schemas.runs import RunRequest, RunResponse, ToolCallTrace
from app.tools.calculator.calculate import calculate_tool
from app.tools.sql.news_tool import search_news_tool

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.post("/runs", response_model=RunResponse, tags=["runs"])
async def create_run(
    request: RunRequest,
    model: Annotated[BaseChatModel, Depends(configured_model)],
) -> RunResponse:
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.agent_timeout_seconds):
            result = await run_agent(
                model,
                [search_news_tool, calculate_tool],
                request.question,
                max_steps=settings.agent_max_steps,
                max_retries=settings.verifier_max_retries,
                max_tool_calls=settings.agent_max_tool_calls,
            )
    except TimeoutError, OpenAITimeoutError, AnthropicTimeoutError:
        raise HTTPException(
            status_code=504, detail="The agent run timed out."
        ) from None
    except ModelCallLimitExceededError, AgentRunError:
        raise HTTPException(
            status_code=502, detail="The agent did not reach an answer."
        ) from None
    except OpenAIAPIError, AnthropicAPIError:
        # Provider errors may include credentials or submitted text.
        raise HTTPException(
            status_code=502, detail="The model provider request failed."
        ) from None
    if not result.answer.strip():
        raise HTTPException(status_code=502, detail="The agent returned no text.")
    return RunResponse(
        answer=result.answer,
        tool_calls=[
            ToolCallTrace(name=c.name, args=c.args, ok=c.ok) for c in result.tool_calls
        ],
        verified=result.verified,
        unsupported=result.unsupported,
    )
