"""Stateless chat through the configured LangChain model."""

import asyncio
from typing import Annotated

from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError

from app.core.settings import get_settings
from app.integrations.llm.catalog import Tier
from app.integrations.llm.config import ModelConfigError
from app.integrations.llm.factory import get_chat_model
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


def configured_model(profile: str, tier: Tier = "standard") -> BaseChatModel:
    try:
        return get_chat_model(profile, tier)
    except ModelConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model: Annotated[BaseChatModel, Depends(configured_model)],
) -> ChatResponse:
    try:
        async with asyncio.timeout(get_settings().llm_timeout_seconds):
            response = await model.ainvoke([HumanMessage(content=request.message)])
        answer = StrOutputParser().invoke(response)
    except TimeoutError, OpenAITimeoutError, AnthropicTimeoutError:
        raise HTTPException(
            status_code=504, detail="The model provider timed out."
        ) from None
    except OpenAIAPIError, AnthropicAPIError:
        # Provider errors may include credentials or submitted text.
        raise HTTPException(
            status_code=502, detail="The model provider request failed."
        ) from None
    if not answer.strip():
        raise HTTPException(
            status_code=502, detail="The model provider returned no text."
        )
    return ChatResponse(answer=answer)
