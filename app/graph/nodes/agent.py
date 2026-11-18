"""The single agent loop: LangChain's create_agent (a compiled LangGraph graph).

It runs model -> tool calls -> ToolMessage -> model ... until the model answers without
tool calls. The workflow graph wraps it with the verifier (see app/graph/workflow.py).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool

from app.prompts import load_prompt

PROMPT_NAME = "news_agent"


class AgentRunError(RuntimeError):
    """The agent ended without a usable final answer."""


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool


@dataclass
class RunResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    verified: bool = True
    unsupported: list[str] = field(default_factory=list)
    # Summed over every model call of the run; None when the provider reports no usage.
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Part of input_tokens served from the provider's prompt cache (billed far cheaper).
    cache_read_tokens: int | None = None


def token_usage(
    messages: Sequence[BaseMessage],
) -> tuple[int | None, int | None, int | None]:
    """(input, output, cache-read input) tokens of all model calls in `messages`."""
    usages = [
        m.usage_metadata
        for m in messages
        if isinstance(m, AIMessage) and m.usage_metadata
    ]
    if not usages:
        return None, None, None
    return (
        sum(u.get("input_tokens", 0) for u in usages),
        sum(u.get("output_tokens", 0) for u in usages),
        sum(u.get("input_token_details", {}).get("cache_read", 0) for u in usages),
    )


def build_user_message(question: str, today: date, history: str = "") -> HumanMessage:
    """Wrap the question in source tags so the model treats it as data, not rules.

    `history` is an already-escaped `<conversation_history>` block (app/graph/history.py).
    """
    # A question cannot close the tag early and smuggle text outside it.
    safe = question.replace("</untrusted_input>", "")
    context = f"{history}\n" if history else ""
    return HumanMessage(
        f"<today>{today.isoformat()}</today>\n{context}"
        f"<untrusted_input>\n{safe}\n</untrusted_input>"
    )


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_steps: int,
    max_tool_calls: int | None = None,
):
    middleware: list[Any] = [
        # Hard stop against runaway loops (and cost): raises instead of ending silently.
        ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="error")
    ]
    if max_tool_calls is not None:
        # Calls over the budget are answered with an error ToolMessage, so the model
        # can still write its final answer from what it already has.
        middleware.append(
            ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue")
        )
    return create_agent(
        model,
        tools=list(tools),
        system_prompt=load_prompt(PROMPT_NAME),
        middleware=middleware,
    )


def trace_tool_calls(messages: Sequence[BaseMessage]) -> list[ToolCallRecord]:
    status = {m.tool_call_id: m.status for m in messages if isinstance(m, ToolMessage)}
    return [
        ToolCallRecord(
            call["name"], dict(call["args"]), status.get(call["id"] or "") != "error"
        )
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def final_answer(messages: Sequence[BaseMessage]) -> str:
    """Text of the last message, which must be a plain AI answer (no pending tool calls)."""
    final = messages[-1]
    if not isinstance(final, AIMessage) or final.tool_calls:
        raise AgentRunError("agent did not finish with an answer")
    return StrOutputParser().invoke(final)
