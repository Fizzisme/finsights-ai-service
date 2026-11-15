"""The single agent loop: LangChain's create_agent (a compiled LangGraph graph).

It runs model -> tool calls -> ToolMessage -> model ... until the model answers without
tool calls. The router, verifier and fast-path nodes will wrap this node later.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
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


def build_user_message(question: str, today: date) -> HumanMessage:
    """Wrap the question in source tags so the model treats it as data, not rules."""
    # A question cannot close the tag early and smuggle text outside it.
    safe = question.replace("</untrusted_input>", "")
    return HumanMessage(
        f"<today>{today.isoformat()}</today>\n<untrusted_input>\n{safe}\n</untrusted_input>"
    )


def build_agent(model: BaseChatModel, tools: Sequence[BaseTool], *, max_steps: int):
    return create_agent(
        model,
        tools=list(tools),
        system_prompt=load_prompt(PROMPT_NAME),
        # Hard stop against runaway loops (and cost): raises instead of ending silently.
        middleware=[
            ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="error")
        ],
    )


def _trace(messages: Sequence[BaseMessage]) -> list[ToolCallRecord]:
    status = {m.tool_call_id: m.status for m in messages if isinstance(m, ToolMessage)}
    return [
        ToolCallRecord(
            call["name"], dict(call["args"]), status.get(call["id"] or "") != "error"
        )
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


async def run_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    question: str,
    *,
    max_steps: int,
    today: date | None = None,
) -> RunResult:
    agent = build_agent(model, tools, max_steps=max_steps)
    state = await agent.ainvoke(
        {"messages": [build_user_message(question, today or date.today())]}
    )
    messages: list[BaseMessage] = state["messages"]
    final = messages[-1]
    if not isinstance(final, AIMessage) or final.tool_calls:
        raise AgentRunError("agent did not finish with an answer")
    return RunResult(
        answer=StrOutputParser().invoke(final), tool_calls=_trace(messages)
    )
