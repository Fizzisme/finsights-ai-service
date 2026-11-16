"""Online workflow: agent loop -> verifier -> (bounded retry) -> answer.

START -> agent -> verifier --ok / retries exhausted--> END
            ^         |
            +---------+  (feedback message, at most `max_retries` times)
"""

from collections.abc import Sequence
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from app.graph.nodes.agent import (
    RunResult,
    build_agent,
    build_user_message,
    final_answer,
    trace_tool_calls,
)
from app.graph.nodes.verifier import VerifyResult, feedback_message, verify_answer


class RunState(MessagesState):
    retries: int
    verification: dict[str, Any]


def build_workflow(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    question: str,
    today: date,
    max_steps: int,
    max_retries: int,
    max_tool_calls: int | None = None,
):
    def verifier(state: RunState) -> dict[str, Any]:
        messages: Sequence[BaseMessage] = state["messages"]
        result = verify_answer(
            final_answer(messages), messages, question=question, today=today
        )
        update: dict[str, Any] = {
            "verification": {
                "passed": result.passed,
                "unsupported": result.unsupported,
            }
        }
        if not result.passed and state["retries"] < max_retries:
            update["messages"] = [HumanMessage(feedback_message(result))]
            update["retries"] = state["retries"] + 1
            update["verification"]["retry"] = True
        return update

    def route(state: RunState) -> str:
        return "agent" if state["verification"].get("retry") else END

    builder = StateGraph(RunState)
    builder.add_node(
        "agent",
        build_agent(model, tools, max_steps=max_steps, max_tool_calls=max_tool_calls),
    )
    builder.add_node("verifier", verifier)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "verifier")
    builder.add_conditional_edges("verifier", route, ["agent", END])
    return builder.compile()


async def run_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    question: str,
    *,
    max_steps: int,
    max_retries: int = 1,
    max_tool_calls: int | None = None,
    today: date | None = None,
) -> RunResult:
    today = today or date.today()
    graph = build_workflow(
        model,
        tools,
        question=question,
        today=today,
        max_steps=max_steps,
        max_retries=max_retries,
        max_tool_calls=max_tool_calls,
    )
    state = await graph.ainvoke(
        {
            "messages": [build_user_message(question, today)],
            "retries": 0,
            "verification": {},
        }
    )
    messages: Sequence[BaseMessage] = state["messages"]
    verification = VerifyResult(
        passed=state["verification"]["passed"],
        unsupported=state["verification"]["unsupported"],
    )
    return RunResult(
        answer=final_answer(messages),
        tool_calls=trace_tool_calls(messages),
        verified=verification.passed,
        unsupported=verification.unsupported,
    )
