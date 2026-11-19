"""Online workflow: agent loop -> verifier -> (bounded retry) -> answer.

START -> agent -> verifier --ok / retries exhausted--> END
            ^         |
            +---------+  (feedback message, at most `max_retries` times)
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from app.graph.history import fit, history_block, history_sources
from app.graph.nodes.agent import (
    RunResult,
    build_agent,
    build_user_message,
    final_answer,
    token_usage,
    trace_tool_calls,
)
from app.graph.nodes.verifier import VerifyResult, feedback_message, verify_answer


class RunState(MessagesState):
    retries: int
    verification: dict[str, Any]


# Receives progress events of a run: (name, data); see `progress_events`.
EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class TokenTally:
    """Token usage of a run so far, filled while it runs: a run that times out or fails
    still reports what it spent (the token quota charges it)."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    _seen: set[str] = field(default_factory=set, repr=False)

    def add(self, updates: Mapping[str, Any]) -> None:
        """Count the model replies in one update of the agent loop."""
        for update in updates.values():
            if not isinstance(update, dict):
                continue
            for message in update.get("messages", []):
                if not isinstance(message, AIMessage) or not message.usage_metadata:
                    continue
                if message.id:
                    if message.id in self._seen:
                        continue  # a middleware may hand the same reply back
                    self._seen.add(message.id)
                usage = message.usage_metadata
                self.input += usage.get("input_tokens", 0)
                self.output += usage.get("output_tokens", 0)
                self.cache_read += usage.get("input_token_details", {}).get(
                    "cache_read", 0
                )


def progress_events(
    namespace: Sequence[str], updates: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Progress events of one graph update, for a client waiting on the answer.

    Inside the agent loop: `tool_start` per tool call, `tool_end` per tool result. In the
    workflow: `verifying` once the agent has an answer, `retrying` when the verifier sends
    it back. The answer itself is never streamed: it may still be rejected.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    for node, update in updates.items():
        if not isinstance(update, dict):
            continue  # middleware steps that change nothing
        if namespace:
            for message in update.get("messages", []):
                if isinstance(message, AIMessage):
                    events.extend(
                        ("tool_start", {"name": call["name"], "args": call["args"]})
                        for call in message.tool_calls
                    )
                elif isinstance(message, ToolMessage):
                    events.append(
                        (
                            "tool_end",
                            {"name": message.name, "ok": message.status != "error"},
                        )
                    )
        elif node == "agent":
            events.append(("verifying", {}))
        elif node == "verifier" and update.get("verification", {}).get("retry"):
            events.append(("retrying", {}))
    return events


def build_workflow(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    question: str,
    today: date,
    max_steps: int,
    max_retries: int,
    max_tool_calls: int | None = None,
    sources: Sequence[str] = (),
):
    def verifier(state: RunState) -> dict[str, Any]:
        messages: Sequence[BaseMessage] = state["messages"]
        result = verify_answer(
            final_answer(messages),
            messages,
            question=question,
            today=today,
            history_sources=sources,
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
    history: Sequence[dict[str, Any]] = (),
    context: dict[str, Any] | None = None,
    on_event: EventSink | None = None,
    tally: TokenTally | None = None,
) -> RunResult:
    """One turn of a chat session.

    `history` holds digests of earlier turns; `context` is the rolling summary of the turns
    folded so far (app/graph/history.py). Turns that still do not fit are truncated.
    `on_event` receives the run's progress (see `progress_events`); `tally` the tokens it
    spends, as they are spent.
    """
    today = today or date.today()
    turns = fit(context, history)
    graph = build_workflow(
        model,
        tools,
        question=question,
        today=today,
        max_steps=max_steps,
        max_retries=max_retries,
        max_tool_calls=max_tool_calls,
        sources=history_sources(context, turns),
    )
    state: Mapping[str, Any] = {}
    async for part in graph.astream(
        {
            "messages": [
                build_user_message(question, today, history_block(context, turns))
            ],
            "retries": 0,
            "verification": {},
        },
        stream_mode=["updates", "values"],
        subgraphs=True,
        version="v2",
    ):
        if part["type"] == "values":
            if not part["ns"]:
                state = part["data"]  # the workflow's own state, not the agent's
        elif part["type"] == "updates":
            # Only the agent loop's own updates: the workflow's `agent` update repeats
            # every message of the loop.
            if tally is not None and part["ns"]:
                tally.add(part["data"])
            if on_event is not None:
                for name, data in progress_events(part["ns"], part["data"]):
                    await on_event(name, data)
    messages: Sequence[BaseMessage] = state["messages"]
    verification = VerifyResult(
        passed=state["verification"]["passed"],
        unsupported=state["verification"]["unsupported"],
    )
    input_tokens, output_tokens, cache_read_tokens = token_usage(messages)
    return RunResult(
        answer=final_answer(messages),
        tool_calls=trace_tool_calls(messages),
        verified=verification.passed,
        unsupported=verification.unsupported,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )
