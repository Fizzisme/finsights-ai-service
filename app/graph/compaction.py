"""Fold old turns of a chat session into a rolling summary (see app/graph/history.py).

A cheap model writes only the summary prose. Every number and date in it must come from the
folded turns or the previous summary: the deterministic verifier checks it, the model gets one
chance to fix a rejected summary, and otherwise folding is abandoned (the caller truncates).
Facts and links of the folded turns are carried over by code, never by the model.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.graph.history import TAG, Context, Digest, history_sources, merge_context
from app.graph.nodes.verifier import feedback_message, verify_answer
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

PROMPT_NAME = "compactor"
MAX_ATTEMPTS = 2
SUMMARY_MAX_CHARS = 1200


@dataclass
class FoldResult:
    context: Context | None  # None: folding failed, the caller falls back to truncation
    folded_turns: list[int] = field(default_factory=list)
    attempts: int = 0
    unsupported: list[str] = field(default_factory=list)


def _too_long(chars: int) -> str:
    return (
        "<verifier_feedback>\n"
        f"The summary has {chars} characters; the limit is {SUMMARY_MAX_CHARS}. "
        "Rewrite it shorter: keep every figure, date and ticker, but give older turns "
        "one short sentence each and drop article details.\n"
        "</verifier_feedback>"
    )


def fold_input(context: Context | None, fold: Sequence[Digest]) -> str:
    earlier = (
        {"through_turn": context["through_turn"], "summary": context["summary"]}
        if context and context.get("through_turn")
        else {}
    )
    turns = [
        {
            "turn": d["turn"],
            "question": d["request"],
            "answer": d["summary"],
            "links": d["facts"].get("links", []),
        }
        for d in fold
    ]
    body = json.dumps({"summary_so_far": earlier, "turns": turns}, ensure_ascii=False)
    body = body.replace(f"</{TAG}>", "")
    return (
        f"Fold <conversation_history> into one new summary.\n<{TAG}>\n{body}\n</{TAG}>"
    )


async def fold_turns(
    model: BaseChatModel,
    context: Context | None,
    fold: Sequence[Digest],
    *,
    today: date,
) -> FoldResult:
    """Fold `fold` into `context`. Never raises for a bad summary; provider errors propagate."""
    result = FoldResult(context=None, folded_turns=[d["turn"] for d in fold])
    if not fold:
        return result
    # The previous summary passed this same check, and its facts stay sources.
    sources = [*history_sources(context, fold)]
    if context and context.get("summary"):
        sources.append(context["summary"])
    messages: list[BaseMessage] = [
        SystemMessage(load_prompt(PROMPT_NAME)),
        HumanMessage(fold_input(context, fold)),
    ]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result.attempts = attempt
        reply = await model.ainvoke(messages)
        summary = StrOutputParser().invoke(reply).strip()
        if not summary:
            break
        if len(summary) > SUMMARY_MAX_CHARS:
            # Never cut it: the newest turns sit at the end and would be lost.
            result.unsupported = [f"too long: {len(summary)} chars"]
            messages += [AIMessage(summary), HumanMessage(_too_long(len(summary)))]
            continue
        check = verify_answer(
            summary, [], question="", today=today, history_sources=sources
        )
        if check.passed:
            result.context = merge_context(context, fold, summary)
            result.unsupported = []
            return result
        result.unsupported = check.unsupported
        messages += [AIMessage(summary), HumanMessage(feedback_message(check))]
    logger.warning(
        "compaction rejected after %d attempts: %s", result.attempts, result.unsupported
    )
    return result


def trace(result: FoldResult) -> dict[str, Any]:
    """What a turn records about the fold it ran."""
    return {
        "folded_turns": result.folded_turns,
        "ok": result.context is not None,
        "attempts": result.attempts,
        "unsupported": result.unsupported,
    }
