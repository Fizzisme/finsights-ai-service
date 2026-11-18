"""Conversation history: what earlier turns of a chat session left for the next one.

Each turn leaves a digest built by code (not by a model): the question and the answer, both
clipped, plus the numbers, dates and links of the answer when the verifier passed it.

A new turn reads, within `HISTORY_TOKENS`:
- `earlier`: a rolling summary of old turns (the checkpoint, see app/graph/compaction.py);
- `turns`: the digests of the turns the summary does not cover yet, verbatim.

When that view outgrows the budget, the turns before the last `KEEP_RECENT_TURNS` are folded
into the summary by a cheap model. The facts of folded turns are carried over by code, never
rewritten by the model. If folding fails, the oldest digests are dropped instead (truncation),
so a turn never fails because of its history.

The facts are sources for the verifier, so a follow-up may reuse a figure a previous turn
already checked. Unverified turns leave no facts, so an unsupported figure cannot spread.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from app.graph.nodes.verifier import facts_of

KEEP_RECENT_TURNS = 3
HISTORY_TOKENS = 2000
# Unfolded turns loaded per request; more only happens if folding keeps failing.
MAX_UNFOLDED_TURNS = 20
REQUEST_CHARS = 400
SUMMARY_CHARS = 600
MAX_LINKS = 10
# A rolling summary carries at most this many facts/links (newest kept).
MAX_CONTEXT_FACTS = 200
MAX_CONTEXT_LINKS = 20
# No tokenizer here: Vietnamese costs more tokens per character than English, so count high.
CHARS_PER_TOKEN = 3
TAG = "conversation_history"
FACT_KINDS = ("numbers", "dates", "day_months")

# A markdown link ends at ")"; a bare URL at whitespace. Trailing punctuation is not part of it.
_LINK = re.compile(r"https?://[^\s)\]>\"']+")

Digest = dict[str, Any]
Context = dict[str, Any]


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_digest(question: str, answer: str, *, verified: bool) -> Digest:
    """What later turns keep of this turn."""
    facts: dict[str, list[str]] = {
        "numbers": [],
        "dates": [],
        "day_months": [],
        "links": [],
    }
    if verified:
        facts.update(facts_of(answer))
        links = [link.rstrip(".,;:") for link in _LINK.findall(answer)]
        facts["links"] = list(dict.fromkeys(links))[:MAX_LINKS]
    return {
        "request": _cut(question, REQUEST_CHARS),
        "summary": _cut(answer, SUMMARY_CHARS),
        "facts": facts,
    }


def unfolded(context: Context | None, digests: Sequence[Digest]) -> list[Digest]:
    """Digests the rolling summary does not cover yet, oldest first."""
    through = (context or {}).get("through_turn", 0)
    return sorted((d for d in digests if d["turn"] > through), key=lambda d: d["turn"])


def _view(context: Context | None, turns: Sequence[Digest]) -> dict[str, Any]:
    """What the model reads: the summary and links, then each turn as said (no fact lists)."""
    view: dict[str, Any] = {}
    if context and context.get("through_turn"):
        view["earlier"] = {
            "through_turn": context["through_turn"],
            "summary": context["summary"],
            "links": context.get("links", []),
        }
    view["turns"] = [
        {
            "turn": d["turn"],
            "question": d["request"],
            "answer": d["summary"],
            "links": d["facts"].get("links", []),
        }
        for d in turns
    ]
    return view


def estimate_tokens(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False)) // CHARS_PER_TOKEN + 1


def to_fold(context: Context | None, digests: Sequence[Digest]) -> list[Digest]:
    """Turns to fold into the summary: those before the window, once the view is over budget."""
    turns = unfolded(context, digests)
    if estimate_tokens(_view(context, turns)) <= HISTORY_TOKENS:
        return []
    return turns[:-KEEP_RECENT_TURNS]


def fit(context: Context | None, digests: Sequence[Digest]) -> list[Digest]:
    """Truncation fallback: unfolded turns that fit the budget, dropping the oldest first.

    The newest turn always stays.
    """
    turns = unfolded(context, digests)
    while len(turns) > 1 and estimate_tokens(_view(context, turns)) > HISTORY_TOKENS:
        turns.pop(0)
    return turns


def merge_context(
    context: Context | None, fold: Sequence[Digest], summary: str
) -> Context:
    """The checkpoint after `summary` folded `fold` into `context`.

    Facts and links come from the folded digests themselves, so the model cannot alter them.
    """
    old = context or {}
    facts: dict[str, list[str]] = {}
    for kind in FACT_KINDS:
        values = [*old.get("facts", {}).get(kind, [])]
        for d in fold:
            values.extend(d["facts"].get(kind, []))
        facts[kind] = list(dict.fromkeys(values))[-MAX_CONTEXT_FACTS:]
    links = [*old.get("links", [])]
    for d in fold:
        links.extend(d["facts"].get("links", []))
    return {
        "through_turn": fold[-1]["turn"],
        "summary": summary,
        "facts": facts,
        "links": list(dict.fromkeys(links))[-MAX_CONTEXT_LINKS:],
    }


def history_block(context: Context | None, turns: Sequence[Digest]) -> str:
    """The history as a data block for the user message; empty when there is none."""
    view = _view(context, turns)
    if not view["turns"] and "earlier" not in view:
        return ""
    # Earlier questions were typed by the user: they must not close the tag themselves.
    body = json.dumps(view, ensure_ascii=False).replace(f"</{TAG}>", "")
    return f"<{TAG}>\n{body}\n</{TAG}>"


def _facts_text(facts: dict[str, list[str]]) -> str:
    return " ".join(v for kind in FACT_KINDS for v in facts.get(kind, []))


def history_sources(context: Context | None, turns: Sequence[Digest]) -> list[str]:
    """Texts the verifier accepts as sources: earlier questions, verified facts and the turn
    numbers themselves ("Lượt 2: ...")."""
    sources: list[str] = []
    turn_numbers = [d["turn"] for d in turns]
    if context and context.get("through_turn"):
        sources.append(_facts_text(context.get("facts", {})))
        turn_numbers.extend(range(1, context["through_turn"] + 1))
    for d in turns:
        sources.append(d["request"])
        sources.append(_facts_text(d["facts"]))
    if turn_numbers:
        sources.append(" ".join(str(n) for n in sorted(set(turn_numbers))))
    return sources
