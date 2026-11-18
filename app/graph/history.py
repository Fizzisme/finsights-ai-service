"""Conversation history: what earlier turns of a chat session left for the next one.

Each turn leaves a digest built by code (not by a model): the question and the answer, both
clipped, plus the numbers, dates and links of the answer when the verifier passed it. A new
turn reads the digests of the last `KEEP_RECENT_TURNS` turns (sliding window) within a fixed
budget; past it the oldest digests are dropped, so a turn never fails because of its history.

The digests' facts are sources for the verifier, so a follow-up may reuse a figure a previous
turn already checked. Unverified turns leave no facts, so an unsupported figure cannot spread.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from app.graph.nodes.verifier import facts_of

KEEP_RECENT_TURNS = 3
HISTORY_TOKENS = 2000
REQUEST_CHARS = 400
SUMMARY_CHARS = 600
MAX_LINKS = 10
# No tokenizer here: Vietnamese costs more tokens per character than English, so count high.
CHARS_PER_TOKEN = 3
TAG = "conversation_history"

# A markdown link ends at ")"; a bare URL at whitespace. Trailing punctuation is not part of it.
_LINK = re.compile(r"https?://[^\s)\]>\"']+")


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_digest(question: str, answer: str, *, verified: bool) -> dict[str, Any]:
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


def _view(window: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The window as the model reads it: no internal fact lists, only what was said and linked."""
    return [
        {
            "turn": d["turn"],
            "question": d["request"],
            "answer": d["summary"],
            "links": d["facts"].get("links", []),
        }
        for d in window
    ]


def estimate_tokens(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False)) // CHARS_PER_TOKEN + 1


def select_window(digests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Digests (each with `turn`) a new turn reads: the most recent ones within the budget."""
    window = sorted(digests, key=lambda d: d["turn"])[-KEEP_RECENT_TURNS:]
    while len(window) > 1 and estimate_tokens(_view(window)) > HISTORY_TOKENS:
        window.pop(0)
    return window


def history_block(window: Sequence[dict[str, Any]]) -> str:
    """The window as a data block for the user message; empty when there is no history."""
    if not window:
        return ""
    # Earlier questions were typed by the user: they must not close the tag themselves.
    body = json.dumps(_view(window), ensure_ascii=False).replace(f"</{TAG}>", "")
    return f"<{TAG}>\n{body}\n</{TAG}>"


def history_sources(window: Sequence[dict[str, Any]]) -> list[str]:
    """Texts the verifier accepts as sources: earlier questions and verified facts."""
    sources: list[str] = []
    for d in window:
        facts = d["facts"]
        sources.append(d["request"])
        sources.append(
            " ".join(
                [
                    *facts.get("numbers", []),
                    *facts.get("dates", []),
                    *facts.get("day_months", []),
                ]
            )
        )
    return sources
