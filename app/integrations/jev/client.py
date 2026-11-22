"""Jev (TypeSafe System One) over the REST API, used here to re-rank retrieved passages.

Jev is a decision model, not an LLM: one request carries a `state` and typed questions, and the
answer is probabilities. It has no rerank endpoint, so a rerank is built from two question types,
and the eval compares them:

- `rerank_pointwise`: one `noul` (a yes/no probability) per passage, "does this passage answer the
  question?". Many calls, each passage judged alone, but scores sit near 0 or 1, so passages tie.
- `rerank_listwise`: one `choice` whose options are the passages. One call, passages compared with
  each other, but the probabilities add up to 1 and the request is large.

Same rules as the other clients: plain httpx, the key only in the Authorization header and never
in a log or error message. Instructions are English, the language Jev reads best.
"""

import asyncio
import logging
import random
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.settings import get_settings
from app.integrations.voyage.rerank import Reranked

logger = logging.getLogger(__name__)

URL = "https://api.typesafe.ai/v1/systemone"
# Pinned: thresholds are tuned against a version, and the latest alias moves between releases.
MODEL = "jev-1.14.0"
# Rate limit, TypeSafe overload, and transient server errors.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
RETRIES = 4
MAX_BACKOFF_SECONDS = 20.0
# Passages judged at the same time by `rerank_pointwise`.
CONCURRENCY = 16

_RELEVANT = {
    "type": "noul",
    "instructions": "Does the passage contain information that answers the question?",
    "criteria": {
        "true": "The passage states the fact, figure or event the question asks about.",
        "false": "The passage is about something else, or only shares a topic or a name with "
        "the question without answering it.",
    },
}
_BEST = "Which passage best answers the question?"


class JevError(RuntimeError):
    pass


async def system_one(
    client: httpx.AsyncClient, state: Any, questions: dict[str, Any]
) -> dict[str, Any]:
    """One System One call, retried on rate limits and overload; returns the parsed body."""
    key = get_settings().typesafe_api_key.get_secret_value()
    if not key:
        raise JevError("TYPESAFE_API_KEY is not set")
    payload = {"model": MODEL, "state": state, "questions": questions}
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(
                URL, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
        except httpx.TransportError as exc:
            if attempt == RETRIES:
                raise JevError(f"Jev API unreachable: {type(exc).__name__}") from None
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                raise JevError(f"Jev API {response.status_code}: {response.text[:200]}")
        wait = min(MAX_BACKOFF_SECONDS, 2**attempt) * (0.5 + random.random() / 2)
        logger.warning("Jev API busy, retrying in %.1fs", wait)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def _tokens(body: dict[str, Any]) -> int:
    usage = body.get("usage") or {}
    return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))


async def rerank_pointwise(
    query: str,
    passages: Sequence[str],
    *,
    client: httpx.AsyncClient | None = None,
    concurrency: int = CONCURRENCY,
) -> Reranked:
    """Score each passage on its own. Passages with equal scores keep the order they came in."""
    owned = client is None
    client = client or httpx.AsyncClient(timeout=30)
    limit = asyncio.Semaphore(concurrency)

    async def judge(passage: str) -> tuple[float, int]:
        async with limit:
            # Field order is fixed: Jev reads the state in the same order every time.
            body = await system_one(
                client,
                {"question": query, "passage": passage},
                {"relevant": _RELEVANT},
            )
        try:
            return float(body["answers"]["relevant"]["noul"]), _tokens(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(f"unexpected Jev answer: {exc!r}") from None

    try:
        judged = await asyncio.gather(*(judge(p) for p in passages))
    finally:
        if owned:
            await client.aclose()
    ranked = sorted(
        ((i, score) for i, (score, _) in enumerate(judged)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return Reranked(ranked, sum(tokens for _, tokens in judged))


async def rerank_listwise(
    query: str,
    passages: Sequence[str],
    *,
    client: httpx.AsyncClient | None = None,
) -> Reranked:
    """One call with every passage as an option; the probability of an option is its score."""
    if not passages:
        return Reranked([], 0)
    owned = client is None
    client = client or httpx.AsyncClient(timeout=120)
    options = {f"p{i}": passage for i, passage in enumerate(passages)}
    try:
        body = await system_one(
            client,
            query,
            {"best": {"type": "choice", "instructions": _BEST, "criteria": options}},
        )
    finally:
        if owned:
            await client.aclose()
    try:
        probabilities = body["answers"]["best"]["probabilities"]
        scores = [float(probabilities.get(name, 0.0)) for name in options]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise JevError(f"unexpected Jev answer: {exc!r}") from None
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return Reranked(ranked, _tokens(body))
