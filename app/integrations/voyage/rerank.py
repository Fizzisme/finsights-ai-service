"""Voyage AI reranker over the REST API (docs.voyageai.com/reference/reranker-api).

Embeddings score a question and a text separately and compare the two vectors. A reranker
reads the question and each text together, which tells apart texts that are close in topic but
answer different questions ("lãi suất cho vay" against "lãi suất huy động"). It is slower and
costs per call, so it re-orders a short list that a vector search has already found.

Same rules as the embeddings client: plain httpx, the key only in the Authorization header and
never in a log or error message.
"""

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from app.core.settings import get_settings
from app.integrations.voyage.embeddings import (
    MAX_BACKOFF_SECONDS,
    RETRIES,
    RETRY_STATUSES,
    EmbeddingError,
)

logger = logging.getLogger(__name__)

URL = "https://api.voyageai.com/v1/rerank"
MODEL = "rerank-2.5"
# API limit: at most 1,000 documents, and (query tokens x documents + document tokens) <= 600K.
MAX_DOCUMENTS = 200


@dataclass
class Reranked:
    # (index into the documents sent, relevance score), best first.
    ranked: list[tuple[int, float]]
    tokens: int


async def rerank(
    query: str,
    documents: Sequence[str],
    *,
    model: str = MODEL,
    client: httpx.AsyncClient | None = None,
) -> Reranked:
    """Score `documents` against `query`; every document comes back, best first."""
    key = get_settings().voyage_api_key.get_secret_value()
    if not key:
        raise EmbeddingError("VOYAGE_API_KEY is not set")
    if not documents:
        return Reranked([], 0)
    if len(documents) > MAX_DOCUMENTS:
        raise ValueError(f"at most {MAX_DOCUMENTS} documents per call")
    payload = {"query": query, "documents": list(documents), "model": model}
    owned = client is None
    client = client or httpx.AsyncClient(timeout=60)
    try:
        return await _post(client, key, payload)
    finally:
        if owned:
            await client.aclose()


async def _post(client: httpx.AsyncClient, key: str, payload: dict) -> Reranked:
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(
                URL, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
        except httpx.TransportError as exc:
            if attempt == RETRIES:
                raise EmbeddingError(
                    f"Voyage API unreachable: {type(exc).__name__}"
                ) from None
        else:
            if response.status_code == 200:
                body = response.json()
                ranked = [(d["index"], d["relevance_score"]) for d in body["data"]]
                ranked.sort(key=lambda pair: pair[1], reverse=True)
                return Reranked(ranked, body["usage"]["total_tokens"])
            if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                raise EmbeddingError(
                    f"Voyage API {response.status_code}: {response.text[:200]}"
                )
        wait = min(MAX_BACKOFF_SECONDS, 2**attempt) * (0.5 + random.random() / 2)
        logger.warning("Voyage API busy, retrying in %.1fs", wait)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")
