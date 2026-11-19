"""Voyage AI text embeddings over the REST API (docs.voyageai.com/reference/embeddings-api).

Plain httpx, no SDK: one endpoint. The API key only travels in the Authorization header and
is never logged or put in an error message.

`input_type` matters for retrieval: Voyage prepends a different instruction to queries and
to documents, so embed questions with "query" and the searched texts with "document".
"""

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import httpx

from app.core.settings import get_settings

logger = logging.getLogger(__name__)

URL = "https://api.voyageai.com/v1/embeddings"
InputType = Literal["query", "document"]

# API limits per request: at most 1,000 texts, and 320K tokens for voyage-4. Batches stay
# far below both, so one slow or rate-limited request never holds much work.
MAX_TEXTS_PER_BATCH = 128
MAX_TOKENS_PER_BATCH = 60_000
# No tokenizer here: Vietnamese runs about 3 characters per token, so count high.
CHARS_PER_TOKEN = 2.5
RETRIES = 6
MAX_BACKOFF_SECONDS = 60
RETRY_STATUSES = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """Voyage did not return embeddings."""


@dataclass
class Embeddings:
    vectors: list[list[float]]
    tokens: int


def batches(texts: Sequence[str]) -> list[list[str]]:
    """Split texts into requests under the per-request text and (estimated) token limits."""
    result: list[list[str]] = [[]]
    budget = 0.0
    for text in texts:
        cost = len(text) / CHARS_PER_TOKEN + 1
        if result[-1] and (
            len(result[-1]) >= MAX_TEXTS_PER_BATCH
            or budget + cost > MAX_TOKENS_PER_BATCH
        ):
            result.append([])
            budget = 0.0
        result[-1].append(text)
        budget += cost
    return [b for b in result if b]


async def _request(
    client: httpx.AsyncClient, key: str, payload: dict
) -> tuple[list[list[float]], int]:
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
                data = sorted(body["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data], body["usage"]["total_tokens"]
            if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                # Voyage error bodies carry a reason, never the key.
                raise EmbeddingError(
                    f"Voyage API {response.status_code}: {response.text[:200]}"
                )
        wait = min(MAX_BACKOFF_SECONDS, 2**attempt) * (0.5 + random.random() / 2)
        logger.warning("Voyage API busy, retrying in %.1fs", wait)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def embed(
    texts: Sequence[str],
    input_type: InputType,
    *,
    client: httpx.AsyncClient | None = None,
) -> Embeddings:
    """Embed `texts` in order, in as many requests as the limits need."""
    settings = get_settings()
    key = settings.voyage_api_key.get_secret_value()
    if not key:
        raise EmbeddingError("VOYAGE_API_KEY is not set")
    owned = client is None
    client = client or httpx.AsyncClient(timeout=60)
    vectors: list[list[float]] = []
    tokens = 0
    try:
        for batch in batches(texts):
            got, used = await _request(
                client,
                key,
                {
                    "input": batch,
                    "model": settings.embedding_model,
                    "input_type": input_type,
                    "output_dimension": settings.embedding_dimensions,
                },
            )
            if len(got) != len(batch):
                raise EmbeddingError(
                    f"Voyage returned {len(got)} of {len(batch)} vectors"
                )
            vectors.extend(got)
            tokens += used
    finally:
        if owned:
            await client.aclose()
    return Embeddings(vectors, tokens)
