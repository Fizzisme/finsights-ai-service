"""Voyage and Cloudflare Workers AI clients used only by the Dagster ingestion job."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
CLOUDFLARE_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
)
VECTOR_DIMENSIONS = 1024
MAX_TEXTS_PER_BATCH = 64
MAX_CONCURRENT_REQUESTS = 4
RETRIES = 5
RETRY_STATUSES = {429, 500, 502, 503, 504}
CLOUDFLARE_1024D_MODELS = {
    "@cf/baai/bge-m3",
    "@cf/qwen/qwen3-embedding-0.6b",
}


class EmbeddingRequestError(RuntimeError):
    """Provider request failed; error messages never include the API key or article text."""


@dataclass(frozen=True)
class EmbeddedBatch:
    vectors: list[list[float]]
    input_tokens: int | None
    requests: int = 1


def _vectors_from_cloudflare(result: dict[str, Any]) -> list[list[float]]:
    for field in ("response", "data", "embeddings"):
        value = result.get(field)
        if not isinstance(value, list):
            continue
        vectors: list[list[float]] = []
        valid = True
        for item in value:
            if isinstance(item, dict):
                item = item.get("embedding") or item.get("values")
            if not isinstance(item, list) or not item:
                valid = False
                break
            vectors.append(item)
        if valid and vectors:
            return vectors
    raise EmbeddingRequestError(
        "Cloudflare returned an unrecognized embedding response"
    )


def _validate_vectors(vectors: list[list[float]], expected: int, provider: str) -> None:
    if len(vectors) != expected:
        raise EmbeddingRequestError(
            f"{provider} returned {len(vectors)} vectors for {expected} inputs"
        )
    for index, vector in enumerate(vectors):
        if len(vector) != VECTOR_DIMENSIONS:
            raise EmbeddingRequestError(
                f"{provider} vector {index} has {len(vector)} dimensions; expected {VECTOR_DIMENSIONS}"
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise EmbeddingRequestError(
                f"{provider} vector {index} contains a non-finite value"
            )


def _reported_tokens(payload: dict[str, Any], result: dict[str, Any]) -> int | None:
    candidates = [payload.get("usage"), result.get("usage"), result.get("meta")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("total_tokens", "prompt_tokens", "input_tokens", "tokens"):
            value = candidate.get(key)
            if isinstance(value, int) and value >= 0:
                return value
    return None


class EmbeddingClient:
    """Call a model selected by ID; ``@cf/...`` uses Workers AI, other IDs use Voyage."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        account_id: str | None = None,
        timeout_seconds: float = 90,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model.strip()
        self.api_key = api_key
        self.account_id = account_id
        self.timeout_seconds = timeout_seconds
        if not self.model:
            raise ValueError("embedding_model is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if self.is_cloudflare and not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID is required for @cf models")
        if self.is_cloudflare and self.model not in CLOUDFLARE_1024D_MODELS:
            raise ValueError(
                "This pgvector table requires 1024 dimensions; supported Cloudflare models are "
                + ", ".join(sorted(CLOUDFLARE_1024D_MODELS))
            )
        self._owned_client = http_client is None
        self.client = http_client or httpx.Client(timeout=timeout_seconds)

    @property
    def is_cloudflare(self) -> bool:
        return self.model.startswith("@cf/")

    def close(self) -> None:
        if self._owned_client:
            self.client.close()

    def embed(self, texts: Sequence[str]) -> EmbeddedBatch:
        if not texts:
            return EmbeddedBatch(vectors=[], input_tokens=0, requests=0)
        all_vectors: list[list[float]] = []
        total_tokens: int | None = 0
        requests = 0
        batches = [
            list(texts[start : start + MAX_TEXTS_PER_BATCH])
            for start in range(0, len(texts), MAX_TEXTS_PER_BATCH)
        ]
        # Keep the response order deterministic while overlapping network latency. Four
        # concurrent calls are well below Workers AI's current text-embedding rate limit.
        with ThreadPoolExecutor(
            max_workers=min(MAX_CONCURRENT_REQUESTS, len(batches))
        ) as executor:
            for batch, (vectors, tokens) in zip(
                batches, executor.map(self._request, batches), strict=True
            ):
                _validate_vectors(
                    vectors,
                    len(batch),
                    "Cloudflare Workers AI" if self.is_cloudflare else "Voyage",
                )
                all_vectors.extend(
                    [[float(value) for value in vector] for vector in vectors]
                )
                requests += 1
                if tokens is None:
                    total_tokens = None
                elif total_tokens is not None:
                    total_tokens += tokens
        return EmbeddedBatch(
            vectors=all_vectors, input_tokens=total_tokens, requests=requests
        )

    def _request(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        if self.is_cloudflare:
            return self._cloudflare_request(texts)
        return self._voyage_request(texts)

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(RETRIES + 1):
            try:
                response = self.client.post(
                    url, json=body, headers=headers, timeout=self.timeout_seconds
                )
            except httpx.TransportError as exc:
                if attempt >= RETRIES:
                    raise EmbeddingRequestError(
                        f"Embedding provider transport error: {type(exc).__name__}"
                    ) from None
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise EmbeddingRequestError(
                            "Embedding provider returned invalid JSON"
                        ) from None
                    if not isinstance(payload, dict):
                        raise EmbeddingRequestError(
                            "Embedding provider returned an invalid response"
                        )
                    return payload
                if response.status_code not in RETRY_STATUSES or attempt >= RETRIES:
                    # Do not include response text: providers may echo input documents.
                    raise EmbeddingRequestError(
                        f"Embedding provider returned HTTP {response.status_code}"
                    )
            delay = min(30.0, 2**attempt) * (0.5 + random.random() / 2)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _voyage_request(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        payload = self._post(
            VOYAGE_URL,
            {
                "input": texts,
                "model": self.model,
                "input_type": "document",
                "output_dimension": VECTOR_DIMENSIONS,
                "truncation": False,
            },
        )
        if not isinstance(payload.get("data"), list):
            raise EmbeddingRequestError(
                "Voyage returned an unrecognized embedding response"
            )
        ordered = sorted(payload["data"], key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        usage = payload.get("usage") or {}
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        return vectors, tokens if isinstance(tokens, int) else None

    def _cloudflare_request(
        self, texts: list[str]
    ) -> tuple[list[list[float]], int | None]:
        assert self.account_id is not None
        model_path = self.model
        url = CLOUDFLARE_URL.format(account_id=self.account_id, model=model_path)
        body = {"text": texts}
        payload = self._post(url, body)
        if payload.get("success") is False:
            errors = payload.get("errors")
            error_code = None
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                error_code = errors[0].get("code")
            raise EmbeddingRequestError(
                f"Cloudflare Workers AI rejected the request (code={error_code})"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise EmbeddingRequestError(
                "Cloudflare returned an unrecognized embedding response"
            )
        vectors = _vectors_from_cloudflare(result)
        return vectors, _reported_tokens(payload, result)


def embed_texts(
    texts: Sequence[str],
    *,
    model: str,
    api_key: str,
    account_id: str | None = None,
) -> EmbeddedBatch:
    client = EmbeddingClient(model=model, api_key=api_key, account_id=account_id)
    try:
        return client.embed(texts)
    finally:
        client.close()
