import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.settings import get_settings
from app.integrations.voyage import rerank as mod
from app.integrations.voyage.embeddings import EmbeddingError

KEY = "pa-secret-key-for-tests"


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "voyage_api_key", SecretStr(KEY))
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)


async def _no_sleep(seconds: float) -> None:
    return None


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_documents_come_back_best_first() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        # Not in score order on purpose.
        data = [
            {"index": 0, "relevance_score": 0.2},
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 42}})

    result = await mod.rerank("câu hỏi", ["a", "b", "c"], client=_client(handler))

    assert result.ranked == [(2, 0.9), (1, 0.5), (0, 0.2)] and result.tokens == 42
    assert seen[0]["query"] == "câu hỏi" and seen[0]["documents"] == ["a", "b", "c"]
    assert seen[0]["model"] == "rerank-2.5"


@pytest.mark.anyio
async def test_rate_limit_is_retried_and_client_error_is_not() -> None:
    calls = 0

    def busy_then_ok(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, json={"detail": "slow down"})
        data = [{"index": 0, "relevance_score": 1.0}]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 1}})

    assert (await mod.rerank("q", ["a"], client=_client(busy_then_ok))).ranked
    assert calls == 3

    bad_calls = 0

    def bad(request: httpx.Request) -> httpx.Response:
        nonlocal bad_calls
        bad_calls += 1
        return httpx.Response(400, json={"detail": "bad input"})

    with pytest.raises(EmbeddingError) as err:
        await mod.rerank("q", ["a"], client=_client(bad))
    assert bad_calls == 1 and KEY not in str(err.value)


@pytest.mark.anyio
async def test_nothing_to_rerank_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    result = await mod.rerank("q", [], client=_client(handler))
    assert result.ranked == [] and result.tokens == 0


@pytest.mark.anyio
async def test_missing_key_and_too_many_documents_fail_early(monkeypatch) -> None:
    with pytest.raises(ValueError):
        await mod.rerank("q", ["a"] * (mod.MAX_DOCUMENTS + 1))
    monkeypatch.setattr(get_settings(), "voyage_api_key", SecretStr(""))
    with pytest.raises(EmbeddingError, match="not set"):
        await mod.rerank("q", ["a"])
