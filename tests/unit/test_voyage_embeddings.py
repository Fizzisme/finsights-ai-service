import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.settings import get_settings
from app.integrations.voyage import embeddings as mod

KEY = "pa-secret-key-for-tests"


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "voyage_api_key", SecretStr(KEY))
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)


async def _no_sleep(seconds: float) -> None:
    return None


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    # Out of order on purpose: results must follow `index`, not response order.
    data = [
        {"object": "embedding", "embedding": [float(i), 1.0], "index": i}
        for i in reversed(range(len(texts)))
    ]
    return httpx.Response(
        200, json={"data": data, "usage": {"total_tokens": 3 * len(texts)}}
    )


def test_batches_respect_text_and_token_limits() -> None:
    assert [len(b) for b in mod.batches(["a"] * 300)] == [128, 128, 44]
    long = "x" * int(mod.MAX_TOKENS_PER_BATCH * mod.CHARS_PER_TOKEN * 0.6)
    assert [len(b) for b in mod.batches([long, long, long])] == [1, 1, 1]
    assert mod.batches([]) == []


@pytest.mark.anyio
async def test_embeds_in_order_across_batches() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return _ok(request)

    result = await mod.embed(["t"] * 130, "document", client=_client(handler))

    assert len(result.vectors) == 130 and result.tokens == 390
    assert result.vectors[0] == [0.0, 1.0] and result.vectors[129] == [1.0, 1.0]
    assert [len(b["input"]) for b in seen] == [128, 2]
    assert seen[0]["input_type"] == "document" and seen[0]["model"] == "voyage-4"


@pytest.mark.anyio
async def test_rate_limit_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(429, json={"detail": "slow down"})
            if calls < 3
            else _ok(request)
        )

    result = await mod.embed(["q"], "query", client=_client(handler))
    assert calls == 3 and len(result.vectors) == 1


@pytest.mark.anyio
async def test_client_error_is_not_retried_and_hides_the_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"detail": "bad input"})

    with pytest.raises(mod.EmbeddingError) as err:
        await mod.embed(["q"], "query", client=_client(handler))
    assert calls == 1
    assert "400" in str(err.value) and KEY not in str(err.value)


@pytest.mark.anyio
async def test_missing_key_fails_before_any_request(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "voyage_api_key", SecretStr(""))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(mod.EmbeddingError, match="not set"):
        await mod.embed(["q"], "query", client=_client(handler))
