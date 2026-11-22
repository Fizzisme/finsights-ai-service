import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.settings import get_settings
from app.integrations.jev import client as mod

KEY = "ts-secret-key-for-tests"


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "typesafe_api_key", SecretStr(KEY))
    monkeypatch.setattr(mod.asyncio, "sleep", _no_sleep)


async def _no_sleep(seconds: float) -> None:
    return None


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _noul(value: float) -> httpx.Response:
    body = {
        "answers": {"relevant": {"type": "noul", "noul": value}},
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    return httpx.Response(200, json=body)


@pytest.mark.anyio
async def test_pointwise_ranks_by_score_and_ties_keep_input_order() -> None:
    scores = {"a": 0.98, "b": 0.02, "c": 0.98, "d": 0.5}

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert sent["model"] == mod.MODEL
        assert list(sent["state"]) == ["question", "passage"]
        return _noul(scores[sent["state"]["passage"]])

    result = await mod.rerank_pointwise(
        "q", ["a", "b", "c", "d"], client=_client(handler)
    )
    # 0 and 2 tie at 0.98 and stay in input order.
    assert [i for i, _ in result.ranked] == [0, 2, 3, 1]
    assert result.tokens == 4 * 110


@pytest.mark.anyio
async def test_listwise_uses_one_call_and_maps_probabilities_to_passages() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        body = {
            "answers": {
                "best": {
                    "type": "choice",
                    "choice": "p2",
                    "confidence": 0.7,
                    # p1 is left out: a missing option counts as 0.
                    "probabilities": {"p0": 0.1, "p2": 0.9},
                }
            },
            "usage": {"input_tokens": 500, "output_tokens": 30},
        }
        return httpx.Response(200, json=body)

    result = await mod.rerank_listwise("q", ["x", "y", "z"], client=_client(handler))
    assert len(calls) == 1
    assert calls[0]["state"] == "q"
    assert calls[0]["questions"]["best"]["criteria"] == {
        "p0": "x",
        "p1": "y",
        "p2": "z",
    }
    assert [i for i, _ in result.ranked] == [2, 0, 1]
    assert result.tokens == 530


@pytest.mark.anyio
async def test_no_passages_means_no_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call expected")

    assert (await mod.rerank_listwise("q", [], client=_client(handler))).ranked == []
    assert (await mod.rerank_pointwise("q", [], client=_client(handler))).ranked == []


@pytest.mark.anyio
async def test_rate_limit_is_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429) if attempts < 3 else _noul(0.9)

    result = await mod.rerank_pointwise("q", ["a"], client=_client(handler))
    assert attempts == 3
    assert result.ranked == [(0, 0.9)]


@pytest.mark.anyio
async def test_errors_never_contain_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    with pytest.raises(mod.JevError) as error:
        await mod.rerank_pointwise("q", ["a"], client=_client(handler))
    assert "401" in str(error.value)
    assert KEY not in str(error.value)


@pytest.mark.anyio
async def test_missing_key_fails_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_settings(), "typesafe_api_key", SecretStr(""))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call expected")

    with pytest.raises(mod.JevError, match="TYPESAFE_API_KEY"):
        await mod.rerank_pointwise("q", ["a"], client=_client(handler))


@pytest.mark.anyio
async def test_unexpected_answer_shape_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {}})

    with pytest.raises(mod.JevError, match="unexpected"):
        await mod.rerank_pointwise("q", ["a"], client=_client(handler))
