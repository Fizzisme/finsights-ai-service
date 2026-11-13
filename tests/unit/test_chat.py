"""Provider configuration and HTTP behavior without external API calls."""

from unittest.mock import AsyncMock

import httpx2
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import AuthenticationError

from app.api.routes import chat
from app.core.settings import Settings
from app.integrations.llm.catalog import TIERS, load_catalog
from app.integrations.llm.config import Capabilities, ModelConfig, ModelConfigError
from app.integrations.llm.factory import (
    LLMConfigurationError,
    build_chat_model,
    get_chat_model,
)
from app.main import app


@pytest.fixture
def client(monkeypatch):
    model = AsyncMock(spec=BaseChatModel)
    model.ainvoke.return_value = AIMessage(content="Hello from the test model")
    app.dependency_overrides[chat.configured_model] = lambda: model
    monkeypatch.setattr(
        chat, "get_settings", lambda: Settings.model_construct(llm_timeout_seconds=1)
    )
    with TestClient(app) as test_client:
        yield test_client, model
    app.dependency_overrides.clear()


def test_chat_returns_model_text(client):
    http, model = client
    response = http.post("/chat", json={"message": "  Hello  "})
    assert response.status_code == 200
    assert response.json() == {"answer": "Hello from the test model"}
    assert model.ainvoke.call_args.args[0][0].content == "Hello"


@pytest.mark.parametrize("message", ["", "   ", "x" * 16001])
def test_invalid_messages_do_not_call_provider(client, message):
    http, model = client
    assert http.post("/chat", json={"message": message}).status_code == 422
    model.ainvoke.assert_not_called()


def test_provider_cannot_be_overridden_by_request(client):
    http, model = client
    response = http.post("/chat", json={"message": "Hello", "provider": "other"})
    assert response.status_code == 422
    model.ainvoke.assert_not_called()


def test_unconfigured_chat_does_not_break_health(monkeypatch):
    def missing_config(profile, tier):
        raise LLMConfigurationError("Set OPENAI_MODEL.")

    monkeypatch.setattr(chat, "get_chat_model", missing_config)
    with TestClient(app) as http:
        assert http.get("/healthz").json() == {"status": "ok"}
        response = http.post("/chat?profile=openai", json={"message": "Hello"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Set OPENAI_MODEL."}


def test_unknown_profile_is_service_unavailable():
    with TestClient(app) as http:
        response = http.post("/chat?profile=nope", json={"message": "Hello"})
        assert response.status_code == 503
        assert "nope" in response.json()["detail"]


def test_profile_is_required_and_tier_is_validated():
    with TestClient(app) as http:
        assert http.post("/chat", json={"message": "Hello"}).status_code == 422
        response = http.post(
            "/chat?profile=openai&tier=huge", json={"message": "Hello"}
        )
        assert response.status_code == 422


def test_profile_and_tier_reach_the_factory(monkeypatch):
    seen = {}
    model = AsyncMock(spec=BaseChatModel)
    model.ainvoke.return_value = AIMessage(content="ok")

    def fake_get_chat_model(profile, tier):
        seen["args"] = (profile, tier)
        return model

    monkeypatch.setattr(chat, "get_chat_model", fake_get_chat_model)
    with TestClient(app) as http:
        response = http.post(
            "/chat?profile=deepseek&tier=cheap", json={"message": "Hello"}
        )
    assert response.status_code == 200
    assert seen["args"] == ("deepseek", "cheap")


def test_provider_errors_do_not_expose_secrets(client):
    http, model = client
    model.ainvoke.side_effect = AuthenticationError(
        "secret-api-key and private prompt",
        response=httpx2.Response(
            401, request=httpx2.Request("POST", "https://example.com")
        ),
        body=None,
    )
    response = http.post("/chat", json={"message": "Hello"})
    assert response.status_code == 502
    assert "secret-api-key" not in response.text
    assert "private prompt" not in response.text


def test_timeout_returns_gateway_timeout(client):
    http, model = client
    model.ainvoke.side_effect = TimeoutError
    assert http.post("/chat", json={"message": "Hello"}).status_code == 504


def test_text_blocks_exclude_reasoning(client):
    http, model = client
    model.ainvoke.return_value = AIMessage(
        content=[
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "Answer"},
        ]
    )
    response = http.post("/chat", json={"message": "Hello"})
    assert response.json() == {"answer": "Answer"}


def test_empty_response_returns_bad_gateway(client):
    http, model = client
    model.ainvoke.return_value = AIMessage(content="")
    assert http.post("/chat", json={"message": "Hello"}).status_code == 502


@pytest.fixture
def provider_env(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(name, "fake-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-gpt")


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    ("profile", "expected_class"),
    [
        ("anthropic", "ChatAnthropic"),
        ("openai", "ChatOpenAI"),
        ("deepseek", "ChatOpenAI"),
    ],
)
def test_every_profile_and_tier_builds(provider_env, profile, tier, expected_class):
    assert type(get_chat_model(profile, tier)).__name__ == expected_class


def test_openai_profile_uses_responses_api(provider_env):
    model = get_chat_model("openai", "standard")
    assert isinstance(model, ChatOpenAI)
    assert model.use_responses_api is True
    assert model.model_name == "test-gpt"


def test_deepseek_profile_uses_chat_completions_at_its_base_url(provider_env):
    model = get_chat_model("deepseek", "cheap")
    assert isinstance(model, ChatOpenAI)
    assert not model.use_responses_api
    assert str(model.openai_api_base) == "https://api.deepseek.com"
    assert model.model_name == "deepseek-v4-flash"


def test_max_retries_excludes_the_first_attempt(provider_env):
    config = load_catalog().select("anthropic", "cheap")
    model = build_chat_model(config)
    assert model.max_retries == config.max_attempts - 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(("supported", "expected"), [(True, "high"), (False, None)])
def test_effort_is_sent_only_when_the_model_accepts_it(
    provider_env, supported, expected
):
    config = load_catalog().select("openai", "standard", effort="high")
    config = config.model_copy(
        update={"capabilities": Capabilities(reasoning_effort=supported)}
    )
    model = build_chat_model(config)
    assert isinstance(model, ChatOpenAI)
    assert model.reasoning_effort == expected


def test_missing_key_is_rejected(provider_env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ModelConfigError, match="ANTHROPIC_API_KEY"):
        get_chat_model("anthropic", "cheap")


def test_missing_openai_model_is_rejected(provider_env, monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL")
    with pytest.raises(ModelConfigError, match="OPENAI_MODEL"):
        get_chat_model("openai", "standard")


def test_unsupported_dialect_is_rejected(provider_env):
    config = ModelConfig.model_construct(
        id="x", dialect="unknown", model="m", api_key_ref="env:ANTHROPIC_API_KEY"
    )
    with pytest.raises(LLMConfigurationError, match="unknown"):
        build_chat_model(config)
