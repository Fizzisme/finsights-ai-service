from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.api.routes import chat, sessions
from app.graph.nodes.agent import AgentRunError, RunResult, ToolCallRecord
from app.main import app
from app.storage.mysql import chat_store

USER = "11111111-1111-4111-8111-111111111111"
SID = "22222222-2222-4222-8222-222222222222"
HEADERS = {"X-User-Id": USER}
NOW = datetime(2026, 11, 18, 6, 0)
SESSION = {
    "id": 7,
    "session_uid": SID,
    "title": None,
    "created_at": NOW,
    "updated_at": NOW,
}


class FakeStore:
    """In-memory stand-in for chat_store, recording what the route saves."""

    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {
            (USER, SID): dict(SESSION)
        }
        self.saved: list[dict[str, Any]] = []
        self.digests: list[dict[str, Any]] = []
        self.context: dict[str, Any] | None = None
        self.fail_save: Exception | None = None

    async def get_session(self, user: str, uid: str) -> dict[str, Any] | None:
        return self.sessions.get((user, uid))

    async def latest_context(self, session_id: int) -> dict[str, Any] | None:
        return self.context

    async def digests_after(
        self, session_id: int, after_turn: int, limit: int
    ) -> list[dict[str, Any]]:
        return [d for d in self.digests if d["turn"] > after_turn][-limit:]

    async def save_turn(self, session: dict[str, Any], **turn: Any) -> int:
        if self.fail_save:
            raise self.fail_save
        self.saved.append(turn)
        return len(self.saved)

    async def create_session(self, user: str) -> dict[str, Any]:
        return {**SESSION, "session_uid": SID}

    async def delete_session(self, user: str, uid: str) -> bool:
        return self.sessions.pop((user, uid), None) is not None


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore()
    for name in (
        "get_session",
        "latest_context",
        "digests_after",
        "save_turn",
        "create_session",
        "delete_session",
    ):
        monkeypatch.setattr(chat_store, name, getattr(fake, name))
    return fake


@pytest.fixture
def client():
    app.dependency_overrides[chat.configured_model] = lambda: object()
    # No compaction unless a test turns it on.
    app.dependency_overrides[sessions.compactor_model] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _stub_agent(monkeypatch, *, result=None, error=None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake(model, tools, question, **kwargs):
        calls.append({"question": question, **kwargs})
        if error:
            raise error
        return result

    monkeypatch.setattr(sessions, "run_agent", fake)
    return calls


def _ask(client, question: str = "Còn TCB?", headers=HEADERS, sid: str = SID):
    return client.post(
        f"/sessions/{sid}/messages?profile=openai",
        json={"question": question},
        headers=headers,
    )


@pytest.mark.parametrize("headers", [{}, {"X-User-Id": "not-a-uuid"}])
def test_requests_without_a_valid_user_are_rejected(client, store, headers):
    assert _ask(client, headers=headers).status_code == 401
    assert client.get("/sessions", headers=headers).status_code == 401


def test_other_users_session_is_not_found(client, store, monkeypatch):
    _stub_agent(monkeypatch, result=RunResult("x"))
    other = {"X-User-Id": "33333333-3333-4333-8333-333333333333"}
    assert _ask(client, headers=other).status_code == 404
    assert _ask(client, sid="not-a-uuid").status_code == 404
    assert store.saved == []


def test_turn_uses_history_and_saves_digest(client, store, monkeypatch):
    store.digests = [
        {"turn": 1, "request": "Tin VCB?", "summary": "Có 2 tin.", "facts": {}}
    ]
    calls = _stub_agent(
        monkeypatch,
        result=RunResult(
            "TCB có 297 bài.", [ToolCallRecord("count_news", {"symbol": "TCB"}, True)]
        ),
    )
    response = _ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["turn"] == 1 and body["session_uid"] == SID
    assert body["answer"] == "TCB có 297 bài."
    assert calls[0]["history"] == store.digests

    saved = store.saved[0]
    assert saved["status"] == chat_store.COMPLETE
    trace = saved["ai_trace"]
    assert trace["verified"] is True
    assert "297" in trace["digest"]["facts"]["numbers"]
    assert trace["tool_calls"] == [
        {"name": "count_news", "args": {"symbol": "TCB"}, "ok": True}
    ]


def test_failed_run_is_saved_and_reported(client, store, monkeypatch):
    _stub_agent(monkeypatch, error=AgentRunError("x"))
    response = _ask(client)

    assert response.status_code == 502
    saved = store.saved[0]
    assert saved["status"] == chat_store.FAILED and saved["error_code"] == "no_answer"
    assert saved["ai_trace"]["digest"]["facts"]["numbers"] == []


def test_save_failure_still_returns_the_answer(client, store, monkeypatch):
    _stub_agent(monkeypatch, result=RunResult("Có 2 tin."))
    store.fail_save = OperationalError("INSERT", {}, Exception("db down"))
    response = _ask(client)

    assert response.status_code == 200
    assert response.json()["turn"] is None and response.json()["answer"] == "Có 2 tin."


def test_concurrent_turn_is_a_conflict(client, store, monkeypatch):
    _stub_agent(monkeypatch, result=RunResult("Có 2 tin."))
    store.fail_save = chat_store.TurnConflict()
    assert _ask(client).status_code == 409


def test_history_db_down_is_unavailable_and_runs_nothing(client, store, monkeypatch):
    calls = _stub_agent(monkeypatch, result=RunResult("x"))

    async def down(session_id: int, *args: Any):
        raise OperationalError("SELECT", {}, Exception("host 1.2.3.4"))

    monkeypatch.setattr(chat_store, "digests_after", down)
    response = _ask(client)
    assert response.status_code == 503 and "1.2.3.4" not in response.text
    assert calls == []


def test_create_and_delete_session(client, store):
    created = client.post("/sessions", headers=HEADERS)
    assert created.status_code == 201
    assert created.json()["session_uid"] == SID
    assert (
        created.json()["created_at"].endswith("Z")
        or "+00:00" in created.json()["created_at"]
    )
    assert "id" not in created.json()

    assert client.delete(f"/sessions/{SID}", headers=HEADERS).status_code == 204
    assert client.delete(f"/sessions/{SID}", headers=HEADERS).status_code == 404


def test_token_usage_is_summed_and_saved(client, store, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from app.graph.nodes.agent import token_usage

    cached = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_token_details": {"cache_read": 80},
    }
    uncached = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    messages = [
        HumanMessage("q"),
        AIMessage("", usage_metadata=cached),
        AIMessage("a", usage_metadata=uncached),  # provider without cache details
    ]
    assert token_usage(messages) == (200, 40, 80)
    assert token_usage([AIMessage("a")]) == (None, None, None)

    _stub_agent(
        monkeypatch,
        result=RunResult(
            "Có 2 tin.", input_tokens=200, output_tokens=40, cache_read_tokens=80
        ),
    )
    assert _ask(client).status_code == 200
    assert store.saved[0]["prompt_tokens"] == 200
    assert store.saved[0]["completion_tokens"] == 40
    assert store.saved[0]["ai_trace"]["cache_read_tokens"] == 80


# --- compaction --------------------------------------------------------------------------


def _long_digests(n: int) -> list[dict[str, Any]]:
    from app.graph.history import build_digest

    return [
        {
            "turn": t,
            **build_digest(f"q{t} " + "y" * 390, f"a{t} " + "x" * 590, verified=True),
        }
        for t in range(1, n + 1)
    ]


def _compaction_on(monkeypatch, fold_turns) -> None:
    from app.graph import history

    monkeypatch.setattr(history, "HISTORY_TOKENS", 1000)  # ~340 tokens per digest
    app.dependency_overrides[sessions.compactor_model] = lambda: object()
    monkeypatch.setattr(sessions.compaction, "fold_turns", fold_turns)


def test_long_session_is_folded_and_checkpoint_saved(client, store, monkeypatch):
    from app.graph.compaction import FoldResult
    from app.graph.history import merge_context

    store.digests = _long_digests(6)
    folded: list[list[int]] = []

    async def fake_fold(model, context, fold, *, today):
        folded.append([d["turn"] for d in fold])
        return FoldResult(
            context=merge_context(context, fold, "Lượt 1-3: tóm tắt."),
            folded_turns=[d["turn"] for d in fold],
            attempts=1,
        )

    _compaction_on(monkeypatch, fake_fold)
    calls = _stub_agent(monkeypatch, result=RunResult("Có 2 tin."))
    assert _ask(client).status_code == 200

    assert folded == [[1, 2, 3]]  # all but the 3 most recent turns
    assert calls[0]["context"]["through_turn"] == 3
    trace = store.saved[0]["ai_trace"]
    assert trace["context"]["summary"] == "Lượt 1-3: tóm tắt."
    assert trace["compaction"] == {
        "folded_turns": [1, 2, 3],
        "ok": True,
        "attempts": 1,
        "unsupported": [],
    }


def test_failed_fold_falls_back_and_still_answers(client, store, monkeypatch):
    from openai import APITimeoutError

    store.digests = _long_digests(6)

    async def broken_fold(model, context, fold, *, today):
        raise APITimeoutError(request=None)  # type: ignore[arg-type]

    _compaction_on(monkeypatch, broken_fold)
    calls = _stub_agent(monkeypatch, result=RunResult("Có 2 tin."))
    assert _ask(client).status_code == 200

    assert calls[0]["context"] is None
    trace = store.saved[0]["ai_trace"]
    assert "context" not in trace
    assert trace["compaction"] == {"folded_turns": [1, 2, 3], "ok": False}


def test_saved_checkpoint_is_resumed(client, store, monkeypatch):
    from app.graph.history import merge_context

    store.digests = _long_digests(5)
    store.context = merge_context(None, store.digests[:3], "Lượt 1-3: tóm tắt.")
    calls = _stub_agent(monkeypatch, result=RunResult("Có 2 tin."))
    assert _ask(client).status_code == 200

    assert calls[0]["context"]["through_turn"] == 3
    assert [d["turn"] for d in calls[0]["history"]] == [
        4,
        5,
    ]  # only unfolded turns loaded
    assert "context" not in store.saved[0]["ai_trace"]  # nothing new folded
