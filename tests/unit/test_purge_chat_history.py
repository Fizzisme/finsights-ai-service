import pytest

from app.storage.mysql import chat_store
from scripts import purge_chat_history as mod


class FakeStore:
    def __init__(self, expired: list[int]) -> None:
        self.expired = list(expired)
        self.purged: list[list[int]] = []

    async def count_expired(self) -> dict[str, int]:
        n = len(self.expired)
        return {"sessions": n, "deleted_sessions": 0, "messages": 2 * n}

    async def expired_session_ids(self, limit: int) -> list[int]:
        return self.expired[:limit]

    async def purge_sessions(self, ids: list[int]) -> tuple[int, int]:
        self.purged.append(ids)
        self.expired = [i for i in self.expired if i not in ids]
        return 2 * len(ids), len(ids)


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore(list(range(1, 251)))
    for name in ("count_expired", "expired_session_ids", "purge_sessions"):
        monkeypatch.setattr(chat_store, name, getattr(fake, name))
    return fake


@pytest.mark.anyio
async def test_dry_run_only_counts(store) -> None:
    out = await mod.purge(apply=False, batch_size=100, max_batches=10)
    assert out["sessions"] == 250 and out["purged_sessions"] == 0
    assert store.purged == []


@pytest.mark.anyio
async def test_apply_deletes_in_batches_until_done(store) -> None:
    out = await mod.purge(apply=True, batch_size=100, max_batches=10)
    assert [len(b) for b in store.purged] == [100, 100, 50]
    assert out["purged_sessions"] == 250 and out["purged_messages"] == 500


@pytest.mark.anyio
async def test_max_batches_bounds_one_run(store) -> None:
    out = await mod.purge(apply=True, batch_size=100, max_batches=1)
    assert out["purged_sessions"] == 100 and len(store.expired) == 150


@pytest.mark.anyio
async def test_stops_when_candidates_turned_active(monkeypatch, store) -> None:
    async def nothing_confirmed(ids: list[int]) -> tuple[int, int]:
        store.purged.append(ids)
        return 0, 0

    monkeypatch.setattr(chat_store, "purge_sessions", nothing_confirmed)
    out = await mod.purge(apply=True, batch_size=100, max_batches=10)
    assert len(store.purged) == 1 and out["purged_sessions"] == 0


def test_retention_windows_are_bound_parameters() -> None:
    assert chat_store.DELETED_RETENTION_DAYS == 30
    assert chat_store.INACTIVE_RETENTION_DAYS == 180
    assert "30" not in chat_store._EXPIRED and "180" not in chat_store._EXPIRED
    assert chat_store._RETENTION["deleted_days"] == 30
