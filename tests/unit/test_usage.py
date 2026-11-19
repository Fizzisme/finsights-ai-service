from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.mysql import usage_store
from app.storage.mysql.usage_store import FIVE_HOUR, WEEKLY, billable_tokens, windows

NOW = datetime(2026, 11, 19, 12, 0)
LIMITS = {FIVE_HOUR: 1000, WEEKLY: 5000}
USER = "11111111-1111-4111-8111-111111111111"


def _row(five_hour_ago: timedelta, week_ago: timedelta, used=(400, 2000)) -> dict:
    return {
        "window_started_at": NOW - five_hour_ago,
        "window_tokens": used[0],
        "week_started_at": NOW - week_ago,
        "week_tokens": used[1],
    }


def test_cache_reads_count_a_tenth() -> None:
    # 1000 input of which 800 from cache: 200 fresh + 80 + 300 output.
    assert billable_tokens(1000, 300, 800) == 580
    assert billable_tokens(None, None, None) == 0
    assert billable_tokens(500, 100, None) == 600


def test_no_usage_yet_means_no_window_open() -> None:
    result = windows(None, NOW, LIMITS)
    assert result[FIVE_HOUR] == usage_store.Window(0, 1000, None)
    assert result[WEEKLY] == usage_store.Window(0, 5000, None)


def test_open_windows_report_use_and_reset_time() -> None:
    result = windows(_row(timedelta(hours=1), timedelta(days=2)), NOW, LIMITS)
    assert result[FIVE_HOUR] == usage_store.Window(400, 1000, NOW + timedelta(hours=4))
    assert result[WEEKLY] == usage_store.Window(2000, 5000, NOW + timedelta(days=5))
    assert not result[FIVE_HOUR].exhausted


def test_ended_window_starts_from_zero() -> None:
    # Anchored, not sliding: 5 hours after it opened, the whole window is gone.
    result = windows(_row(timedelta(hours=5), timedelta(days=2)), NOW, LIMITS)
    assert result[FIVE_HOUR] == usage_store.Window(0, 1000, None)
    assert result[WEEKLY].used == 2000


def test_limit_zero_is_never_exhausted() -> None:
    row = _row(timedelta(hours=1), timedelta(days=1), used=(10**9, 10**9))
    assert not any(
        w.exhausted for w in windows(row, NOW, {FIVE_HOUR: 0, WEEKLY: 0}).values()
    )
    assert windows(row, NOW, LIMITS)[FIVE_HOUR].exhausted


def test_charge_restarts_only_ended_windows() -> None:
    sql = usage_store._CHARGE
    # Tokens are assigned before the start column, which MySQL applies left to right.
    assert sql.index("window_tokens = IF(") < sql.index("window_started_at = IF(")
    assert sql.index("week_tokens = IF(") < sql.index("week_started_at = IF(")
    assert "INTERVAL 18000 SECOND" in sql and "INTERVAL 604800 SECOND" in sql


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_usage_endpoint_reports_both_windows(client, monkeypatch) -> None:
    started = usage_store.utc_now() - timedelta(hours=1)

    async def fake_get_usage(user: str):
        assert user == USER
        return {
            "window_started_at": started,
            "window_tokens": 4500,
            "week_started_at": started,
            "week_tokens": 4500,
        }

    monkeypatch.setattr(usage_store, "get_usage", fake_get_usage)
    response = client.get("/usage", headers={"X-User-Id": USER})

    assert response.status_code == 200
    body = response.json()
    assert body["five_hour"]["used"] == 4500
    assert body["five_hour"]["limit"] == 150_000
    assert body["five_hour"]["resets_at"].startswith(
        (started + timedelta(hours=5)).isoformat()[:16]
    )
    assert body["weekly"]["limit"] == 1_500_000


def test_usage_endpoint_needs_a_user(client) -> None:
    assert client.get("/usage").status_code == 401
