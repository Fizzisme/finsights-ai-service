from datetime import date
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from app.schemas.news import CountNewsInput, CountNewsResult
from app.tools.sql import count_news as mod


def test_total_query_only_published_and_flat() -> None:
    sql, params = mod._build_total_query(CountNewsInput(group_by="symbol"))
    assert "a.article_status = 'published'" in sql
    assert "GROUP BY" not in sql and "news_article_securities" not in sql
    assert params == {}


def test_values_bound_and_dates_inclusive() -> None:
    args = CountNewsInput(
        symbol="vcb",
        query="x'; DROP TABLE a;--",
        date_from=date(2026, 1, 1),
        date_to=date(2026, 1, 31),
    )
    sql, params = mod._build_total_query(args)
    assert "DROP" not in sql and "VCB" not in sql
    assert params["symbol"] == "VCB"
    assert params["date_to_excl"] == date(2026, 2, 1)


def test_symbol_filter_with_symbol_grouping_restricts_joined_rows() -> None:
    sql, params = mod._build_groups_query(
        CountNewsInput(symbol="VCB", group_by="symbol")
    )
    assert "s.symbol_at_publication = :symbol" in sql and "EXISTS" not in sql
    assert "COUNT(DISTINCT a.id)" in sql and "ORDER BY n DESC" in sql
    assert params["limit"] == 10


def test_day_grouping_sorted_by_date() -> None:
    sql, _ = mod._build_groups_query(CountNewsInput(group_by="day"))
    assert "DATE(a.published_at)" in sql and "ORDER BY key_ ASC" in sql


def test_invalid_range_rejected() -> None:
    with pytest.raises(ValueError):
        CountNewsInput(date_from=date(2026, 2, 1), date_to=date(2026, 1, 1))


@pytest.mark.anyio
async def test_count_news_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        if "GROUP BY" in sql:
            return [{"key_": "VCB", "n": 5}, {"key_": "TCB", "n": 2}]
        return [{"n": 6}]

    monkeypatch.setattr(mod, "fetch_all", fake)
    out = await mod.count_news(CountNewsInput(group_by="symbol"))
    assert out.total == 6
    assert [(g.key, g.count) for g in out.groups] == [("VCB", 5), ("TCB", 2)]


@pytest.mark.anyio
async def test_zero_total_skips_groups_query(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 0

    async def fake(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        nonlocal n
        n += 1
        return [{"n": 0}]

    monkeypatch.setattr(mod, "fetch_all", fake)
    out = await mod.count_news(CountNewsInput(group_by="publisher"))
    assert out == CountNewsResult(total=0) and n == 1


def test_tool_schema() -> None:
    tool = mod.count_news_tool
    assert tool.name == "count_news"
    assert set(tool.args) == {
        "symbol",
        "query",
        "date_from",
        "date_to",
        "publisher",
        "group_by",
        "limit",
    }
    assert all("description" in p for p in tool.args.values())


@pytest.mark.anyio
async def test_db_failure_returns_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(args: CountNewsInput) -> CountNewsResult:
        raise OperationalError("SELECT secret", {}, Exception("host 1.2.3.4 denied"))

    monkeypatch.setattr(mod, "count_news", boom)
    out = await mod.count_news_tool.ainvoke({})
    assert out["total"] is None
    assert "1.2.3.4" not in str(out) and "secret" not in str(out)


def test_publisher_filter_works_without_join_in_total() -> None:
    sql, params = mod._build_total_query(CountNewsInput(publisher="CafeF"))
    assert "code = :publisher" in sql and "JOIN" not in sql
    assert params["publisher"] == "cafef"
    sql, _ = mod._build_groups_query(
        CountNewsInput(publisher="cafef", group_by="symbol")
    )
    assert "code = :publisher" in sql
