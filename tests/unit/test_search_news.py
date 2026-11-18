from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.schemas.news import NewsSearchInput
from app.tools.sql import search_news as mod


def test_no_filters_only_published_and_limit() -> None:
    sql, params = mod._build_articles_query(NewsSearchInput())
    assert "a.article_status = 'published'" in sql
    assert params == {"limit": 8}


def test_user_values_are_bound_never_inlined() -> None:
    args = NewsSearchInput(
        symbol="vcb", query="x'; DROP TABLE a;--", date_from=date(2026, 1, 1), date_to=date(2026, 1, 31)
    )
    sql, params = mod._build_articles_query(args)
    assert "DROP" not in sql and "VCB" not in sql
    assert params["symbol"] == "VCB"
    # Vietnam days -> UTC bounds; inclusive end -> start of the next VN day.
    assert params["date_from"] == datetime(2025, 12, 31, 17, 0)
    assert params["date_to_excl"] == datetime(2026, 1, 31, 17, 0)


def test_symbol_search_ranks_title_mentions_first() -> None:
    sql, _ = mod._build_articles_query(NewsSearchInput(symbol="HPG"))
    order = sql.split("ORDER BY", 1)[1]
    # Case- and accent-exact, so "điều hòa phát" does not count as naming Hòa Phát.
    assert order.lstrip().startswith("EXISTS")
    assert "LOCATE(t.matched_alias, a.title COLLATE utf8mb4_bin)" in order
    assert order.index("DESC") < order.index("a.published_at DESC")

    plain, _ = mod._build_articles_query(NewsSearchInput(query="lãi suất"))
    assert plain.split("ORDER BY", 1)[1].lstrip().startswith("a.published_at DESC")


def test_like_wildcards_escaped() -> None:
    _, params = mod._build_articles_query(NewsSearchInput(query="50%_off"))
    assert params["kw"] == "%50\\%\\_off%"


@pytest.mark.anyio
async def test_search_news_assembles_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        calls.append((sql, params or {}))
        if "MAX(published_at)" in sql:
            return [{"latest": datetime(2026, 5, 2, 1, 0)}]
        if "FROM news_articles" in sql:
            return [
                {
                    "article_id": 7,
                    "article_uid": "uid-7",
                    "title": "VCB tăng vốn",
                    "summary": None,
                    "published_at": datetime(2026, 5, 1, 8, 0),
                    "publisher": "CafeF",
                    "url": "https://example.com/7",
                    "content_rights": "link_only",
                }
            ]
        return [
            {
                "article_id": 7,
                "symbol": "VCB",
                "match_method": "ticker_boundary",
                "review_status": "auto",
                "confidence": Decimal("0.9500"),
            }
        ]

    monkeypatch.setattr(mod, "fetch_all", fake_fetch_all)
    result = await mod.search_news(NewsSearchInput(symbol="VCB"))

    assert len(calls) == 3  # articles, matches, data_as_of
    assert result.data_as_of is not None
    assert result.data_as_of.isoformat() == "2026-05-02T08:00:00+07:00"
    article = result.articles[0]
    assert article.article_uid == "uid-7"
    assert article.symbols[0].confidence == 0.95


@pytest.mark.anyio
async def test_no_rows_skips_second_query(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 0

    async def fake_fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        nonlocal n
        n += 1
        return []

    monkeypatch.setattr(mod, "fetch_all", fake_fetch_all)
    result = await mod.search_news(NewsSearchInput())
    assert result.articles == [] and result.data_as_of is None
    assert n == 2  # articles + data_as_of, no matches query


def test_publisher_filter_bound_and_lowercased() -> None:
    sql, params = mod._build_articles_query(NewsSearchInput(publisher=" VnExpress "))
    assert "p.code = :publisher" in sql and "vnexpress" not in sql
    assert params["publisher"] == "vnexpress"


def test_publisher_rejects_injection_chars() -> None:
    with pytest.raises(ValueError):
        NewsSearchInput(publisher="cafef' OR 1=1")


@pytest.mark.anyio
async def test_data_as_of_hidden_for_a_complete_period(monkeypatch) -> None:
    # Newest article: 2026-11-17 05:10 UTC = 17/11 12:10 in Vietnam.
    async def fake_fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        return [{"latest": datetime(2026, 11, 17, 5, 10)}]

    monkeypatch.setattr(mod, "fetch_all", fake_fetch_all)
    assert await mod.latest_published_at(date(2026, 10, 31)) is None  # October is over
    assert await mod.latest_published_at(date(2026, 11, 17)) is not None  # same day
    assert await mod.latest_published_at(date(2026, 11, 30)) is not None
    assert await mod.latest_published_at(None) is not None


def test_article_for_model_is_compact() -> None:
    from app.schemas.news import NewsArticle, NewsSecurityMatch

    article = NewsArticle(
        article_uid="uid-7",
        title="VCB và TCB",
        summary="x" * 500,
        published_at=datetime(2026, 5, 1, 1, 0),
        publisher="CafeF",
        url="https://example.com/7",
        content_rights="link_only",
        symbols=[
            NewsSecurityMatch(symbol="VCB", match_method="ticker_boundary", review_status="auto", confidence=0.98),
            NewsSecurityMatch(symbol="TCB", match_method="company_phrase", review_status="pending", confidence=0.97),
            NewsSecurityMatch(symbol="MBB", match_method="ticker_boundary", review_status="auto", confidence=0.5),
        ],
    )
    view = mod.article_for_model(article)
    assert set(view) == {"title", "summary", "published_at", "publisher", "url", "tickers", "tickers_unconfirmed"}
    assert view["tickers"] == ["VCB"] and view["tickers_unconfirmed"] == ["TCB", "MBB"]
    assert len(view["summary"]) == mod.SUMMARY_CHARS_FOR_MODEL + 1  # plus the ellipsis
    assert view["published_at"] == "2026-05-01T08:00:00+07:00"

    plain = mod.article_for_model(article.model_copy(update={"symbols": [], "summary": None}))
    assert "tickers_unconfirmed" not in plain and plain["summary"] is None
