from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.schemas.news import NewsSearchInput
from app.tools.sql import search_news as mod


def test_no_filters_only_published_and_limit() -> None:
    sql, params = mod._build_articles_query(NewsSearchInput())
    assert "a.article_status = 'published'" in sql
    assert params == {"limit": 10}


def test_user_values_are_bound_never_inlined() -> None:
    args = NewsSearchInput(
        symbol="vcb", query="x'; DROP TABLE a;--", date_from=date(2026, 1, 1), date_to=date(2026, 1, 31)
    )
    sql, params = mod._build_articles_query(args)
    assert "DROP" not in sql and "VCB" not in sql
    assert params["symbol"] == "VCB"
    assert params["date_from"] == date(2026, 1, 1)
    assert params["date_to_excl"] == date(2026, 2, 1)  # inclusive end -> next day


def test_like_wildcards_escaped() -> None:
    _, params = mod._build_articles_query(NewsSearchInput(query="50%_off"))
    assert params["kw"] == "%50\\%\\_off%"


@pytest.mark.anyio
async def test_search_news_assembles_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        calls.append((sql, params or {}))
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

    assert len(calls) == 2
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
    assert (await mod.search_news(NewsSearchInput())).articles == []
    assert n == 1
