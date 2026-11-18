from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from app.schemas.news import GetNewsArticleInput, GetNewsArticleResult
from app.tools.sql import get_news_article as mod


def test_requires_exactly_one_identifier() -> None:
    with pytest.raises(ValueError):
        GetNewsArticleInput()
    with pytest.raises(ValueError):
        GetNewsArticleInput(article_uid="abcdefgh", url="https://x.vn/a")


def test_rejects_non_http_url() -> None:
    with pytest.raises(ValueError):
        GetNewsArticleInput(url="javascript:alert(1)")


def test_query_binds_value() -> None:
    sql, params = mod._build_query(GetNewsArticleInput(article_uid="abcd-1234"))
    assert "a.article_status = 'published'" in sql and "abcd-1234" not in sql
    assert params == {"uid": "abcd-1234"}
    sql, params = mod._build_query(GetNewsArticleInput(url="https://x.vn/a"))
    assert ":url" in sql and params == {"url": "https://x.vn/a"}


@pytest.mark.anyio
async def test_found_and_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict] = []

    async def fake(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        if "FROM news_articles" in sql:
            return rows
        return [
            {
                "article_id": 7,
                "symbol": "VCB",
                "match_method": "ticker_boundary",
                "review_status": "auto",
                "confidence": Decimal("0.9800"),
            }
        ]

    monkeypatch.setattr(mod, "fetch_all", fake)
    import app.tools.sql.search_news as sn

    monkeypatch.setattr(sn, "fetch_all", fake)

    args = GetNewsArticleInput(article_uid="abcd-1234")
    assert (await mod.get_news_article(args)).article is None

    rows.append(
        {
            "article_id": 7,
            "article_uid": "abcd-1234",
            "title": "VCB tăng vốn",
            "summary": None,
            "published_at": datetime(2026, 5, 1, 8, 0),
            "publisher": "CafeF",
            "url": "https://x.vn/a",
            "content_rights": "link_only",
        }
    )
    article = (await mod.get_news_article(args)).article
    assert article is not None and article.symbols[0].symbol == "VCB"


@pytest.mark.anyio
async def test_db_failure_returns_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(args: GetNewsArticleInput) -> GetNewsArticleResult:
        raise OperationalError("SELECT secret", {}, Exception("host 1.2.3.4 denied"))

    monkeypatch.setattr(mod, "get_news_article", boom)
    out = await mod.get_news_article_tool.ainvoke({"article_uid": "abcd-1234"})
    assert out["article"] is None
    assert "1.2.3.4" not in str(out) and "secret" not in str(out)
