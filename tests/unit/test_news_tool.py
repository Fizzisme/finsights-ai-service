from datetime import datetime
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from app.schemas.news import NewsArticle, NewsSearchInput, NewsSearchResult
from app.tools.sql import news_tool as mod


def test_tool_schema_exposes_name_and_descriptions() -> None:
    tool = mod.search_news_tool
    assert tool.name == "search_news"
    props = tool.args
    assert set(props) == {"symbol", "query", "date_from", "date_to", "limit"}
    assert all("description" in p for p in props.values())
    assert tool.description


@pytest.mark.anyio
async def test_tool_validates_and_returns_json_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[NewsSearchInput] = []

    async def fake(args: NewsSearchInput) -> NewsSearchResult:
        seen.append(args)
        return NewsSearchResult(
            articles=[
                NewsArticle(
                    article_uid="u1",
                    title="t",
                    published_at=datetime(2026, 5, 1, 8, 0),
                    publisher="CafeF",
                    url="https://example.com",
                    content_rights="link_only",
                )
            ]
        )

    monkeypatch.setattr(mod, "search_news", fake)
    out: dict[str, Any] = await mod.search_news_tool.ainvoke(
        {"symbol": "vcb", "limit": 3}
    )

    assert seen[0].symbol == "VCB" and seen[0].limit == 3
    assert out["articles"][0]["published_at"] == "2026-05-01T08:00:00"


@pytest.mark.anyio
async def test_tool_rejects_invalid_args() -> None:
    with pytest.raises(Exception):  # noqa: B017 - ValidationError subclass varies by version
        await mod.search_news_tool.ainvoke({"limit": 99})


@pytest.mark.anyio
async def test_db_failure_returns_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(args: NewsSearchInput) -> NewsSearchResult:
        raise OperationalError("SELECT secret", {}, Exception("host 1.2.3.4 denied"))

    monkeypatch.setattr(mod, "search_news", boom)
    out = await mod.search_news_tool.ainvoke({})
    assert out["articles"] == []
    assert "1.2.3.4" not in str(out) and "secret" not in str(out)
