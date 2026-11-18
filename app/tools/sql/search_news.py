"""search_news: fixed-parameter SQL plus its LangChain tool. Reads finsights_news only."""

import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.news import (
    VN_UTC_OFFSET,
    NewsArticle,
    NewsSearchInput,
    NewsSearchResult,
    NewsSecurityMatch,
)
from app.storage.mysql.engine import fetch_all

logger = logging.getLogger(__name__)


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so user text is matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def vn_day_start_utc(d: date) -> datetime:
    """Start of a Vietnam calendar day, as the naive UTC value MySQL stores."""
    return datetime.combine(d, time()) - VN_UTC_OFFSET


def _build_articles_query(args: NewsSearchInput) -> tuple[str, dict[str, Any]]:
    # Only constant SQL fragments are appended; every user value is a bound parameter.
    where = ["a.article_status = 'published'"]
    params: dict[str, Any] = {"limit": args.limit}

    if args.symbol:
        where.append(
            "EXISTS (SELECT 1 FROM news_article_securities s "
            "WHERE s.article_id = a.id AND s.symbol_at_publication = :symbol)"
        )
        params["symbol"] = args.symbol

    if args.query:
        where.append("(a.title LIKE :kw OR a.summary LIKE :kw)")
        params["kw"] = f"%{_escape_like(args.query)}%"

    if args.publisher:
        where.append("p.code = :publisher")
        params["publisher"] = args.publisher

    if args.date_from:
        where.append("a.published_at >= :date_from")
        params["date_from"] = vn_day_start_utc(args.date_from)

    if args.date_to:
        # date_to is inclusive: compare against the start of the next day.
        where.append("a.published_at < :date_to_excl")
        params["date_to_excl"] = vn_day_start_utc(args.date_to + timedelta(days=1))

    sql = (
        "SELECT a.id AS article_id, a.article_uid, a.title, a.summary, a.published_at, "
        "p.name AS publisher, a.canonical_url AS url, a.content_rights "
        "FROM news_articles a "
        "JOIN news_publishers p ON p.id = a.publisher_id "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY a.published_at DESC, a.id DESC "
        "LIMIT :limit"
    )
    return sql, params


async def _fetch_matches(article_ids: list[int]) -> dict[int, list[NewsSecurityMatch]]:
    if not article_ids:
        return {}
    # One named placeholder per id (max 20): still bound parameters, no string-built values.
    names = [f"id{i}" for i in range(len(article_ids))]
    sql = (
        "SELECT article_id, symbol_at_publication AS symbol, match_method, "
        "review_status, match_confidence AS confidence "
        "FROM news_article_securities "
        f"WHERE article_id IN ({', '.join(':' + n for n in names)}) "
        "AND symbol_at_publication IS NOT NULL "
        "ORDER BY is_primary DESC, match_confidence DESC"
    )
    rows = await fetch_all(sql, dict(zip(names, article_ids, strict=True)))

    grouped: dict[int, list[NewsSecurityMatch]] = defaultdict(list)
    for row in rows:
        grouped[row["article_id"]].append(
            NewsSecurityMatch(
                symbol=row["symbol"],
                match_method=row["match_method"],
                review_status=row["review_status"],
                confidence=float(row["confidence"])
                if row["confidence"] is not None
                else None,
            )
        )
    return grouped


async def latest_published_at() -> datetime | None:
    """Newest publication time in the database, so the model can tell stale data."""
    rows = await fetch_all(
        "SELECT MAX(published_at) AS latest FROM news_articles "
        "WHERE article_status = 'published'"
    )
    return rows[0]["latest"] if rows else None


async def search_news(args: NewsSearchInput) -> NewsSearchResult:
    sql, params = _build_articles_query(args)
    rows = await fetch_all(sql, params)
    matches = await _fetch_matches([row["article_id"] for row in rows])

    articles = [
        NewsArticle(
            article_uid=row["article_uid"],
            title=row["title"],
            summary=row["summary"],
            published_at=row["published_at"],
            publisher=row["publisher"],
            url=row["url"],
            content_rights=row["content_rights"],
            symbols=matches.get(row["article_id"], []),
        )
        for row in rows
    ]
    return NewsSearchResult(articles=articles, data_as_of=await latest_published_at())


@tool("search_news", args_schema=NewsSearchInput)
async def search_news_tool(
    symbol: str | None = None,
    query: str | None = None,
    date_from: Any = None,
    date_to: Any = None,
    publisher: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search published Vietnamese financial news, newest first.

    Filter by stock ticker (symbol), keyword (query), news source (publisher) and/or
    publication date range.
    Returns article titles, summaries, publisher, link (url) and the tickers each
    article was matched to. Full article text is not available; cite the url.
    """
    # args_schema has already validated/normalised the arguments.
    args = NewsSearchInput(
        symbol=symbol,
        query=query,
        date_from=date_from,
        date_to=date_to,
        publisher=publisher,
        limit=limit,
    )
    try:
        result = await search_news(args)
    except SQLAlchemyError, RuntimeError:
        # Never leak DSN/host details to the model; give it something it can act on.
        logger.exception("search_news failed")
        return {"error": "News database is temporarily unavailable.", "articles": []}
    return result.model_dump(mode="json")
