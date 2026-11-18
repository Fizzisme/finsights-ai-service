"""count_news: fixed-parameter aggregate over finsights_news (counts, rankings)."""

import logging
from datetime import timedelta
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.news import (
    CountNewsInput,
    CountNewsResult,
    NewsCountGroup,
)
from app.storage.mysql.engine import fetch_all
from app.tools.sql.search_news import _escape_like

logger = logging.getLogger(__name__)

# Constant SQL fragments only; user values are always bound parameters.
_GROUP_EXPR = {
    "symbol": "s.symbol_at_publication",
    "publisher": "p.name",
    "day": "DATE(a.published_at)",
}


def _build_where(args: CountNewsInput) -> tuple[str, dict[str, Any]]:
    where = ["a.article_status = 'published'"]
    params: dict[str, Any] = {}

    if args.symbol:
        if args.group_by == "symbol":
            # The securities table is already joined: restrict its rows directly.
            where.append("s.symbol_at_publication = :symbol")
        else:
            where.append(
                "EXISTS (SELECT 1 FROM news_article_securities s2 "
                "WHERE s2.article_id = a.id AND s2.symbol_at_publication = :symbol)"
            )
        params["symbol"] = args.symbol

    if args.query:
        where.append("(a.title LIKE :kw OR a.summary LIKE :kw)")
        params["kw"] = f"%{_escape_like(args.query)}%"

    if args.publisher:
        # Subquery so the total query needs no publishers join.
        where.append(
            "a.publisher_id IN (SELECT id FROM news_publishers WHERE code = :publisher)"
        )
        params["publisher"] = args.publisher

    if args.date_from:
        where.append("a.published_at >= :date_from")
        params["date_from"] = args.date_from

    if args.date_to:
        where.append("a.published_at < :date_to_excl")
        params["date_to_excl"] = args.date_to + timedelta(days=1)

    return " AND ".join(where), params


def _build_total_query(args: CountNewsInput) -> tuple[str, dict[str, Any]]:
    # `total` is always the number of distinct articles, whatever the grouping.
    flat = args.model_copy(update={"group_by": "none"})
    where, params = _build_where(flat)
    sql = f"SELECT COUNT(*) AS n FROM news_articles a WHERE {where}"
    return sql, params


def _build_groups_query(args: CountNewsInput) -> tuple[str, dict[str, Any]]:
    where, params = _build_where(args)
    expr = _GROUP_EXPR[args.group_by]
    joins = "JOIN news_publishers p ON p.id = a.publisher_id "
    if args.group_by == "symbol":
        joins += (
            "JOIN news_article_securities s ON s.article_id = a.id "
            "AND s.symbol_at_publication IS NOT NULL "
        )
    order = "key_ ASC" if args.group_by == "day" else "n DESC, key_ ASC"
    sql = (
        f"SELECT {expr} AS key_, COUNT(DISTINCT a.id) AS n "
        f"FROM news_articles a {joins}"
        f"WHERE {where} GROUP BY key_ ORDER BY {order} LIMIT :limit"
    )
    params["limit"] = args.limit
    return sql, params


async def count_news(args: CountNewsInput) -> CountNewsResult:
    sql, params = _build_total_query(args)
    total = int((await fetch_all(sql, params))[0]["n"])

    groups: list[NewsCountGroup] = []
    if args.group_by != "none" and total:
        sql, params = _build_groups_query(args)
        rows = await fetch_all(sql, params)
        groups = [NewsCountGroup(key=str(r["key_"]), count=int(r["n"])) for r in rows]
    return CountNewsResult(total=total, groups=groups)


@tool("count_news", args_schema=CountNewsInput)
async def count_news_tool(
    symbol: str | None = None,
    query: str | None = None,
    date_from: Any = None,
    date_to: Any = None,
    publisher: str | None = None,
    group_by: str = "none",
    limit: int = 10,
) -> dict[str, Any]:
    """Count published Vietnamese financial news articles, optionally broken down.

    Use for "how many articles" and "which stock/source is covered most" questions
    instead of counting search_news results yourself. Filter by ticker, keyword, news
    source and date range; group_by gives a ranking per ticker, publisher or day. Only the 15
    tickers present in the news database are tagged to articles.
    """
    args = CountNewsInput(
        symbol=symbol,
        query=query,
        date_from=date_from,
        date_to=date_to,
        publisher=publisher,
        group_by=group_by,
        limit=limit,
    )
    try:
        result = await count_news(args)
    except SQLAlchemyError, RuntimeError:
        logger.exception("count_news failed")
        return {"error": "News database is temporarily unavailable.", "total": None}
    return result.model_dump(mode="json")
