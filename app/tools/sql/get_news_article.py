"""get_news_article: fetch one published article by article_uid or url."""

import logging
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.news import GetNewsArticleInput, GetNewsArticleResult, NewsArticle
from app.storage.mysql.engine import fetch_all
from app.tools.sql.search_news import _fetch_matches

logger = logging.getLogger(__name__)

_COLUMNS = (
    "SELECT a.id AS article_id, a.article_uid, a.title, a.summary, a.published_at, "
    "p.name AS publisher, a.canonical_url AS url, a.content_rights "
    "FROM news_articles a JOIN news_publishers p ON p.id = a.publisher_id "
    "WHERE a.article_status = 'published' AND "
)


def _build_query(args: GetNewsArticleInput) -> tuple[str, dict[str, Any]]:
    if args.article_uid:
        return _COLUMNS + "a.article_uid = :uid LIMIT 1", {"uid": args.article_uid}
    return _COLUMNS + "a.canonical_url = :url LIMIT 1", {"url": args.url}


async def get_news_article(args: GetNewsArticleInput) -> GetNewsArticleResult:
    sql, params = _build_query(args)
    rows = await fetch_all(sql, params)
    if not rows:
        return GetNewsArticleResult()
    row = rows[0]
    matches = await _fetch_matches([row["article_id"]])
    return GetNewsArticleResult(
        article=NewsArticle(
            article_uid=row["article_uid"],
            title=row["title"],
            summary=row["summary"],
            published_at=row["published_at"],
            publisher=row["publisher"],
            url=row["url"],
            content_rights=row["content_rights"],
            symbols=matches.get(row["article_id"], []),
        )
    )


@tool("get_news_article", args_schema=GetNewsArticleInput)
async def get_news_article_tool(
    article_uid: str | None = None, url: str | None = None
) -> dict[str, Any]:
    """Get one published article (title, summary, publisher, link, matched tickers).

    Give exactly one of article_uid or url, both taken from a search_news result.
    Full article text is not available; send the user to the link for details.
    """
    args = GetNewsArticleInput(article_uid=article_uid, url=url)
    try:
        result = await get_news_article(args)
    except SQLAlchemyError, RuntimeError:
        logger.exception("get_news_article failed")
        return {"error": "News database is temporarily unavailable.", "article": None}
    return result.model_dump(mode="json")
