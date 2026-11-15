"""LangChain tool wrapper around the fixed-parameter news query."""

import logging
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.news import NewsSearchInput
from app.tools.sql.search_news import search_news

logger = logging.getLogger(__name__)


@tool("search_news", args_schema=NewsSearchInput)
async def search_news_tool(
    symbol: str | None = None,
    query: str | None = None,
    date_from: Any = None,
    date_to: Any = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search published Vietnamese financial news, newest first.

    Filter by stock ticker (symbol), keyword (query) and/or publication date range.
    Returns article titles, summaries, publisher, link (url) and the tickers each
    article was matched to. Full article text is not available; cite the url.
    """
    # args_schema has already validated/normalised the arguments.
    args = NewsSearchInput(
        symbol=symbol, query=query, date_from=date_from, date_to=date_to, limit=limit
    )
    try:
        result = await search_news(args)
    except SQLAlchemyError, RuntimeError:
        # Never leak DSN/host details to the model; give it something it can act on.
        logger.exception("search_news failed")
        return {"error": "News database is temporarily unavailable.", "articles": []}
    return result.model_dump(mode="json")
