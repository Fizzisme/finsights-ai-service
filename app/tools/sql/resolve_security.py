"""resolve_security: company name / brand -> ticker, derived from news tagging.

Source is news_article_securities only (the read-only account cannot see the
securities master), so coverage is limited to tickers that already have news.
"""

import logging
import re
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.news import (
    ResolvedSecurity,
    ResolveSecurityInput,
    ResolveSecurityResult,
)
from app.storage.mysql.engine import fetch_all
from app.tools.sql.search_news import _escape_like

logger = logging.getLogger(__name__)

_TICKER = re.compile(r"^[A-Za-z0-9]{2,10}$")
_MAX_ALIASES = 5


def _build_query(args: ResolveSecurityInput) -> tuple[str, dict[str, Any]]:
    where = ["matched_alias LIKE :kw"]
    params: dict[str, Any] = {"kw": f"%{_escape_like(args.name)}%", "limit": 20}
    if _TICKER.match(args.name):
        where.append("symbol_at_publication = :symbol")
        params["symbol"] = args.name.upper()
    sql = (
        "SELECT symbol_at_publication AS symbol, matched_alias AS alias, COUNT(*) AS n "
        "FROM news_article_securities "
        f"WHERE symbol_at_publication IS NOT NULL AND ({' OR '.join(where)}) "
        "GROUP BY symbol_at_publication, matched_alias "
        "ORDER BY n DESC LIMIT :limit"
    )
    return sql, params


async def resolve_security(args: ResolveSecurityInput) -> ResolveSecurityResult:
    sql, params = _build_query(args)
    rows = await fetch_all(sql, params)

    aliases: dict[str, list[str]] = {}  # insertion order = most-mentioned symbol first
    for row in rows:
        names = aliases.setdefault(row["symbol"], [])
        alias = row["alias"]
        if alias and alias.upper() != row["symbol"] and alias not in names:
            names.append(alias)
    return ResolveSecurityResult(
        matches=[
            ResolvedSecurity(symbol=sym, aliases=names[:_MAX_ALIASES])
            for sym, names in aliases.items()
        ]
    )


@tool("resolve_security", args_schema=ResolveSecurityInput)
async def resolve_security_tool(name: str) -> dict[str, Any]:
    """Find the stock ticker for a company name, brand or ticker (e.g. 'Hòa Phát' -> HPG).

    Coverage is limited to tickers that appear in the news database, so an empty
    result does not prove the company is not listed; then search news by name.
    """
    args = ResolveSecurityInput(name=name)
    try:
        result = await resolve_security(args)
    except SQLAlchemyError, RuntimeError:
        logger.exception("resolve_security failed")
        return {"error": "News database is temporarily unavailable.", "matches": []}
    return result.model_dump(mode="json")
