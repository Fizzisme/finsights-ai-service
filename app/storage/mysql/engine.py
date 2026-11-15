from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.settings import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    dsn = get_settings().mysql_source_readonly_dsn.get_secret_value()
    if not dsn:
        raise RuntimeError("MYSQL_SOURCE_READONLY_DSN is not set")
    return create_async_engine(
        dsn,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=0,
        pool_recycle=1800,
        connect_args={
            "connect_timeout": 10,
            # Defense in depth: refuse writes even if the account gains privileges
            "init_command": "SET SESSION TRANSACTION READ ONLY",
        },
    )


async def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    async with get_engine().connect() as conn:
        result = await conn.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings()]
