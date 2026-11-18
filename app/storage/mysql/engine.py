import ssl
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.settings import get_settings


class EndpointNameContext(ssl.SSLContext):
    """Verify the server certificate against a fixed name instead of the host we dialled.

    The MySQL nodes are reached by IP, but their certificates are issued to one endpoint
    name. asyncio passes the dialled host as `server_hostname`; this replaces it, so the
    chain and the name are both still verified (nothing is switched off).
    """

    endpoint_name: str = ""

    def wrap_bio(
        self, incoming, outgoing, server_side=False, server_hostname=None, session=None
    ):  # type: ignore[override]
        return super().wrap_bio(
            incoming, outgoing, server_side, self.endpoint_name, session
        )


def tls_context(ca_file: Path, endpoint_name: str) -> ssl.SSLContext:
    # PROTOCOL_TLS_CLIENT = CERT_REQUIRED + check_hostname.
    context = EndpointNameContext(ssl.PROTOCOL_TLS_CLIENT)
    context.endpoint_name = endpoint_name
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_verify_locations(ca_file)
    return context


def _connect_args(ca_file: Path, **extra: Any) -> dict[str, Any]:
    settings = get_settings()
    args: dict[str, Any] = {"connect_timeout": 10, **extra}
    if settings.mysql_tls:
        # Note: aiomysql's TLS upgrade fails on Windows' ProactorEventLoop (WinError 87);
        # uvicorn --reload and Linux use SelectorEventLoop, which works.
        args["ssl"] = tls_context(ca_file, settings.mysql_tls_server_name)
    return args


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    dsn = settings.mysql_source_readonly_dsn.get_secret_value()
    if not dsn:
        raise RuntimeError("MYSQL_SOURCE_READONLY_DSN is not set")
    return create_async_engine(
        dsn,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=0,
        pool_recycle=1800,
        connect_args=_connect_args(
            settings.mysql_source_tls_ca,
            # Defense in depth: refuse writes even if the account gains privileges
            init_command="SET SESSION TRANSACTION READ ONLY",
        ),
    )


@lru_cache
def get_chat_engine() -> AsyncEngine:
    """Chat history on Node 1; unlike the source engine it writes."""
    settings = get_settings()
    dsn = settings.mysql_chat_dsn.get_secret_value()
    if not dsn:
        raise RuntimeError("MYSQL_CHAT_DSN is not set")
    return create_async_engine(
        dsn,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=0,
        pool_recycle=1800,
        connect_args=_connect_args(settings.mysql_chat_tls_ca),
    )


async def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    async with get_engine().connect() as conn:
        result = await conn.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings()]
