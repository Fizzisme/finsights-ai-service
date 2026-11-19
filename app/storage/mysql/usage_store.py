"""Per-user token quota in finsights_ai.chat_usage (Node 1). Schema is owned by finsights-coredb.

Two anchored windows, as in Claude: 5 hours and 7 days. A window opens at the first turn
after the previous one ended and drops to zero when it ends; it does not slide. A turn is
charged after it runs, since its tokens are only known then, so the last turn of a window
may go over the limit.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from app.storage.mysql.engine import get_chat_engine

FIVE_HOUR, WEEKLY = "five_hour", "weekly"
WINDOW_LENGTHS = {FIVE_HOUR: timedelta(hours=5), WEEKLY: timedelta(days=7)}
_COLUMNS = {
    FIVE_HOUR: ("window_started_at", "window_tokens"),
    WEEKLY: ("week_started_at", "week_tokens"),
}
# Prompt-cache reads are billed at about a tenth of a fresh input token.
CACHE_READ_WEIGHT = 0.1


def utc_now() -> datetime:
    """Naive UTC, as finsights_ai stores it (server time_zone is UTC)."""
    return datetime.now(UTC).replace(tzinfo=None)


def billable_tokens(
    input_tokens: int | None, output_tokens: int | None, cache_read_tokens: int | None
) -> int:
    """Tokens a turn counts against the quota: fresh input + output + a tenth of cache reads."""
    cached = cache_read_tokens or 0
    fresh = max(0, (input_tokens or 0) - cached)
    return fresh + (output_tokens or 0) + round(cached * CACHE_READ_WEIGHT)


@dataclass(frozen=True)
class Window:
    used: int
    limit: int  # 0 = no limit
    resets_at: datetime | None  # None: no window open yet

    @property
    def exhausted(self) -> bool:
        return self.limit > 0 and self.used >= self.limit


def windows(
    row: dict[str, Any] | None, now: datetime, limits: dict[str, int]
) -> dict[str, Window]:
    """Both windows as of `now`; one that has ended counts as unused."""
    result: dict[str, Window] = {}
    for name, (started_column, tokens_column) in _COLUMNS.items():
        result[name] = Window(0, limits[name], None)
        if row is None or row[started_column] is None:
            continue
        ends = row[started_column] + WINDOW_LENGTHS[name]
        if ends > now:
            result[name] = Window(int(row[tokens_column]), limits[name], ends)
    return result


async def get_usage(user_uid: str) -> dict[str, Any] | None:
    async with get_chat_engine().connect() as conn:
        result = await conn.execute(
            text(
                "SELECT window_started_at, window_tokens, week_started_at, week_tokens "
                "FROM chat_usage WHERE user_uid = :user"
            ),
            {"user": user_uid},
        )
        row = result.mappings().first()
    return dict(row) if row else None


def _rollover(name: str) -> tuple[str, str]:
    """SET clauses of one window: restart it at :at if it has ended, else add to it.

    The tokens column is assigned first: MySQL applies the assignments left to right, so the
    started column must still hold its old value when the tokens condition reads it.
    """
    started, tokens = _COLUMNS[name]
    seconds = int(WINDOW_LENGTHS[name].total_seconds())
    ended = f"{started} <= :at - INTERVAL {seconds} SECOND"
    return (
        f"{tokens} = IF({ended}, :tokens, {tokens} + :tokens)",
        f"{started} = IF({ended}, :at, {started})",
    )


_CHARGE = (
    "INSERT INTO chat_usage "
    "(user_uid, window_started_at, window_tokens, week_started_at, week_tokens) "
    "VALUES (:user, :at, :tokens, :at, :tokens) "
    "ON DUPLICATE KEY UPDATE "
    + ", ".join([*_rollover(FIVE_HOUR), *_rollover(WEEKLY)])
    + ", updated_at = CURRENT_TIMESTAMP(3)"
)


async def charge(user_uid: str, tokens: int, at: datetime) -> None:
    """Add a turn's tokens. `at` is when the turn started, so a new window opens at it.

    One statement, so concurrent turns of the same user cannot lose an update.
    """
    async with get_chat_engine().begin() as conn:
        await conn.execute(
            text(_CHARGE), {"user": user_uid, "tokens": tokens, "at": at}
        )
