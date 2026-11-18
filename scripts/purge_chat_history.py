"""Chat-history retention: remove sessions deleted 30+ days ago or inactive 180+ days.

uv run --env-file .env python -m scripts.purge_chat_history            # dry run: counts only
uv run --env-file .env python -m scripts.purge_chat_history --apply    # delete

Meant to run once a day (cron, later Dagster). Needs MYSQL_CHAT_DSN on finsights_ai_app,
the only account with DELETE on finsights_ai. Deletes in small batches, one transaction
each, so it can be stopped at any time and resumed by the next run.
"""

import argparse
import asyncio
import logging
import sys

from app.storage.mysql import chat_store
from app.storage.mysql.engine import get_chat_engine

logger = logging.getLogger("purge_chat_history")


async def purge(*, apply: bool, batch_size: int, max_batches: int) -> dict[str, int]:
    found = await chat_store.count_expired()
    logger.info(
        "expired: %(sessions)d sessions (%(deleted_sessions)d deleted by users), "
        "%(messages)d messages",
        found,
    )
    totals = {"sessions": 0, "messages": 0, "batches": 0}
    if not apply or not found["sessions"]:
        return {**found, **{f"purged_{k}": v for k, v in totals.items()}}

    for _ in range(max_batches):
        ids = await chat_store.expired_session_ids(batch_size)
        if not ids:
            break
        messages, sessions = await chat_store.purge_sessions(ids)
        totals["messages"] += messages
        totals["sessions"] += sessions
        totals["batches"] += 1
        if sessions == 0:
            break  # every candidate became active again; nothing left to do
    logger.info(
        "purged %(sessions)d sessions, %(messages)d messages in %(batches)d batches",
        totals,
    )
    return {**found, **{f"purged_{k}": v for k, v in totals.items()}}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true", help="delete (default: dry run)"
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        await purge(
            apply=args.apply, batch_size=args.batch_size, max_batches=args.max_batches
        )
    finally:
        await get_chat_engine().dispose()
    return 0


if __name__ == "__main__":
    # SelectorEventLoop: aiomysql cannot start TLS on Windows' default ProactorEventLoop.
    sys.exit(asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop))
