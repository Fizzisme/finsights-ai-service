"""Chat sessions and messages in finsights_ai (Node 1). Schema is owned by finsights-coredb.

A turn is two `chat_messages` rows (speaker `user` and `assistant`) sharing a `turn` number;
the assistant row's `ai_trace` holds the digest later turns read (app/graph/history.py).
Every user-facing query is scoped by `user_uid`, so one user can never reach another's
session. Retention (`purge_*`) is the only cross-user operation; it runs from a scheduled
script, never from a request.
"""

import json
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.storage.mysql.engine import get_chat_engine

ACTIVE, DELETED = "active", "deleted"
USER, ASSISTANT = "user", "assistant"
COMPLETE, FAILED = "complete", "failed"
TITLE_CHARS = 240


class TurnConflict(Exception):
    """Another request saved the same turn of this session first."""


async def _fetch(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    async with get_chat_engine().connect() as conn:
        result = await conn.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]


async def create_session(user_uid: str) -> dict[str, Any]:
    session_uid = str(uuid.uuid4())
    async with get_chat_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO chat_sessions (session_uid, user_uid, session_status) "
                "VALUES (:uid, :user, :status)"
            ),
            {"uid": session_uid, "user": user_uid, "status": ACTIVE},
        )
    session = await get_session(user_uid, session_uid)
    if session is None:
        raise RuntimeError("chat session not readable after insert")
    return session


async def get_session(user_uid: str, session_uid: str) -> dict[str, Any] | None:
    rows = await _fetch(
        "SELECT id, session_uid, title, created_at, updated_at FROM chat_sessions "
        "WHERE session_uid = :uid AND user_uid = :user AND session_status = :status",
        {"uid": session_uid, "user": user_uid, "status": ACTIVE},
    )
    return rows[0] if rows else None


async def list_sessions(user_uid: str, limit: int) -> list[dict[str, Any]]:
    return await _fetch(
        "SELECT session_uid, title, created_at, updated_at FROM chat_sessions "
        "WHERE user_uid = :user AND session_status = :status "
        "ORDER BY updated_at DESC, id DESC LIMIT :limit",
        {"user": user_uid, "status": ACTIVE, "limit": limit},
    )


async def list_messages(session_id: int) -> list[dict[str, Any]]:
    return await _fetch(
        "SELECT turn, speaker, message_text, message_status, created_at "
        "FROM chat_messages WHERE session_id = :sid AND turn IS NOT NULL "
        "ORDER BY turn, (speaker = :assistant), id",
        {"sid": session_id, "assistant": ASSISTANT},
    )


def _json(value: Any) -> Any:
    # aiomysql returns JSON columns as text.
    return json.loads(value) if isinstance(value, str) else value


async def latest_context(session_id: int) -> dict[str, Any] | None:
    """The newest rolling-summary checkpoint, stored on the turn that folded it."""
    rows = await _fetch(
        "SELECT JSON_EXTRACT(ai_trace, '$.context') AS context FROM chat_messages "
        "WHERE session_id = :sid AND speaker = :assistant AND turn IS NOT NULL "
        "AND JSON_CONTAINS_PATH(ai_trace, 'one', '$.context') "
        "ORDER BY turn DESC LIMIT 1",
        {"sid": session_id, "assistant": ASSISTANT},
    )
    context = _json(rows[0]["context"]) if rows else None
    return (
        context if isinstance(context, dict) and context.get("through_turn") else None
    )


async def digests_after(
    session_id: int, after_turn: int, limit: int
) -> list[dict[str, Any]]:
    """Digests of the latest `limit` turns after `after_turn`, each with its `turn`, oldest first."""
    rows = await _fetch(
        "SELECT turn, ai_trace FROM chat_messages "
        "WHERE session_id = :sid AND speaker = :assistant AND turn > :after "
        "AND ai_trace IS NOT NULL ORDER BY turn DESC LIMIT :limit",
        {
            "sid": session_id,
            "assistant": ASSISTANT,
            "after": after_turn,
            "limit": limit,
        },
    )
    digests = []
    for row in reversed(rows):
        trace = _json(row["ai_trace"])
        if isinstance(trace, dict) and isinstance(trace.get("digest"), dict):
            digests.append({"turn": row["turn"], **trace["digest"]})
    return digests


async def save_turn(
    session: dict[str, Any],
    *,
    question: str,
    answer: str,
    status: str,
    model_name: str | None,
    ai_trace: dict[str, Any],
    error_code: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> int:
    """Store the user and assistant rows of the next turn in one transaction."""
    try:
        async with get_chat_engine().begin() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            "SELECT COALESCE(MAX(turn), 0) + 1 AS next_turn "
                            "FROM chat_messages WHERE session_id = :sid"
                        ),
                        {"sid": session["id"]},
                    )
                )
                .mappings()
                .one()
            )
            turn = int(row["next_turn"])
            insert = text(
                "INSERT INTO chat_messages (message_uid, session_id, turn, speaker, "
                "message_text, message_status, model_name, error_code, ai_trace, "
                "prompt_tokens, completion_tokens) "
                "VALUES (:uid, :sid, :turn, :speaker, :text, :status, :model, :error, "
                ":trace, :prompt_tokens, :completion_tokens)"
            )
            await conn.execute(
                insert,
                {
                    "uid": str(uuid.uuid4()),
                    "sid": session["id"],
                    "turn": turn,
                    "speaker": USER,
                    "text": question,
                    "status": COMPLETE,
                    "model": None,
                    "error": None,
                    "trace": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                },
            )
            await conn.execute(
                insert,
                {
                    "uid": str(uuid.uuid4()),
                    "sid": session["id"],
                    "turn": turn,
                    "speaker": ASSISTANT,
                    "text": answer,
                    "status": status,
                    "model": model_name,
                    "error": error_code,
                    "trace": json.dumps(ai_trace, ensure_ascii=False),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
            # The first question names the session; updated_at orders the session list.
            await conn.execute(
                text(
                    "UPDATE chat_sessions SET title = COALESCE(title, :title), "
                    "updated_at = CURRENT_TIMESTAMP(3) WHERE id = :sid"
                ),
                {"title": question[:TITLE_CHARS], "sid": session["id"]},
            )
    except IntegrityError as exc:
        raise TurnConflict from exc
    return turn


async def delete_session(user_uid: str, session_uid: str) -> bool:
    async with get_chat_engine().begin() as conn:
        result = await conn.execute(
            text(
                "UPDATE chat_sessions SET session_status = :deleted, "
                "closed_at = CURRENT_TIMESTAMP(3), updated_at = CURRENT_TIMESTAMP(3) "
                "WHERE session_uid = :uid AND user_uid = :user AND session_status = :active"
            ),
            {
                "deleted": DELETED,
                "uid": session_uid,
                "user": user_uid,
                "active": ACTIVE,
            },
        )
        return result.rowcount > 0


# --- Retention -------------------------------------------------------------------------
# Sessions the user deleted are kept 30 days (to undo a mistake), then removed; sessions
# with no new turn for 180 days are removed. Deleting needs the DELETE privilege, which
# finsights_ai_app has.
DELETED_RETENTION_DAYS = 30
INACTIVE_RETENTION_DAYS = 180

# Constant SQL: the day counts are bound parameters. Server time is UTC, like the columns.
_EXPIRED = (
    "((session_status = :deleted "
    "AND COALESCE(closed_at, updated_at) < UTC_TIMESTAMP(3) - INTERVAL :deleted_days DAY) "
    "OR updated_at < UTC_TIMESTAMP(3) - INTERVAL :inactive_days DAY)"
)
_RETENTION = {
    "deleted": DELETED,
    "deleted_days": DELETED_RETENTION_DAYS,
    "inactive_days": INACTIVE_RETENTION_DAYS,
}


async def count_expired() -> dict[str, int]:
    """What a purge would remove now (dry run)."""
    rows = await _fetch(
        "SELECT COUNT(*) AS sessions, "
        "COALESCE(SUM(session_status = :deleted), 0) AS deleted_sessions, "
        "(SELECT COUNT(*) FROM chat_messages WHERE session_id IN "
        f"(SELECT id FROM chat_sessions WHERE {_EXPIRED})) AS messages "
        f"FROM chat_sessions WHERE {_EXPIRED}",
        _RETENTION,
    )
    return {key: int(value) for key, value in rows[0].items()}


async def expired_session_ids(limit: int) -> list[int]:
    rows = await _fetch(
        f"SELECT id FROM chat_sessions WHERE {_EXPIRED} ORDER BY id LIMIT :limit",
        {**_RETENTION, "limit": limit},
    )
    return [int(row["id"]) for row in rows]


async def purge_sessions(session_ids: list[int]) -> tuple[int, int]:
    """Hard-delete the given sessions and their messages, if still expired.

    The candidates are locked and re-checked inside the transaction, so a session that got
    a new turn after it was listed is kept. Returns (messages deleted, sessions deleted).
    """
    if not session_ids:
        return 0, 0
    names = [f"id{i}" for i in range(len(session_ids))]
    ids = dict(zip(names, session_ids, strict=True))
    in_list = ", ".join(f":{n}" for n in names)
    async with get_chat_engine().begin() as conn:
        rows = await conn.execute(
            text(
                f"SELECT id FROM chat_sessions WHERE id IN ({in_list}) AND {_EXPIRED} "
                "FOR UPDATE"
            ),
            {**ids, **_RETENTION},
        )
        confirmed = [int(row.id) for row in rows]
        if not confirmed:
            return 0, 0
        names = [f"c{i}" for i in range(len(confirmed))]
        params = dict(zip(names, confirmed, strict=True))
        in_list = ", ".join(f":{n}" for n in names)
        messages = await conn.execute(
            text(f"DELETE FROM chat_messages WHERE session_id IN ({in_list})"), params
        )
        sessions = await conn.execute(
            text(f"DELETE FROM chat_sessions WHERE id IN ({in_list})"), params
        )
        return messages.rowcount, sessions.rowcount
