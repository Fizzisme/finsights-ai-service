"""Chat sessions and messages in finsights_ai (Node 1). Schema is owned by finsights-coredb.

A turn is two `chat_messages` rows (speaker `user` and `assistant`) sharing a `turn` number;
the assistant row's `ai_trace` holds the digest later turns read (app/graph/history.py).
Every query is scoped by `user_uid`, so one user can never reach another's session.
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


async def recent_digests(session_id: int, limit: int) -> list[dict[str, Any]]:
    """Digests of the latest `limit` turns, each with its `turn`, oldest first."""
    rows = await _fetch(
        "SELECT turn, ai_trace FROM chat_messages "
        "WHERE session_id = :sid AND speaker = :assistant AND turn IS NOT NULL "
        "AND ai_trace IS NOT NULL ORDER BY turn DESC LIMIT :limit",
        {"sid": session_id, "assistant": ASSISTANT, "limit": limit},
    )
    digests = []
    for row in reversed(rows):
        trace = row["ai_trace"]
        trace = json.loads(trace) if isinstance(trace, str) else trace
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
