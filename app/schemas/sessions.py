"""Request and response contracts for chat sessions."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app.schemas.runs import RunResponse


def _as_utc(v: datetime) -> datetime:
    # finsights_ai stores naive UTC (server time_zone is UTC).
    return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class MessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class SessionSummary(BaseModel):
    session_uid: str
    title: str | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class SessionMessage(BaseModel):
    turn: int
    speaker: str
    text: str
    status: str
    created_at: UtcDatetime


class SessionDetail(SessionSummary):
    messages: list[SessionMessage] = Field(default_factory=list)


class TurnResponse(RunResponse):
    session_uid: str
    # None when the answer could not be saved; the answer is still returned.
    turn: int | None
