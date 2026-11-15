"""Request and response contracts for one agent run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class ToolCallTrace(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool


class RunResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallTrace]
