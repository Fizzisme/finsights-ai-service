"""Request and response contracts for stateless text chat."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(min_length=1, max_length=16000)


class ChatResponse(BaseModel):
    answer: str
