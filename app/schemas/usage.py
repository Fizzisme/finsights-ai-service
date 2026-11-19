from pydantic import BaseModel, Field

from app.schemas.sessions import UtcDatetime


class QuotaWindow(BaseModel):
    used: int = Field(description="Cache-weighted tokens used in the current window.")
    limit: int | None = Field(description="Tokens allowed per window; null = no limit.")
    resets_at: UtcDatetime | None = Field(
        description="When the current window ends; null = no window open."
    )


class UsageResponse(BaseModel):
    five_hour: QuotaWindow
    weekly: QuotaWindow
