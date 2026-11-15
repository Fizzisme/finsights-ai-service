from datetime import date, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NewsSearchInput(BaseModel):
    """Arguments of the search_news tool. Descriptions are shown to the LLM."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9]{2,10}$",
        description=(
            "Stock ticker such as VCB or FPT. Use when the user asks about a specific "
            "listed company. Case-insensitive."
        ),
    )
    query: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        description=(
            "Keyword or short phrase matched against article titles and summaries. "
            "Use for topics (e.g. 'lãi suất', 'trái phiếu') rather than a single company."
        ),
    )
    date_from: date | None = Field(
        default=None,
        description="Earliest publication date, inclusive (YYYY-MM-DD).",
    )
    date_to: date | None = Field(
        default=None,
        description="Latest publication date, inclusive (YYYY-MM-DD).",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of articles to return (1-20).",
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def upper_symbol(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot be later than date_to")
        return self


class NewsSecurityMatch(BaseModel):
    """A ticker linked to an article, with how the link was made."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    match_method: str
    review_status: str
    confidence: float | None = None


class NewsArticle(BaseModel):
    """One article. Body text is never exposed (content is link_only/unknown)."""

    model_config = ConfigDict(frozen=True)

    article_uid: str
    title: str
    summary: str | None = None
    # Naive datetime straight from MySQL; timezone not yet confirmed.
    published_at: datetime
    publisher: str
    url: str
    content_rights: str
    symbols: list[NewsSecurityMatch] = Field(default_factory=list)


class NewsSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    articles: list[NewsArticle] = Field(default_factory=list)
