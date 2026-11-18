from datetime import UTC, date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# MySQL stores published_at as naive UTC. Users mean Vietnam calendar days
# (UTC+7, no daylight saving), so filters and output are converted to it.
VN_UTC_OFFSET = timedelta(hours=7)
VN_TZ = timezone(VN_UTC_OFFSET, "Asia/Ho_Chi_Minh")


def _utc_to_vn(v: datetime) -> datetime:
    if v.tzinfo is None:
        v = v.replace(tzinfo=UTC)
    return v.astimezone(VN_TZ)


# A datetime read from MySQL (naive UTC), exposed in Vietnam time (+07:00).
VnDatetime = Annotated[datetime, AfterValidator(_utc_to_vn)]


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
    publisher: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_-]{2,32}$",
        description=(
            "Only articles from this news source, by code: cafef, vnexpress, tuoitre, "
            "thanhnien, vneconomy, baomoi. Use only when the user names a source."
        ),
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description=(
            "Maximum number of articles to return (1-20). Keep the default unless the "
            "user asks for more articles than it."
        ),
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def upper_symbol(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("publisher", mode="before")
    @classmethod
    def lower_publisher(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

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
    published_at: VnDatetime
    publisher: str
    url: str
    content_rights: str
    symbols: list[NewsSecurityMatch] = Field(default_factory=list)


class NewsSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    articles: list[NewsArticle] = Field(default_factory=list)
    # Publication time of the newest article in the whole database: news after
    # this moment has not been collected yet.
    data_as_of: VnDatetime | None = None


class CountNewsInput(BaseModel):
    """Arguments of the count_news tool. Descriptions are shown to the LLM."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    symbol: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9]{2,10}$",
        description="Only count articles linked to this stock ticker, such as VCB. Case-insensitive.",
    )
    query: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Only count articles whose title or summary contains this keyword.",
    )
    date_from: date | None = Field(
        default=None, description="Earliest publication date, inclusive (YYYY-MM-DD)."
    )
    date_to: date | None = Field(
        default=None, description="Latest publication date, inclusive (YYYY-MM-DD)."
    )
    publisher: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_-]{2,32}$",
        description=(
            "Only articles from this news source, by code: cafef, vnexpress, tuoitre, "
            "thanhnien, vneconomy, baomoi. Use only when the user names a source."
        ),
    )
    group_by: Literal["none", "symbol", "publisher", "day"] = Field(
        default="none",
        description=(
            "How to break the count down: 'none' = total only; 'symbol' = per ticker "
            "(use for 'which stock is mentioned most'); 'publisher' = per news source; "
            "'day' = per publication day."
        ),
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=31,
        description="Maximum number of groups returned when group_by is not 'none' (1-31).",
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def upper_symbol(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("publisher", mode="before")
    @classmethod
    def lower_publisher(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot be later than date_to")
        return self


class NewsCountGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    count: int


class CountNewsResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Distinct published articles matching the filters (an article tagged with
    # several tickers counts once here, but once per ticker inside `groups`).
    total: int
    groups: list[NewsCountGroup] = Field(default_factory=list)
    data_as_of: VnDatetime | None = None


class GetNewsArticleInput(BaseModel):
    """Arguments of the get_news_article tool. Give exactly one of article_uid / url."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    article_uid: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9-]{8,64}$",
        description="The article_uid returned by search_news.",
    )
    url: str | None = Field(
        default=None,
        pattern=r"^https?://\S+$",
        max_length=2048,
        description="The exact article link (url) returned by search_news.",
    )

    @model_validator(mode="after")
    def exactly_one(self) -> Self:
        if (self.article_uid is None) == (self.url is None):
            raise ValueError("provide exactly one of article_uid or url")
        return self


class GetNewsArticleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    article: NewsArticle | None = None


class ResolveSecurityInput(BaseModel):
    """Arguments of the resolve_security tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(
        min_length=2,
        max_length=100,
        description=(
            "Company name, brand or ticker, e.g. 'Hòa Phát', 'Techcombank' or 'HPG'. "
            "Accents and case are ignored."
        ),
    )


class ResolvedSecurity(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    aliases: list[str] = Field(default_factory=list)


class ResolveSecurityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    matches: list[ResolvedSecurity] = Field(default_factory=list)
    note: str = (
        "Only tickers that appear in the news database can be resolved; "
        "an empty result does not mean the company is not listed."
    )
