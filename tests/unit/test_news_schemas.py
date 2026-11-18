from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.schemas.news import NewsArticle, NewsSearchInput, NewsSearchResult


def test_empty_input_is_valid_with_default_limit() -> None:
    args = NewsSearchInput()
    assert args.limit == 8
    assert args.symbol is None


def test_symbol_is_uppercased() -> None:
    assert NewsSearchInput(symbol="vcb").symbol == "VCB"


@pytest.mark.parametrize("symbol", ["VC B", "V", "VCB!", "ABCDEFGHIJK"])
def test_bad_symbol_rejected(symbol: str) -> None:
    with pytest.raises(ValidationError):
        NewsSearchInput(symbol=symbol)


@pytest.mark.parametrize("limit", [0, 21])
def test_limit_out_of_range_rejected(limit: int) -> None:
    with pytest.raises(ValidationError):
        NewsSearchInput(limit=limit)


def test_query_too_short_rejected() -> None:
    with pytest.raises(ValidationError):
        NewsSearchInput(query="a")


def test_date_order_checked() -> None:
    with pytest.raises(ValidationError):
        NewsSearchInput(date_from=date(2026, 5, 2), date_to=date(2026, 5, 1))
    ok = NewsSearchInput(date_from=date(2026, 5, 1), date_to=date(2026, 5, 1))
    assert ok.date_from == ok.date_to


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        NewsSearchInput.model_validate({"foo": 1})


def test_article_is_frozen_and_built_from_dict() -> None:
    article = NewsArticle.model_validate(
        {
            "article_uid": "a1",
            "title": "VCB tăng vốn",
            "summary": None,
            "published_at": datetime(2026, 5, 1, 8, 0),
            "publisher": "CafeF",
            "url": "https://example.com/a1",
            "content_rights": "link_only",
            "symbols": [
                {
                    "symbol": "VCB",
                    "match_method": "ticker_boundary",
                    "review_status": "auto",
                    "confidence": 0.9,
                }
            ],
        }
    )
    assert article.symbols[0].symbol == "VCB"
    with pytest.raises(ValidationError):
        article.title = "other"  # type: ignore[misc]
    assert NewsSearchResult(articles=[article]).articles[0] is article


def test_published_at_utc_converted_to_vietnam_day() -> None:
    article = NewsArticle(
        article_uid="a1",
        title="t",
        published_at=datetime(2026, 10, 31, 20, 0),  # naive UTC from MySQL
        publisher="CafeF",
        url="https://example.com/a1",
        content_rights="link_only",
    )
    assert article.published_at.isoformat() == "2026-11-01T03:00:00+07:00"
