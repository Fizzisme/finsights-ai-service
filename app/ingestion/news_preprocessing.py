"""Deterministic cleanup for licensed news bodies before chunking and embedding.

This module is intentionally separate from the online agent and the existing Voyage
demo. It treats article text as untrusted source data: it never executes markup or
instructions found in the article, and it preserves numbers, names, punctuation, and
paragraph order because those carry financial meaning.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup

_HTML_TAG = re.compile(r"</?[a-z][a-z0-9:-]*(?:\s[^<>]*?)?/?>", re.IGNORECASE)
_BLOCK_TAGS = (
    "article",
    "blockquote",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "p",
    "section",
    "tr",
)
_BOILERPLATE = (
    re.compile(
        r"^(?:tin liên quan|xem thêm|đọc thêm|bài liên quan|có thể bạn quan tâm)\s*:?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:đăng ký|theo dõi chúng tôi)\s+(?:nhận|để nhận|tại)\b.*$", re.IGNORECASE
    ),
    re.compile(
        r"^(?:chia sẻ bài viết|copy link|bình luận|ý kiến của bạn)\s*:?$", re.IGNORECASE
    ),
)
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}
_SPACE_RUN = re.compile(r"[\t\v\f\r ]+")
_PARAGRAPH_RUN = re.compile(r"\n{3,}")
_URL_ONLY = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)


@dataclass(frozen=True)
class CleanedNewsBody:
    text: str
    original_chars: int
    cleaned_chars: int
    paragraphs_before: int
    paragraphs_after: int
    duplicate_paragraphs_removed: int
    boilerplate_paragraphs_removed: int


def _html_to_text(value: str) -> str:
    """Extract visible text while retaining paragraph, heading, and list boundaries."""
    soup = BeautifulSoup(value, "html.parser")
    for node in soup.find_all(
        ("script", "style", "noscript", "iframe", "svg", "form", "button")
    ):
        node.decompose()

    # Remove common page chrome only when markup is present. The body field is normally
    # already article-scoped, so this deliberately uses a small, conservative selector set.
    for node in list(soup.find_all(True)):
        identity = " ".join(
            [str(node.get("id") or ""), " ".join(map(str, node.get("class") or []))]
        ).casefold()
        if any(
            term in identity
            for term in (
                "related-news",
                "related_article",
                "recommendation",
                "newsletter",
                "advertisement",
                "social-share",
            )
        ):
            node.decompose()

    for node in soup.find_all("br"):
        node.replace_with("\n")
    for node in soup.find_all(_BLOCK_TAGS):
        node.insert_before("\n\n")
        node.insert_after("\n\n")
    # Do not strip the separator strings: they carry block boundaries inserted above.
    # Spaces between remaining inline text nodes still prevent words joining together.
    return soup.get_text(" ", strip=False)


def _normalize_characters(value: str) -> str:
    value = unicodedata.normalize("NFC", html.unescape(value))
    for character in _ZERO_WIDTH:
        value = value.replace(character, "")
    value = (
        value.replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    # Retain newlines and tabs, but remove other control/format characters commonly left
    # by copy/paste or HTML extraction. This does not remove Vietnamese combining marks.
    value = "".join(
        character
        for character in value
        if character in "\n\t"
        or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    )

    lines: list[str] = []
    for line in value.split("\n"):
        line = _SPACE_RUN.sub(" ", line).strip()
        line = re.sub(r"\s+([,.;:!?%])", r"\1", line)
        # HTML text extraction can place adjacent inline fragments together. Restore a
        # missing separator after terminal punctuation without changing decimal numbers.
        line = re.sub(r"([.!?;:])(?=[^\s\d])", r"\1 ", line)
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    return _PARAGRAPH_RUN.sub("\n\n", "\n".join(lines)).strip()


def _paragraphs(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]


def clean_news_body(body: str | None, *, title: str = "") -> CleanedNewsBody:
    """Clean a scraped body without summarizing or rewriting its claims.

    Operations are deterministic: decode entities, remove markup/page chrome, normalize
    Unicode and whitespace, remove exact duplicate paragraphs, and drop only standalone
    navigation/subscription labels. A short or empty result is kept here so the caller can
    record a precise skip reason instead of silently substituting a title/summary.
    """
    raw = body or ""
    extracted = _html_to_text(raw) if _HTML_TAG.search(raw) else raw
    normalized = _normalize_characters(extracted)
    parts = _paragraphs(normalized)
    paragraphs_before = len(parts)

    normalized_title = _normalize_characters(title).casefold().strip()
    seen: set[str] = set()
    cleaned: list[str] = []
    duplicates = 0
    boilerplate = 0

    for paragraph in parts:
        paragraph = paragraph.strip()
        compact = _SPACE_RUN.sub(" ", paragraph).strip()
        folded = compact.casefold()

        if normalized_title and folded == normalized_title:
            duplicates += 1
            continue
        if _URL_ONLY.fullmatch(compact):
            boilerplate += 1
            continue
        if any(pattern.fullmatch(compact) for pattern in _BOILERPLATE):
            boilerplate += 1
            continue

        # Exact long-paragraph repeats usually come from duplicated DOM blocks. Do not
        # remove short repeated phrases, which may be meaningful in quotes or lists.
        if len(compact) >= 80 and folded in seen:
            duplicates += 1
            continue
        if len(compact) >= 80:
            seen.add(folded)
        cleaned.append(compact)

    text = "\n\n".join(cleaned).strip()
    return CleanedNewsBody(
        text=text,
        original_chars=len(raw),
        cleaned_chars=len(text),
        paragraphs_before=paragraphs_before,
        paragraphs_after=len(cleaned),
        duplicate_paragraphs_removed=duplicates,
        boilerplate_paragraphs_removed=boilerplate,
    )


def body_skip_reason(text: str) -> str | None:
    """Reject absent/near-empty bodies; never replace full text with a summary."""
    compact = text.strip()
    if not compact:
        return "empty_after_cleanup"
    if len(compact) < 80:
        return "body_too_short"
    if len(re.findall(r"\w+", compact, re.UNICODE)) < 10:
        return "too_few_words"
    return None
