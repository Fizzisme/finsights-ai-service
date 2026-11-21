"""Cut a news article into chunks for embedding.

An article is the parent and its chunks are the children: a search ranks chunks, then answers
with the article they belong to. Paragraphs (blank-line separated) are packed together up to
`MAX_CHARS`; a paragraph longer than that is cut at sentence ends. Every chunk starts with the
article title, so a chunk from the middle of a story still says what story it is.

About 1,200 characters is ~450 tokens of Vietnamese: long enough to carry one point of the
story, short enough that a single figure or name is not diluted by the rest of the article.
There is no overlap between chunks; paragraph boundaries are the cut points.
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_CHARS = 1200
# The first ~14,000 characters: later parts of very long articles are rarely what a search
# is after, and they would dominate the embedding cost.
MAX_CHUNKS = 12

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _pack(pieces: list[str], separator: str, limit: int) -> list[str]:
    """Join consecutive pieces while the result stays within `limit`."""
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(separator) + len(piece) > limit:
            packed.append(current)
            current = piece
        else:
            current = f"{current}{separator}{piece}" if current else piece
    if current:
        packed.append(current)
    return packed


def _split_long(paragraph: str, limit: int) -> list[str]:
    """A paragraph over the limit, cut at sentence ends (and, as a last resort, anywhere)."""
    pieces: list[str] = []
    for chunk in _pack(_SENTENCE_END.split(paragraph), " ", limit):
        pieces.extend(chunk[i : i + limit] for i in range(0, len(chunk), limit))
    return pieces


def split_body(body: str | None) -> list[str]:
    """Chunks of an article body, without the title; empty when there is no body."""
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(body or "") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHARS:
            pieces.extend(_split_long(paragraph, MAX_CHARS))
        else:
            pieces.append(paragraph)
    return _pack(pieces, "\n\n", MAX_CHARS)[:MAX_CHUNKS]


def chunk_article(
    title: str, body: str | None, summary: str | None = None
) -> list[str]:
    """Texts to embed for one article: the title plus each chunk of the body.

    An article without a body falls back to its title and summary, so it stays searchable.
    """
    chunks = split_body(body)
    if not chunks:
        return [f"{title}\n{summary or ''}".strip()]
    return [f"{title}\n{chunk}" for chunk in chunks]


# --- Structure and meaning -----------------------------------------------------------------
#
# The size-based chunker above ignores how the article is laid out. These work on the stored
# content blocks (news_article_content_blocks): headings start a new section, a table is a
# chunk of its own, and inside a section paragraphs are grouped by size (`size_group`) or by
# meaning (`semantic_group`). Only ~13% of articles have headings, so meaning does most of the
# work for the rest.

MIN_CHARS = 300  # a semantic cut never leaves a chunk shorter than this
CUT_STDS = 0.5  # cut where similarity is this many deviations below the article's mean
MAX_TABLE_CHARS = MAX_CHARS
_TEXT_BLOCKS = {"paragraph", "list", "quote"}


@dataclass(frozen=True)
class Block:
    type: str
    text: str = ""
    table: Any = None  # decoded table_payload_json, {"rows": [[cell, ...], ...]}


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


def table_text(table: Any) -> str:
    """A table as readable lines, one row per line, cells joined by ' | '."""
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except ValueError:
            return ""
    rows = table.get("rows", []) if isinstance(table, dict) else []
    lines = [" | ".join(str(c).strip() for c in row if str(c).strip()) for row in rows]
    return "\n".join(line for line in lines if line)[:MAX_TABLE_CHARS]


def sections(blocks: Sequence[Block]) -> list[Section]:
    """Split blocks at headings; images, videos and line breaks carry no text and are dropped."""
    result = [Section("")]
    for block in blocks:
        text = block.text.strip()
        if block.type == "heading" and text:
            result.append(Section(text))
        elif block.type in _TEXT_BLOCKS and text:
            result[-1].paragraphs.append(text)
        elif block.type == "table" and (rendered := table_text(block.table)):
            result[-1].tables.append(rendered)
    return [s for s in result if s.paragraphs or s.tables]


def size_group(paragraphs: Sequence[str]) -> list[str]:
    """Pack consecutive paragraphs up to MAX_CHARS."""
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(
            _split_long(paragraph, MAX_CHARS)
            if len(paragraph) > MAX_CHARS
            else [paragraph]
        )
    return _pack(pieces, "\n\n", MAX_CHARS)


def semantic_group(
    paragraphs: Sequence[str], vectors: Sequence[Sequence[float]]
) -> list[str]:
    """Group paragraphs where the topic holds, cutting where it shifts.

    `vectors` are unit-length embeddings of the paragraphs. The similarity of each pair of
    neighbours is compared with the article's own average: a pair well below it marks a change
    of subject. A cut needs the chunk so far to reach MIN_CHARS, and a chunk is closed at
    MAX_CHARS regardless. Fewer than three paragraphs are left to `size_group`.
    """
    if len(paragraphs) < 3:
        return size_group(paragraphs)
    sims = [
        sum(a * b for a, b in zip(vectors[i], vectors[i + 1], strict=True))
        for i in range(len(paragraphs) - 1)
    ]
    mean = sum(sims) / len(sims)
    deviation = (sum((s - mean) ** 2 for s in sims) / len(sims)) ** 0.5
    threshold = mean - CUT_STDS * deviation
    groups: list[list[str]] = [[]]
    size = 0
    for i, paragraph in enumerate(paragraphs):
        shifted = i > 0 and sims[i - 1] < threshold and size >= MIN_CHARS
        if groups[-1] and (shifted or size + len(paragraph) > MAX_CHARS):
            groups.append([])
            size = 0
        groups[-1].append(paragraph)
        size += len(paragraph) + 2
    return [chunk for group in groups for chunk in size_group(group)]


def build_chunks(
    title: str,
    parts: Sequence[Section],
    group: Callable[[Sequence[str]], list[str]] = size_group,
    summary: str | None = None,
) -> list[str]:
    """Texts to embed: title, then the section heading (if any), then the chunk.

    No usable body falls back to title and summary, so the article stays searchable.
    """
    chunks: list[str] = []
    for part in parts:
        prefix = f"{title}\n{part.heading}\n" if part.heading else f"{title}\n"
        chunks.extend(
            f"{prefix}{text}" for text in [*group(part.paragraphs), *part.tables]
        )
    chunks = chunks[:MAX_CHUNKS]
    return chunks or [f"{title}\n{summary or ''}".strip()]
