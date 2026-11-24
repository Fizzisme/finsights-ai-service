"""Paragraph-aware, overlap-based chunking for historical news ingestion.

The existing ``app.rag.chunking.news`` module is used by local experiments and is left
untouched. This module has no per-article chunk cap: every cleaned body passage is kept.
Character limits are conservative token proxies for Vietnamese; provider requests still
validate their own hard context and batch limits.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

MAX_CHARS = 1_400
OVERLAP_CHARS = 180
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class NewsChunk:
    chunk_index: int
    text: str
    content_hash: str


def _word_safe_tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value.strip()
    tail = value[-limit:]
    boundary = re.search(r"\s", tail)
    if boundary:
        tail = tail[boundary.end() :]
    return tail.strip()


def _split_long_piece(value: str, limit: int) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_END.split(value) if part.strip()]
    result: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                result.append(current)
                current = ""
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > limit:
                    result.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                result.append(piece)
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            result.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def split_body(
    body: str, *, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[str]:
    """Pack neighboring paragraphs, splitting oversized paragraphs at sentence/word ends."""
    if max_chars < 200 or overlap_chars < 0 or overlap_chars >= max_chars // 2:
        raise ValueError("invalid chunk/overlap sizes")
    atomic_limit = max_chars - overlap_chars - 4
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(body or "") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > atomic_limit:
            pieces.extend(_split_long_piece(paragraph, atomic_limit))
        else:
            pieces.append(paragraph)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(current)
            overlap = _word_safe_tail(current, overlap_chars)
            current = f"{overlap}\n\n{piece}" if overlap else piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_article(
    title: str,
    body: str,
    *,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[NewsChunk]:
    """Prefix every body chunk with the title and hash the final embedding input."""
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        raise ValueError("article title is required for chunking")
    prefix = f"Tiêu đề: {title}\n\n"
    body_limit = max_chars - len(prefix)
    if body_limit < 160:
        raise ValueError("article title is too long for the configured chunk size")
    bodies = split_body(
        body, max_chars=body_limit, overlap_chars=min(overlap_chars, body_limit // 4)
    )
    chunks: list[NewsChunk] = []
    for index, content in enumerate(bodies):
        text = f"{prefix}{content}".strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks.append(NewsChunk(chunk_index=index, text=text, content_hash=digest))
    return chunks
