"""Read, clean, chunk, embed, and upsert a bounded news publication-date window."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from app.ingestion.news_preprocessing import body_skip_reason, clean_news_body
from app.integrations.embeddings.client import (
    EmbeddedBatch,
    EmbeddingClient,
)
from app.rag.chunking.news_backfill import NewsChunk, chunk_article
from app.storage.mysql.engine import fetch_all
from app.storage.postgres.news_chunks import NewsChunkStore

logger = logging.getLogger("finsights.ingestion.news_embeddings")
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
PAGE_SIZE_DEFAULT = 100
FINANCIAL_LABELS = {
    "financial",
    "financial_relevant",
    "financially_relevant",
    "finance_related",
    "relevant",
    "related",
    "yes",
    "in_scope",
}
REVIEW_LABELS = {"needs_review", "review", "unclassified", "unknown", "pending", ""}
EXCLUDED_LABELS = {
    "irrelevant",
    "not_relevant",
    "not_financial",
    "non_financial",
    "nonfinancial",
    "unrelated",
    "out_of_scope",
    "no",
    "exclude",
    "excluded",
    "reject",
    "rejected",
    "drop",
    "blocked",
}


class NewsEmbeddingRunError(RuntimeError):
    """An ingestion run cannot safely proceed."""


class _ChunkStore(Protocol):
    def assert_model_compatible(self, model: str) -> None: ...

    def existing_for_articles(
        self, article_uids: Sequence[str]
    ) -> Mapping[str, Mapping[int, Any]]: ...

    def remove_articles(self, article_uids: Sequence[str]) -> int: ...

    def persist_page(
        self, *, articles: Sequence[dict], embedding_model: str
    ) -> int: ...

    def close(self) -> None: ...


class _Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> EmbeddedBatch: ...

    def close(self) -> None: ...


def local_date_window_utc(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Convert inclusive Vietnam calendar dates to the UTC boundaries MySQL stores."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    start_local = datetime.combine(start, time.min, tzinfo=VIETNAM_TZ)
    end_local_exclusive = datetime.combine(
        end + timedelta(days=1), time.min, tzinfo=VIETNAM_TZ
    )
    return (
        start_local.astimezone(UTC).replace(tzinfo=None),
        end_local_exclusive.astimezone(UTC).replace(tzinfo=None),
    )


def _quality_eligible(row: dict[str, Any]) -> tuple[bool, str | None]:
    relevance = (
        str(row.get("financial_relevance") or "")
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    decision = (
        str(row.get("publication_decision") or "")
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if relevance in EXCLUDED_LABELS or decision in EXCLUDED_LABELS:
        return False, "explicitly_non_financial_or_excluded"
    if relevance not in FINANCIAL_LABELS:
        return False, "financial_quality_label_missing_or_review"
    if decision in REVIEW_LABELS:
        return False, "publication_decision_missing_or_review"
    return True, None


def _utc_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


async def _fetch_page(
    start_utc: datetime, end_utc: datetime, after_id: int, page_size: int
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            a.id AS article_id,
            a.article_uid,
            a.title,
            a.subtitle,
            a.summary,
            a.published_at,
            a.original_section,
            p.code AS source_code,
            p.name AS source_name,
            p.content_rights,
            c.body_plaintext,
            c.content_format,
            c.word_count,
            q.financial_relevance,
            q.topic_label,
            q.publication_decision,
            q.confidence AS quality_confidence,
            (SELECT GROUP_CONCAT(DISTINCT s.symbol_at_publication ORDER BY s.symbol_at_publication SEPARATOR ',')
               FROM news_article_securities s
              WHERE s.article_id = a.id AND s.review_status <> 'rejected') AS symbols
        FROM news_articles a
        JOIN news_publishers p ON p.id = a.publisher_id
        JOIN news_article_contents c ON c.article_id = a.id
        LEFT JOIN news_article_quality q ON q.article_id = a.id
        WHERE a.id > :after_id
          AND a.published_at >= :start_utc
          AND a.published_at < :end_utc
          AND a.article_status = 'published'
          AND a.content_rights = 'licensed_full_text'
          AND p.content_rights = 'licensed_full_text'
          AND c.body_plaintext IS NOT NULL
          AND CHAR_LENGTH(TRIM(c.body_plaintext)) > 0
        ORDER BY a.id
        LIMIT :page_size
    """
    return await fetch_all(
        sql,
        {
            "after_id": after_id,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "page_size": page_size,
        },
    )


class _NewsPageReader:
    """Keep SQLAlchemy's async pool on one event loop for the whole Dagster run."""

    def __init__(self) -> None:
        self._runner = asyncio.Runner()
        self._engine = None

    def fetch(
        self, start_utc: datetime, end_utc: datetime, after_id: int, page_size: int
    ) -> list[dict[str, Any]]:
        from app.storage.mysql.engine import get_engine

        self._engine = get_engine()
        try:
            return self._runner.run(
                _fetch_page(start_utc, end_utc, after_id, page_size)
            )
        finally:
            # Embedding and writing a page can leave the MySQL connection idle for a
            # long time. Drop pooled sockets before that idle period can leave a stale
            # connection that hangs during the next pre-ping/query.
            self._runner.run(self._engine.dispose())

    def close(self) -> None:
        try:
            if self._engine is not None:
                self._runner.run(self._engine.dispose())
        finally:
            self._runner.close()


def _chunks_for_article(
    row: dict[str, Any],
) -> tuple[list[NewsChunk], dict[str, int], str | None]:
    cleaned = clean_news_body(
        str(row.get("body_plaintext") or ""), title=str(row.get("title") or "")
    )
    reason = body_skip_reason(cleaned.text)
    stats = {
        "original_chars": cleaned.original_chars,
        "cleaned_chars": cleaned.cleaned_chars,
        "paragraphs_before": cleaned.paragraphs_before,
        "paragraphs_after": cleaned.paragraphs_after,
        "duplicate_paragraphs_removed": cleaned.duplicate_paragraphs_removed,
        "boilerplate_paragraphs_removed": cleaned.boilerplate_paragraphs_removed,
    }
    if reason:
        return [], stats, reason
    return chunk_article(str(row.get("title") or ""), cleaned.text), stats, None


def run_news_embedding_window(
    *,
    start_date: str,
    end_date: str,
    embedding_model: str = "voyage-4-large",
    api_key: str,
    cloudflare_account_id: str | None = None,
    page_size: int = PAGE_SIZE_DEFAULT,
    dry_run: bool = False,
    store: _ChunkStore | None = None,
    embedder: _Embedder | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one resumable page at a time; each page is committed only after embeddings return."""
    if not 1 <= page_size <= 500:
        raise ValueError("page_size must be between 1 and 500")
    if not api_key.strip() and not dry_run:
        raise ValueError("api_key is required unless dry_run is enabled")
    start_utc, end_utc = local_date_window_utc(start_date, end_date)
    own_store = store is None and not dry_run
    own_embedder = embedder is None and not dry_run
    if store is None and not dry_run:
        store = NewsChunkStore()
    if embedder is None and not dry_run:
        embedder = EmbeddingClient(
            model=embedding_model,
            api_key=api_key,
            account_id=cloudflare_account_id,
        )

    counts: Counter[str] = Counter()
    per_source: Counter[str] = Counter()
    clean_counts: Counter[str] = Counter()
    chunk_total = 0
    batch_total = 0
    request_total = 0
    provider_tokens = 0
    provider_tokens_complete = True
    after_id = 0
    page_reader = _NewsPageReader()

    def report(message: str) -> None:
        if progress is None:
            logger.info(message)
        else:
            progress(message)

    try:
        if store is not None:
            store.assert_model_compatible(embedding_model)
        report(
            "News embedding run started "
            f"window={start_date}..{end_date} "
            f"UTC=[{start_utc.isoformat()},{end_utc.isoformat()}) "
            f"model={embedding_model} page_size={page_size} dry_run={dry_run}"
        )

        while True:
            report(f"Fetching page={batch_total + 1} after_id={after_id}")
            page = page_reader.fetch(start_utc, end_utc, after_id, page_size)
            if not page:
                break
            batch_total += 1
            after_id = max(int(row["article_id"]) for row in page)
            report(
                f"Fetched page={batch_total} articles={len(page)} after_id={after_id}"
            )
            candidate_rows: list[dict[str, Any]] = []
            chunks_by_uid: dict[str, list[NewsChunk]] = {}
            article_uids_to_remove: list[str] = []
            for row in page:
                counts["articles_read"] += 1
                source = str(row.get("source_code") or "unknown")
                eligible, quality_reason = _quality_eligible(row)
                if not eligible:
                    counts[quality_reason or "quality_skipped"] += 1
                    article_uids_to_remove.append(str(row["article_uid"]))
                    continue
                chunks, clean_stats, skip_reason = _chunks_for_article(row)
                for key, value in clean_stats.items():
                    clean_counts[key] += value
                if skip_reason:
                    counts[skip_reason] += 1
                    article_uids_to_remove.append(str(row["article_uid"]))
                    continue
                uid = str(row["article_uid"])
                row["_chunks"] = chunks
                row["_published_at_utc"] = _utc_aware(row.get("published_at"))
                candidate_rows.append(row)
                chunks_by_uid[uid] = chunks
                per_source[source] += 1
                counts["articles_eligible"] += 1
                counts["chunks_created"] += len(chunks)
                chunk_total += len(chunks)

            if not dry_run and store is not None and article_uids_to_remove:
                counts["chunks_removed"] += store.remove_articles(
                    article_uids_to_remove
                )

            if dry_run:
                report(
                    f"Dry-run page={batch_total} articles={len(page)} "
                    f"eligible={len(candidate_rows)} "
                    f"chunks={sum(len(v) for v in chunks_by_uid.values())} "
                    f"after_id={after_id}"
                )
                continue

            assert store is not None and embedder is not None
            existing = store.existing_for_articles(
                [str(row["article_uid"]) for row in candidate_rows]
            )
            changed_by_uid: dict[str, list[NewsChunk]] = {}
            changed_flat: list[tuple[str, NewsChunk]] = []
            for row in candidate_rows:
                uid = str(row["article_uid"])
                known = existing.get(uid, {})
                changed = [
                    chunk
                    for chunk in chunks_by_uid[uid]
                    if chunk.chunk_index not in known
                    or known[chunk.chunk_index].content_hash != chunk.content_hash
                    or known[chunk.chunk_index].embedding_model != embedding_model
                ]
                changed_by_uid[uid] = changed
                changed_flat.extend((uid, chunk) for chunk in changed)
                counts["chunks_unchanged"] += len(chunks_by_uid[uid]) - len(changed)

            vectors_by_uid: dict[str, dict[int, list[float]]] = defaultdict(dict)
            page_request_start = request_total
            if changed_flat:
                report(f"Embedding page={batch_total} chunks={len(changed_flat)}")
                result = embedder.embed([chunk.text for _uid, chunk in changed_flat])
                request_total += result.requests
                if result.input_tokens is None:
                    provider_tokens_complete = False
                else:
                    provider_tokens += result.input_tokens
                for (uid, chunk), vector in zip(
                    changed_flat, result.vectors, strict=True
                ):
                    vectors_by_uid[uid][chunk.chunk_index] = vector
                    counts["chunks_embedded"] += 1
                report(
                    f"Embedded page={batch_total} vectors={len(result.vectors)} "
                    f"requests={result.requests}"
                )

            # Commit article rows and stale-tail cleanup only after every provider request
            # for the page succeeded. Previous pages remain safely resumable by hash.
            report(f"Persisting page={batch_total} articles={len(candidate_rows)}")
            persisted_articles: list[dict[str, Any]] = []
            for row in candidate_rows:
                uid = str(row["article_uid"])
                changed_rows = [
                    {
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "content_hash": chunk.content_hash,
                        "embedding": vectors_by_uid[uid][chunk.chunk_index],
                    }
                    for chunk in changed_by_uid[uid]
                ]
                persisted_articles.append(
                    {
                        "article_uid": uid,
                        "published_at": row["_published_at_utc"],
                        "chunk_count": len(chunks_by_uid[uid]),
                        "changed_chunks": changed_rows,
                    }
                )
            counts["chunks_upserted"] += store.persist_page(
                articles=persisted_articles,
                embedding_model=embedding_model,
            )
            counts["articles_persisted"] += len(persisted_articles)

            report(
                f"Committed page={batch_total} read={len(page)} "
                f"eligible={len(candidate_rows)} changed_chunks={len(changed_flat)} "
                f"requests={request_total - page_request_start} after_id={after_id}"
            )

        summary = {
            "start_date": start_date,
            "end_date": end_date,
            "embedding_model": embedding_model,
            "dry_run": dry_run,
            "pages": batch_total,
            "counts": dict(counts),
            "articles_by_source": dict(per_source),
            "cleaning": dict(clean_counts),
            "chunks_created": chunk_total,
            "provider_requests": request_total,
            "provider_input_tokens": provider_tokens
            if provider_tokens_complete
            else None,
        }
        report(f"News embedding run finished summary={summary}")
        return summary
    finally:
        try:
            page_reader.close()
        finally:
            try:
                if own_embedder and embedder is not None:
                    embedder.close()
            finally:
                if own_store and store is not None:
                    store.close()
