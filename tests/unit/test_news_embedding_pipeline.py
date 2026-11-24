from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import app.ingestion.news_embedding_pipeline as pipeline
from app.ingestion.news_embedding_pipeline import (
    _NewsPageReader,
    _quality_eligible,
    local_date_window_utc,
    run_news_embedding_window,
)
from app.ingestion.news_preprocessing import body_skip_reason, clean_news_body
from app.integrations.embeddings.client import EmbeddedBatch, EmbeddingClient
from app.rag.chunking.news_backfill import chunk_article, split_body


def test_local_date_window_is_inclusive_in_vietnam_timezone() -> None:
    start, end = local_date_window_utc("2026-11-01", "2026-11-23")
    assert start == datetime(2026, 10, 31, 17, tzinfo=UTC).replace(tzinfo=None)
    assert end == datetime(2026, 11, 23, 17, tzinfo=UTC).replace(tzinfo=None)


def test_local_date_window_rejects_reverse_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        local_date_window_utc("2026-11-24", "2026-11-23")


def test_news_page_reader_disposes_pooled_connection_after_each_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEngine:
        disposed = 0

        async def dispose(self) -> None:
            self.disposed += 1

    engine = FakeEngine()

    async def fetch_page(*_args) -> list[dict[str, Any]]:
        return [{"article_id": 17}]

    monkeypatch.setattr("app.storage.mysql.engine.get_engine", lambda: engine)
    monkeypatch.setattr(pipeline, "_fetch_page", fetch_page)
    reader = _NewsPageReader()
    try:
        assert reader.fetch(datetime.now(UTC), datetime.now(UTC), 0, 10) == [
            {"article_id": 17}
        ]
        assert engine.disposed == 1
    finally:
        reader.close()


def test_preprocessing_removes_markup_duplicates_and_standalone_page_chrome() -> None:
    body = """<article>
      <p>Doanh thu tăng 12% trong quý III. Biên lợi nhuận cải thiện thêm hai điểm phần trăm.</p>
      <p>Doanh thu tăng <strong>12%</strong> trong quý III. Biên lợi nhuận cải thiện thêm hai điểm phần trăm.</p>
      <p>Tin liên quan</p>
      <p>Công ty dự kiến đầu tư thêm 500 tỷ đồng.</p>
      <script>malicious();</script>
    </article>"""

    cleaned = clean_news_body(body, title="Bài viết")

    assert "Doanh thu tăng 12% trong quý III." in cleaned.text
    assert "Công ty dự kiến đầu tư thêm 500 tỷ đồng." in cleaned.text
    assert "malicious" not in cleaned.text
    assert "Tin liên quan" not in cleaned.text
    assert cleaned.duplicate_paragraphs_removed == 1
    assert cleaned.boilerplate_paragraphs_removed == 1
    assert body_skip_reason(cleaned.text) is None


def test_preprocessing_does_not_replace_short_body_with_title_or_summary() -> None:
    cleaned = clean_news_body("Nội dung quá ngắn", title="Tiêu đề")
    assert cleaned.text == "Nội dung quá ngắn"
    assert body_skip_reason(cleaned.text) == "body_too_short"


def test_preprocessing_detects_inline_only_html() -> None:
    cleaned = clean_news_body("<strong>Lợi nhuận</strong> tăng 12% trong quý này.")
    assert "<strong>" not in cleaned.text
    assert "Lợi nhuận" in cleaned.text


def test_chunker_keeps_all_paragraphs_with_overlap_and_stable_hashes() -> None:
    paragraphs = [
        f"Đoạn {index}: " + ("thông tin tài chính quan trọng. " * 32)
        for index in range(8)
    ]
    body = "\n\n".join(paragraphs)

    chunks = chunk_article("Kết quả kinh doanh", body, max_chars=600, overlap_chars=80)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 600 for chunk in chunks)
    assert all(
        chunk.text.startswith("Tiêu đề: Kết quả kinh doanh\n\n") for chunk in chunks
    )
    assert all(
        chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        for chunk in chunks
    )
    assert all(
        f"Đoạn {index}:" in "\n".join(chunk.text for chunk in chunks)
        for index in range(8)
    )
    assert chunks == chunk_article(
        "Kết quả kinh doanh", body, max_chars=600, overlap_chars=80
    )


def test_chunker_splits_an_oversized_paragraph_at_sentence_boundaries() -> None:
    body = " ".join(f"Câu số {index} có dữ liệu tài chính." for index in range(90))
    chunks = split_body(body, max_chars=400, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)
    assert "Câu số 89" in "\n".join(chunks)


def test_quality_gate_only_accepts_explicit_financial_and_publishable_rows() -> None:
    assert _quality_eligible(
        {"financial_relevance": "financial", "publication_decision": "publish"}
    ) == (True, None)
    assert (
        _quality_eligible(
            {"financial_relevance": "non_financial", "publication_decision": "publish"}
        )[0]
        is False
    )
    assert (
        _quality_eligible(
            {"financial_relevance": None, "publication_decision": "publish"}
        )[0]
        is False
    )
    assert (
        _quality_eligible(
            {"financial_relevance": "financial", "publication_decision": "needs_review"}
        )[0]
        is False
    )


def test_cloudflare_client_validates_1024d_response_without_logging_input() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": {"data": [[0.125] * 1024]},
                "result_info": {"tokens": 17},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EmbeddingClient(
        model="@cf/baai/bge-m3",
        api_key="fake-test-key",
        account_id="test-account",
        http_client=http_client,
    )
    try:
        result = client.embed(["bài báo thử nghiệm"])
    finally:
        client.close()
        http_client.close()

    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == 1024
    assert result.requests == 1
    url = seen.get("url")
    assert isinstance(url, str) and url.endswith("/@cf/baai/bge-m3")
    assert seen["body"] == {"text": ["bài báo thử nghiệm"]}


def test_embedding_client_runs_batches_concurrently_and_preserves_order() -> None:
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak_active
        texts = json.loads(request.content)["text"]
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.05)
        vectors = [[float(text)] * 1024 for text in texts]
        with lock:
            active -= 1
        return httpx.Response(
            200,
            json={"success": True, "result": {"data": vectors}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EmbeddingClient(
        model="@cf/baai/bge-m3",
        api_key="fake-test-key",
        account_id="test-account",
        http_client=http_client,
    )
    try:
        result = client.embed([str(index) for index in range(130)])
    finally:
        client.close()
        http_client.close()

    assert result.requests == 3
    assert len(result.vectors) == 130
    assert [vector[0] for vector in result.vectors] == [float(i) for i in range(130)]
    assert peak_active > 1


def test_cloudflare_client_rejects_models_incompatible_with_vector_column() -> None:
    with pytest.raises(ValueError, match="1024 dimensions"):
        EmbeddingClient(
            model="@cf/google/embeddinggemma-2", api_key="fake", account_id="test"
        )


def test_pipeline_upserts_once_and_skips_unchanged_chunks_on_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "article_id": 11,
        "article_uid": "f4c7a128-1111-4111-8111-111111111111",
        "title": "Kết quả kinh doanh quý III",
        "body_plaintext": "Công ty ghi nhận doanh thu tăng trong quý. " * 12,
        "published_at": datetime(2026, 11, 10, 5, tzinfo=UTC),
        "source_code": "cafef",
        "financial_relevance": "financial",
        "publication_decision": "publish",
    }

    class FakeReader:
        def __init__(self) -> None:
            self.read = False

        def fetch(self, *_args) -> list[dict]:
            if self.read:
                return []
            self.read = True
            return [row]

        def close(self) -> None:
            pass

    class FakeStore:
        def __init__(self) -> None:
            self.saved: dict[str, dict[int, SimpleNamespace]] = {}
            self.persist_calls = 0

        def assert_model_compatible(self, model: str) -> None:
            assert model == "@cf/baai/bge-m3"

        def existing_for_articles(
            self, article_uids: Sequence[str]
        ) -> Mapping[str, Mapping[int, Any]]:
            return {uid: self.saved.get(uid, {}) for uid in article_uids}

        def persist_page(
            self,
            *,
            articles: Sequence[dict],
            embedding_model: str,
        ) -> int:
            self.persist_calls += 1
            saved_count = 0
            for article in articles:
                changed_chunks = article["changed_chunks"]
                self.saved[article["article_uid"]] = {
                    chunk["chunk_index"]: SimpleNamespace(
                        chunk_index=chunk["chunk_index"],
                        content_hash=chunk["content_hash"],
                        embedding_model=embedding_model,
                    )
                    for chunk in changed_chunks
                }
                assert article["chunk_count"] >= len(changed_chunks)
                saved_count += len(changed_chunks)
            return saved_count

        def remove_articles(self, article_uids: Sequence[str]) -> int:
            assert not article_uids
            return 0

        def close(self) -> None:
            pass

    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: Sequence[str]) -> EmbeddedBatch:
            self.calls += 1
            return EmbeddedBatch(
                vectors=[[0.25] * 1024 for _ in texts],
                input_tokens=22,
                requests=1,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "_NewsPageReader", FakeReader)
    store = FakeStore()
    embedder = FakeEmbedder()
    progress: list[str] = []

    first = run_news_embedding_window(
        start_date="2026-11-01",
        end_date="2026-11-23",
        embedding_model="@cf/baai/bge-m3",
        api_key="fake-test-key",
        cloudflare_account_id="test-account",
        store=store,
        embedder=embedder,
        progress=progress.append,
    )
    second = run_news_embedding_window(
        start_date="2026-11-01",
        end_date="2026-11-23",
        embedding_model="@cf/baai/bge-m3",
        api_key="fake-test-key",
        cloudflare_account_id="test-account",
        store=store,
        embedder=embedder,
    )

    assert first["counts"]["chunks_embedded"] == 1
    assert first["counts"]["chunks_upserted"] == 1
    assert second["counts"]["chunks_unchanged"] == 1
    assert second["provider_requests"] == 0
    assert embedder.calls == 1
    assert store.persist_calls == 2
    assert any(message.startswith("Fetching page=1") for message in progress)
    assert any(message.startswith("Committed page=1") for message in progress)
