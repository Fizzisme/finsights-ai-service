# Dagster news embedding pipeline

The Dagster code location is separate from the existing FastAPI/local RAG flow. The local Voyage integration and `app/rag/chunking/news.py` remain unchanged. Dagster runs `app/ingestion/news_embedding_pipeline.py` through `orchestration/definitions.py`.

## Backfill run config

Select `news_embedding_backfill_job` in Dagster and use this config for the requested window:

```yaml
ops:
  preprocess_chunk_embed_news:
    config:
      start_date: "2026-11-01"
      end_date: "2026-11-23"
      embedding_model: "@cf/baai/bge-m3"
      api_key:
        env: NEWS_EMBEDDING_API_KEY
      page_size: 100
      dry_run: false
```

The API key is resolved inside the run pod from `NEWS_EMBEDDING_API_KEY`; never place its value in Dagster run config. Cloudflare model IDs use `CLOUDFLARE_ACCOUNT_ID`. Omit `embedding_model` to use `voyage-4-large` (1024 dimensions). The pgvector table has one vector space at a time, so changing the model requires an intentional full re-embedding migration.

`start_date` and `end_date` are inclusive Vietnam calendar dates. The job converts them to `[00:00, next-day 00:00)` in `Asia/Ho_Chi_Minh`, then queries the MySQL Node 2 news source. A daily schedule is defined at 05:00 Vietnam time and processes the prior calendar day. It starts stopped until the Cloudflare/Voyage API key and PostgreSQL credentials are configured.

## Data-quality and idempotency behavior

Only published articles with licensed full text at both publisher and article level and an explicit financial relevance label are eligible. Rows marked irrelevant, excluded, pending, unknown, or review are skipped. The pipeline does not substitute a title or summary when the body is absent or too short.

Preprocessing extracts text from HTML if present, removes active/non-content elements and a conservative set of page chrome, decodes entities, normalizes Unicode and whitespace, preserves paragraph boundaries and financial figures, removes exact duplicated long paragraphs, and reports cleaning counters. It does not summarize or rewrite the story.

Chunking packs adjacent paragraphs, splits oversized paragraphs at sentence/word boundaries, adds a small overlap, prefixes every chunk with its title, and computes SHA-256 over the exact embedding input. There is no per-article chunk cap. The writer skips matching `(article_uid, chunk_index, content_hash, embedding_model)` rows, upserts changed chunks transactionally per page, and removes stale tail chunks. A failed provider request does not commit that page; completed pages can be safely rerun.

## Run summary

Dagster logs contain counts for rows read/skipped/eligible, source counts, cleaning changes, chunks created/unchanged/embedded/upserted, request count, and token count when the provider returns it. Cloudflare neuron consumption is authoritative in the account usage dashboard; the API response does not consistently expose a neuron counter.
