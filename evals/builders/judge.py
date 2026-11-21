"""Label the v2 draft: pool what the retrievers return, have an LLM judge each article.

For every case, the top 10 of each retriever in `POOL` (and the seed article) are judged by
an LLM against the question: relevant or not, with a short reason. Judgments are cached in
evals/snapshots, so a rerun only judges new pairs. The output is the dataset the runner
reads, plus a random sample of judgments for a human to check.

    uv run --env-file .env python -m evals.builders.judge
"""

import asyncio
import hashlib
import json
import random
import re
import selectors
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from evals.runners import retrieval

DATASETS = Path(__file__).resolve().parents[1] / "datasets"
DRAFT = DATASETS / "news_topics_v2.draft.jsonl"
OUT = DATASETS / "news_topics_v2.jsonl"
CACHE = retrieval.SNAPSHOTS / "judgments_v2.jsonl"
SAMPLE = Path(__file__).resolve().parents[1] / "reports" / "judge_sample_v2.md"
POOL = ["like", "dense", "chunks-top2", "bm25", "hybrid", "rerank80", "hybrid-rerank80"]
# Facts deep in an article must be seen (bodies average ~2,200 characters).
BODY_CHARS = 8000
# Part of the cache key: a change to the prompt or to what the judge reads judges again.
JUDGE_VERSION = "2"
SAMPLE_SIZE = 50

PROMPT = """Bạn chấm nhãn cho bộ đánh giá tìm kiếm tin tài chính.
Câu hỏi của người dùng (tìm trong tin từ {date_from} đến {date_to}):
{question}

Bài báo (đăng {published}, giờ Việt Nam):
Tiêu đề: {title}
Tóm tắt: {summary}
Nội dung:
{body}

Bài này có LIÊN QUAN không, tức là người trả lời câu hỏi có muốn đọc bài này không vì nó
chứa thông tin trả lời (một phần hoặc toàn bộ) câu hỏi?
- Cùng công ty/chủ đề nhưng nói chuyện khác (khác chỉ tiêu, khác sự kiện) là KHÔNG liên quan.
- Câu hỏi hỏi một dữ kiện cụ thể: chỉ liên quan nếu bài chứa dữ kiện đó.
- Câu hỏi bằng tiếng Anh vẫn chấm như tiếng Việt.
Chỉ trả về JSON: {{"relevant": true hoặc false, "reason": "một câu ngắn"}}"""


def _key(case: retrieval.Case, uid: str) -> str:
    text = f"{JUDGE_VERSION}\x00{case.question}\x00{uid}"
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    rows = (json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines())
    return {r["key"]: r for r in rows}


async def _articles(uids: list[str]) -> dict[str, dict[str, Any]]:
    from app.storage.mysql.engine import fetch_all

    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(uids), 200):
        part = uids[start : start + 200]
        names = ", ".join(f":u{k}" for k in range(len(part)))
        rows = await fetch_all(
            "SELECT a.article_uid, a.title, a.summary, a.published_at, "
            "LEFT(c.body_plaintext, :chars) AS body FROM news_articles a "
            "LEFT JOIN news_article_contents c ON c.article_id = a.id "
            f"WHERE a.article_uid IN ({names})",
            {"chars": BODY_CHARS} | {f"u{k}": v for k, v in enumerate(part)},
        )
        found.update({r["article_uid"]: r for r in rows})
    return found


async def _judge(model: Any, case: retrieval.Case, article: dict[str, Any]) -> dict:
    reply = await model.ainvoke(
        PROMPT.format(
            date_from=case.date_from,
            date_to=case.date_to,
            question=case.question,
            published=f"{article['published_at'] + timedelta(hours=7):%Y-%m-%d %H:%M}",
            title=article["title"],
            summary=article["summary"] or "",
            body=article["body"] or "",
        )
    )
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = json.loads(match.group(0)) if match else {}
    if not isinstance(raw.get("relevant"), bool):
        raise TypeError(f"unreadable judgment: {text[:200]}")
    return {"relevant": raw["relevant"], "reason": str(raw.get("reason", ""))}


async def main() -> None:
    from app.integrations.llm.factory import get_chat_model

    cases = retrieval.load_cases(DRAFT)
    drafts = {
        json.loads(line).get("id"): json.loads(line)
        for line in DRAFT.read_text(encoding="utf-8").splitlines()
    }
    pooled: dict[str, set[str]] = {c.id: {drafts[c.id]["seed"]} for c in cases}
    for name in POOL:
        print(f"pooling {name}...", file=sys.stderr)
        for case in cases:
            hits = await retrieval.RETRIEVERS[name](case)
            pooled[case.id] |= {uid for uid, _, _ in hits[: retrieval.TOP_K]}
    articles = await _articles(sorted(set().union(*pooled.values())))

    cache = _load_cache()
    todo = [
        (case, uid)
        for case in cases
        for uid in sorted(pooled[case.id])
        if _key(case, uid) not in cache and uid in articles
    ]
    print(
        f"{sum(map(len, pooled.values()))} pairs, {len(todo)} to judge", file=sys.stderr
    )
    model = get_chat_model("deepseek", "standard")
    limit = asyncio.Semaphore(8)

    async def one(case: retrieval.Case, uid: str) -> None:
        async with limit:
            verdict: dict[str, Any] | None = None
            for attempt in range(3):
                try:
                    verdict = await _judge(model, case, articles[uid])
                    break
                except Exception as exc:  # noqa: BLE001 - retried, then skipped
                    if attempt == 2:
                        print(f"  {case.id} {uid}: {exc}", file=sys.stderr)
            if verdict is None:
                return
            row = {"key": _key(case, uid), "case": case.id, "uid": uid, **verdict}
            cache[row["key"]] = row
            with CACHE.open("a", encoding="utf-8") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(c, u) for c, u in todo))

    with OUT.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"max_article_id": retrieval.MAX_ARTICLE_ID}) + "\n")
        for case in cases:
            labels = {
                uid: cache[_key(case, uid)]["relevant"]
                for uid in pooled[case.id]
                if _key(case, uid) in cache
            }
            row = dict(drafts[case.id])
            # The question was written from the seed, which answers it by construction.
            labels[row["seed"]] = True
            row["relevant"] = sorted(u for u, rel in labels.items() if rel)
            row["irrelevant"] = sorted(u for u, rel in labels.items() if not rel)
            if not row["relevant"]:
                print(f"  {case.id}: no relevant article", file=sys.stderr)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # A random sample for a human to check the judge against.
    pairs = [(c, u) for c in cases for u in sorted(pooled[c.id]) if _key(c, u) in cache]
    sample = random.Random(7).sample(pairs, min(SAMPLE_SIZE, len(pairs)))
    lines = [
        "# Judge sample (v2)",
        "",
        "Mark `ok` or the right label in the last column.",
        "",
        "| # | question | article | LLM | reason | check |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for n, (case, uid) in enumerate(sample, start=1):
        verdict = cache[_key(case, uid)]
        title = articles[uid]["title"].replace("|", "/")
        lines.append(
            f"| {n} | {case.question.replace('|', '/')} | {title} (`{uid[:8]}`) | "
            f"{'relevant' if verdict['relevant'] else 'irrelevant'} | "
            f"{verdict['reason'].replace('|', '/')} |  |"
        )
    SAMPLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT}\n-> {SAMPLE}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )
