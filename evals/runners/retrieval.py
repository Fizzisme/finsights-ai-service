"""Retrieval eval for news topic search: does a retriever put on-topic articles first?

Dataset `evals/datasets/news_topics.jsonl`, one case per line:
- `question`: the user's question, for retrievers that take natural language (RAG);
- `query`: the keywords the agent would send to `search_news` for it;
- `date_from`/`date_to`: the period (Vietnam days);
- `relevant` / `irrelevant`: judged `article_uid`s.
`category` says what the case tests: `lexical` (the words appear as written), `multiword`
(words not next to each other), `paraphrase` (same meaning, other words) and `english`
(English question, Vietnamese news).

Labels were pooled from the current LIKE search and hand-written title searches, then judged
by title. Irrelevant labels are mostly hard negatives (same company or topic, wrong point).
A new retriever will return articles nobody judged: they are listed under `unjudged` so they
can be judged and added to the dataset, and they never count as wrong in precision.

Metrics, chosen for how the agent uses results (it lists at most 5 articles and searches
again when none fits):
- hit@5: a relevant article is in the top 5;
- precision@5: share of relevant articles among the judged ones in the top 5;
- mrr@10: 1 / rank of the first relevant article in the top 10 (0 if none).

    uv run --env-file .env python -m evals.runners.retrieval --retriever like
"""

import argparse
import asyncio
import heapq
import json
import math
import operator
import selectors
import sys
from array import array
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "news_topics.jsonl"
TOP_K = 10
HIT_K = 5


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    query: str
    date_from: date
    date_to: date
    relevant: frozenset[str]
    irrelevant: frozenset[str]


@dataclass
class Score:
    hit_at_5: bool
    precision_at_5: float | None  # None: nothing judged in the top 5
    mrr_at_10: float
    unjudged: list[dict[str, str]] = field(default_factory=list)


# A retriever ranks articles for a case: [(article_uid, title)], best first.
Retriever = Callable[[Case], Awaitable[list[tuple[str, str]]]]


def load_cases(path: Path = DATASET) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                id=raw["id"],
                category=raw["category"],
                question=raw["question"],
                query=raw["query"],
                date_from=date.fromisoformat(raw["date_from"]),
                date_to=date.fromisoformat(raw["date_to"]),
                relevant=frozenset(raw["relevant"]),
                irrelevant=frozenset(raw["irrelevant"]),
            )
        )
    return cases


def score(case: Case, ranked: list[tuple[str, str]]) -> Score:
    top = ranked[:TOP_K]
    uids = [uid for uid, _ in top]
    first = next((i for i, uid in enumerate(uids) if uid in case.relevant), None)
    judged = [uid for uid in uids[:HIT_K] if uid in case.relevant | case.irrelevant]
    return Score(
        hit_at_5=first is not None and first < HIT_K,
        precision_at_5=(
            sum(uid in case.relevant for uid in judged) / len(judged)
            if judged
            else None
        ),
        mrr_at_10=1 / (first + 1) if first is not None else 0.0,
        unjudged=[
            {"article_uid": uid, "title": title}
            for uid, title in top
            if uid not in case.relevant and uid not in case.irrelevant
        ],
    )


def summarize(scores: list[Score]) -> dict[str, Any]:
    precisions = [s.precision_at_5 for s in scores if s.precision_at_5 is not None]
    return {
        "cases": len(scores),
        "hit@5": round(mean(s.hit_at_5 for s in scores), 3),
        "precision@5": round(mean(precisions), 3) if precisions else None,
        "mrr@10": round(mean(s.mrr_at_10 for s in scores), 3),
        "unjudged@10": sum(len(s.unjudged) for s in scores),
    }


async def like_retriever(case: Case) -> list[tuple[str, str]]:
    """The current search_news: LIKE '%query%' on title/summary, newest first."""
    from app.schemas.news import NewsSearchInput
    from app.tools.sql.search_news import search_news

    result = await search_news(
        NewsSearchInput(
            query=case.query,
            date_from=case.date_from,
            date_to=case.date_to,
            limit=TOP_K,
        )
    )
    return [(a.article_uid, a.title) for a in result.articles]


SNAPSHOTS = Path(__file__).resolve().parents[1] / "snapshots"


class DenseIndex:
    """Naive dense retrieval: one vector per article (title + summary), searched in memory.

    Document vectors are cached in evals/snapshots (git-ignored) per model, dimensions and
    period, so a rerun only embeds the questions.
    """

    def __init__(self, date_from: date, date_to: date) -> None:
        self.date_from, self.date_to = date_from, date_to
        self.uids: list[str] = []
        self.titles: list[str] = []
        self.published: list[datetime] = []
        self.vectors: list[array[float]] = []

    async def load(self) -> None:
        from app.core.settings import get_settings
        from app.integrations.voyage.embeddings import embed
        from app.storage.mysql.engine import fetch_all
        from app.tools.sql.search_news import vn_day_start_utc

        rows = await fetch_all(
            "SELECT article_uid, title, summary, published_at FROM news_articles "
            "WHERE article_status = 'published' "
            "AND published_at >= :start AND published_at < :end ORDER BY id",
            {
                "start": vn_day_start_utc(self.date_from),
                "end": vn_day_start_utc(self.date_to + timedelta(days=1)),
            },
        )
        self.uids = [r["article_uid"] for r in rows]
        self.titles = [r["title"] for r in rows]
        self.published = [r["published_at"] for r in rows]

        settings = get_settings()
        stem = SNAPSHOTS / (
            f"dense_{settings.embedding_model}_{settings.embedding_dimensions}_"
            f"{self.date_from}_{self.date_to}"
        )
        meta, data = stem.with_suffix(".json"), stem.with_suffix(".f32")
        dims = settings.embedding_dimensions
        if (
            meta.exists()
            and data.exists()
            and json.loads(meta.read_text()) == self.uids
        ):
            flat = array("f")
            flat.frombytes(data.read_bytes())
            print(f"dense: {len(self.uids)} cached article vectors", file=sys.stderr)
        else:
            texts = [f"{r['title']}\n{r['summary'] or ''}".strip() for r in rows]
            print(f"dense: embedding {len(texts)} articles...", file=sys.stderr)
            result = await embed(texts, "document")
            print(f"dense: {result.tokens} tokens", file=sys.stderr)
            flat = array("f", (x for v in result.vectors for x in v))
            data.write_bytes(flat.tobytes())
            meta.write_text(json.dumps(self.uids))
        self.vectors = [
            _unit(flat[i * dims : (i + 1) * dims]) for i in range(len(self.uids))
        ]

    async def search(self, case: Case) -> list[tuple[str, str]]:
        from app.integrations.voyage.embeddings import embed
        from app.tools.sql.search_news import vn_day_start_utc

        query = _unit(array("f", (await embed([case.question], "query")).vectors[0]))
        start = vn_day_start_utc(case.date_from)
        end = vn_day_start_utc(case.date_to + timedelta(days=1))
        scored = [
            (sum(map(operator.mul, query, vector)), i)
            for i, vector in enumerate(self.vectors)
            if start <= self.published[i] < end
        ]
        top = heapq.nlargest(TOP_K, scored)
        return [(self.uids[i], self.titles[i]) for _, i in top]


def _unit(vector: array[float]) -> array[float]:
    """Scale to length 1, so a dot product is the cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return array("f", (x / norm for x in vector))


_dense: dict[tuple[date, date], DenseIndex] = {}


async def dense_retriever(case: Case) -> list[tuple[str, str]]:
    """Voyage embedding of the question against article vectors, cosine order."""
    key = (case.date_from, case.date_to)
    if key not in _dense:
        index = DenseIndex(*key)
        await index.load()
        _dense[key] = index
    return await _dense[key].search(case)


RETRIEVERS: dict[str, Retriever] = {"like": like_retriever, "dense": dense_retriever}


async def run(retriever: Retriever, cases: list[Case]) -> dict[str, Any]:
    results = []
    for case in cases:
        s = score(case, await retriever(case))
        results.append((case, s))
        p5 = "-" if s.precision_at_5 is None else f"{s.precision_at_5:.2f}"
        print(
            f"{case.id:<26} {case.category:<10} hit@5={int(s.hit_at_5)} "
            f"p@5={p5:<4} mrr={s.mrr_at_10:.2f} unjudged={len(s.unjudged)}"
        )
    by_category: dict[str, list[Score]] = defaultdict(list)
    for case, s in results:
        by_category[case.category].append(s)
    report = {
        "overall": summarize([s for _, s in results]),
        "by_category": {c: summarize(v) for c, v in sorted(by_category.items())},
        "cases": {case.id: asdict(s) for case, s in results},
    }
    print("\nby category:")
    for category, summary in report["by_category"].items():
        print(f"  {category:<10} {summary}")
    print(f"overall      {report['overall']}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=sorted(RETRIEVERS), default="like")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--out", help="write the full report (with unjudged articles) here")
    args = ap.parse_args()

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    # aiomysql with TLS needs the selector loop on Windows.
    report = asyncio.run(
        run(RETRIEVERS[args.retriever], cases),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
