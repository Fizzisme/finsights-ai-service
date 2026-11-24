"""Retrieval eval for news topic search: does a retriever put on-topic articles first?

Dataset `evals/datasets/news_topics.jsonl`, one case per line:
- `question`: the user's question, for retrievers that take natural language (RAG);
- `query`: the keywords the agent would send to `search_news` for it;
- `date_from`/`date_to`: the period (Vietnam days);
- `relevant` / `irrelevant`: judged `article_uid`s;
- `evidence` (detail cases): sentences copied from the body of the relevant article that
  hold the answer; none of them is in the title or summary.
`category` says what the case tests: `lexical` (the words appear as written), `multiword`
(words not next to each other), `paraphrase` (same meaning, other words) and `english`
(English question, Vietnamese news), `detail` (a fact that is only in the article body).

Labels were pooled from the current LIKE search and hand-written title searches, then judged
by title. Irrelevant labels are mostly hard negatives (same company or topic, wrong point).
A new retriever will return articles nobody judged: they are listed under `unjudged` so they
can be judged and added to the dataset, and they never count as wrong in precision.

Metrics, chosen for how the agent uses results (it lists at most 5 articles and searches
again when none fits):
- hit@5: a relevant article is in the top 5;
- precision@5: share of relevant articles among the judged ones in the top 5;
- mrr@10: 1 / rank of the first relevant article in the top 10 (0 if none);
- evidence@5 (detail cases only): the text a retriever returns for a relevant article in the
  top 5 contains the evidence. That is what the agent would read: title + summary for
  `like` and `dense`, the best chunk for `chunks`. A chunk cut in the middle of the evidence
  does not count.

    uv run --env-file .env python -m evals.runners.retrieval --retriever like

`--dataset evals/datasets/news_topics_v2.jsonl` runs the v2 set (87 cases, September to
20 November 2026, written by evals/builders/generate_cases.py and labelled by
evals/builders/judge.py). Its first line is `{"max_article_id": N}`: articles collected after
N are left out, so the corpus stays the same while the news pipeline keeps adding articles.
"""

import argparse
import asyncio
import hashlib
import heapq
import json
import math
import operator
import os
import selectors
import sys
from array import array
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
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
    evidence: tuple[str, ...] = ()


@dataclass
class Score:
    hit_at_5: bool
    precision_at_5: float | None  # None: nothing judged in the top 5
    mrr_at_10: float
    unjudged: list[dict[str, str]] = field(default_factory=list)
    evidence_at_5: bool | None = None  # None: the case has no evidence
    # Characters the agent would read for the top 5: what a gate that drops chunks saves.
    chars_at_5: int = 0


# A retriever ranks articles for a case: [(article_uid, title, text read)], best first.
Hit = tuple[str, str, str]
Retriever = Callable[[Case], Awaitable[list[Hit]]]


# The corpus limit of the loaded dataset (None: every article).
MAX_ARTICLE_ID: int | None = None


def load_cases(path: Path = DATASET) -> list[Case]:
    global MAX_ARTICLE_ID
    MAX_ARTICLE_ID = None
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if "max_article_id" in raw:
            MAX_ARTICLE_ID = int(raw["max_article_id"])
            continue
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
                evidence=tuple(raw.get("evidence", ())),
            )
        )
    return cases


def _squash(text: str) -> str:
    return " ".join(text.split())


def score(case: Case, ranked: list[Hit]) -> Score:
    top = ranked[:TOP_K]
    uids = [uid for uid, _, _ in top]
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
            for uid, title, _ in top
            if uid not in case.relevant and uid not in case.irrelevant
        ],
        evidence_at_5=(
            any(
                uid in case.relevant
                and any(_squash(e) in _squash(text) for e in case.evidence)
                for uid, _, text in top[:HIT_K]
            )
            if case.evidence
            else None
        ),
        chars_at_5=sum(len(text) for _, _, text in top[:HIT_K]),
    )


def summarize(scores: list[Score]) -> dict[str, Any]:
    precisions = [s.precision_at_5 for s in scores if s.precision_at_5 is not None]
    evidence = [s.evidence_at_5 for s in scores if s.evidence_at_5 is not None]
    return {
        "cases": len(scores),
        "hit@5": round(mean(s.hit_at_5 for s in scores), 3),
        "evidence@5": round(mean(evidence), 3) if evidence else None,
        "precision@5": round(mean(precisions), 3) if precisions else None,
        "mrr@10": round(mean(s.mrr_at_10 for s in scores), 3),
        "unjudged@10": sum(len(s.unjudged) for s in scores),
        "chars@5": round(mean(s.chars_at_5 for s in scores)),
    }


async def like_retriever(case: Case) -> list[Hit]:
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
    hits = [
        (a.article_uid, a.title, f"{a.title}\n{a.summary or ''}")
        for a in result.articles
    ]
    if MAX_ARTICLE_ID is not None and hits:
        from app.storage.mysql.engine import fetch_all

        names = ", ".join(f":u{k}" for k in range(len(hits)))
        rows = await fetch_all(
            f"SELECT article_uid FROM news_articles WHERE id <= :max_id "
            f"AND article_uid IN ({names})",
            {"max_id": MAX_ARTICLE_ID} | {f"u{k}": h[0] for k, h in enumerate(hits)},
        )
        kept = {r["article_uid"] for r in rows}
        hits = [h for h in hits if h[0] in kept]
    return hits


SNAPSHOTS = Path(__file__).resolve().parents[1] / "snapshots"
_rerank_cache: dict[str, list[tuple[int, float]]] | None = None


async def _cached_rerank(
    name: str,
    question: str,
    texts: list[str],
    call: Callable[[str, list[str]], Awaitable[Any]],
) -> Any:
    """`call(question, texts)` once per (reranker, question, texts), kept in evals/snapshots.

    Rerankers cost per call and give the same answer for the same input, so a rerun, or a
    sweep of gate thresholds over the same chunks, pays once. `name` holds the model version.
    """
    from app.integrations.voyage.rerank import Reranked

    global _rerank_cache
    path = SNAPSHOTS / "rerank_cache.jsonl"
    if _rerank_cache is None:
        _rerank_cache = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                _rerank_cache[row["key"]] = [(i, s) for i, s in row["ranked"]]
    digest = hashlib.sha256("\x00".join([name, question, *texts]).encode()).hexdigest()[
        :24
    ]
    if digest not in _rerank_cache:
        result = await call(question, texts)
        _rerank_cache[digest] = list(result.ranked)
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"key": digest, "ranked": result.ranked}) + "\n")
    return Reranked(_rerank_cache[digest], 0)


class DenseIndex:
    """Dense retrieval searched in memory. The mode says what one vector stands for:

    - `summary`: an article (title + summary), the naive baseline;
    - `chunks`: a size-based chunk of the body (app/rag/chunking/news.py);
    - `structure`: a chunk cut at headings, tables kept apart, packed by size inside a section;
    - `semantic`: as `structure`, but paragraphs are grouped by meaning: it embeds every
      paragraph first and cuts where neighbours stop being similar.

    An article scores as its best vector.

    Vectors are cached in evals/snapshots (git-ignored), keyed by a hash of the exact texts
    embedded, so a rerun only embeds the questions and a change to the chunker re-embeds.
    """

    def __init__(self, date_from: date, date_to: date, mode: str = "summary") -> None:
        self.date_from, self.date_to, self.mode = date_from, date_to, mode
        self.uids: list[str] = []
        self.titles: list[str] = []
        self.published: list[datetime] = []
        self.texts: list[str] = []  # text embedded for each vector
        self.owner: list[int] = []  # article index of each vector
        self.vectors: list[array[float]] = []
        self.bm25: Any = None  # built on first keyword search
        # Similarity of every vector to a question, computed once per question.
        self._similar: dict[str, dict[int, float]] = {}

    async def load(self) -> None:
        from app.rag.chunking.news import (
            build_chunks,
            chunk_article,
            sections,
            semantic_group,
            size_group,
        )
        from app.storage.mysql.engine import fetch_all
        from app.tools.sql.search_news import vn_day_start_utc

        # Bodies are ~14 MB for a month: pages keep each result small enough for the link.
        rows: list[dict[str, Any]] = []
        while True:
            page = await fetch_all(
                "SELECT a.id, a.article_uid, a.title, a.summary, a.published_at, "
                "c.body_plaintext FROM news_articles a "
                "LEFT JOIN news_article_contents c ON c.article_id = a.id "
                "WHERE a.article_status = 'published' AND a.id > :last "
                "AND a.id <= :max_id "
                "AND a.published_at >= :start AND a.published_at < :end "
                "ORDER BY a.id LIMIT 500",
                {
                    "last": rows[-1]["id"] if rows else 0,
                    "max_id": MAX_ARTICLE_ID or 2**63 - 1,
                    "start": vn_day_start_utc(self.date_from),
                    "end": vn_day_start_utc(self.date_to + timedelta(days=1)),
                },
            )
            rows.extend(page)
            if len(page) < 500:
                break
        self.uids = [r["article_uid"] for r in rows]
        self.titles = [r["title"] for r in rows]
        self.published = [r["published_at"] for r in rows]

        structured = self.mode in ("structure", "semantic")
        blocks = await _fetch_blocks([r["id"] for r in rows]) if structured else {}
        paragraph_vectors: dict[str, array[float]] = {}
        if self.mode == "semantic":
            # Only sections with three or more paragraphs can be cut by meaning.
            wanted = sorted(
                {
                    paragraph
                    for r in rows
                    for part in sections(blocks.get(r["id"], []))
                    if len(part.paragraphs) >= 3
                    for paragraph in part.paragraphs
                }
            )
            paragraph_vectors = dict(
                zip(wanted, await self._vectors("paragraphs", wanted), strict=True)
            )

        def by_meaning(paragraphs: Sequence[str]) -> list[str]:
            vectors = [
                paragraph_vectors[p] for p in paragraphs if p in paragraph_vectors
            ]
            return semantic_group(paragraphs, vectors)

        texts: list[str] = []
        for i, r in enumerate(rows):
            if structured:
                pieces = build_chunks(
                    r["title"],
                    sections(blocks.get(r["id"], [])),
                    by_meaning if self.mode == "semantic" else size_group,
                    r["summary"],
                )
            elif self.mode == "chunks":
                pieces = chunk_article(r["title"], r["body_plaintext"], r["summary"])
            else:
                pieces = [f"{r['title']}\n{r['summary'] or ''}".strip()]
            texts.extend(pieces)
            self.owner.extend([i] * len(pieces))
        self.texts = texts
        self.vectors = await self._vectors(self.mode, texts)

    async def _vectors(self, name: str, texts: list[str]) -> list[array[float]]:
        """Unit vectors of `texts`, from the snapshot of this exact list when there is one."""
        from app.core.settings import get_settings
        from app.integrations.voyage.embeddings import embed

        settings = get_settings()
        digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
        data = (
            SNAPSHOTS
            / (
                f"{name}_{settings.embedding_model}_{settings.embedding_dimensions}_"
                f"{self.date_from}_{self.date_to}_{digest}"
            )
        ).with_suffix(".f32")
        dims = settings.embedding_dimensions
        if data.exists():
            flat = array("f")
            flat.frombytes(data.read_bytes())
            print(f"{name}: {len(texts)} cached vectors", file=sys.stderr)
        else:
            print(f"{name}: embedding {len(texts)} texts...", file=sys.stderr)
            result = await embed(texts, "document")
            print(f"{name}: {result.tokens} tokens", file=sys.stderr)
            flat = array("f", (x for v in result.vectors for x in v))
            data.write_bytes(flat.tobytes())
        return [_unit(flat[i * dims : (i + 1) * dims]) for i in range(len(texts))]

    async def search(
        self,
        case: Case,
        per_article: int = 1,
        rerank_pool: int = 0,
        method: str = "dense",
        reranker: str = "voyage",
        gate: float = 0.0,
    ) -> list[Hit]:
        """`method`: `dense` (vector similarity), `bm25` (keywords) or `hybrid` (both, merged
        by rank). `reranker`: `voyage`, or Jev asked per passage (`jev-point`) or once over the
        whole list (`jev-list`). `gate`: when above 0, Jev is asked of every chunk in the
        ranked results whether it answers the question; chunks scoring below `gate` are dropped,
        and so are articles left with none. It only removes, never adds."""
        from app.integrations.voyage.embeddings import embed
        from app.rag.retrieval.bm25 import Bm25Index, reciprocal_rank_fusion
        from app.tools.sql.search_news import vn_day_start_utc

        start = vn_day_start_utc(case.date_from)
        end = vn_day_start_utc(case.date_to + timedelta(days=1))

        def in_range(item: int) -> bool:
            return start <= self.published[self.owner[item]] < end

        by_vector: dict[int, float] = {}
        by_keyword: dict[int, float] = {}
        if method in ("dense", "hybrid"):
            if case.question not in self._similar:
                query = _unit(
                    array("f", (await embed([case.question], "query")).vectors[0])
                )
                self._similar[case.question] = {
                    item: sum(map(operator.mul, query, vector))
                    for item, vector in enumerate(self.vectors)
                    if in_range(item)
                }
            by_vector = self._similar[case.question]
        if method in ("bm25", "hybrid"):
            if self.bm25 is None:
                self.bm25 = Bm25Index(self.texts)
            by_keyword = self.bm25.scores(case.question, in_range)
        merged = {
            "dense": by_vector,
            "bm25": by_keyword,
            "hybrid": reciprocal_rank_fusion([by_vector, by_keyword])
            if method == "hybrid"
            else {},
        }[method]
        scored: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for item, value in merged.items():
            scored[self.owner[item]].append((value, item))
        if rerank_pool:
            # The `rerank_pool` chunks closest to the question go to the reranker, which reads
            # question and chunk together and replaces the vector similarity as the score.
            from app.integrations.jev import client as jev
            from app.integrations.voyage import rerank as voyage

            # RERANK_MODEL picks another Voyage reranker (say a cheaper one); the cache name
            # carries the model, so results of different models never mix.
            voyage_model = os.environ.get("RERANK_MODEL", voyage.MODEL)

            async def voyage_rerank(query: str, texts: Sequence[str]) -> Any:
                return await voyage.rerank(query, texts, model=voyage_model)

            rerankers = {
                "voyage": (f"voyage:{voyage_model}", voyage_rerank),
                "jev-point": (f"jev-point:{jev.MODEL}", jev.rerank_pointwise),
                "jev-list": (f"jev-list:{jev.MODEL}", jev.rerank_listwise),
            }
            pool = heapq.nlargest(
                rerank_pool,
                ((value, item) for found in scored.values() for value, item in found),
            )
            name, call = rerankers[reranker]
            result = await _cached_rerank(
                name, case.question, [self.texts[i] for _, i in pool], call
            )
            scored = defaultdict(list)
            for index, relevance in result.ranked:
                item = pool[index][1]
                scored[self.owner[item]].append((relevance, item))
        # Articles rank by their best score; the text read is its `per_article` best.
        ranked = heapq.nlargest(TOP_K, ((max(v)[0], a) for a, v in scored.items()))
        picked = [
            (
                article,
                [
                    self.texts[i]
                    for _, i in heapq.nlargest(per_article, scored[article])
                ],
            )
            for _, article in ranked
        ]
        if gate:
            from app.integrations.jev import client as jev

            chunks = [chunk for _, found in picked for chunk in found]
            verdicts = await _cached_rerank(
                f"jev-point:{jev.MODEL}", case.question, chunks, jev.rerank_pointwise
            )
            relevance = dict(verdicts.ranked)
            kept: list[tuple[int, list[str]]] = []
            at = 0
            for article, found in picked:
                passing = [
                    chunk
                    for n, chunk in enumerate(found, start=at)
                    if relevance[n] >= gate
                ]
                at += len(found)
                if passing:
                    kept.append((article, passing))
            picked = kept
        return [
            (self.uids[article], self.titles[article], "\n---\n".join(found))
            for article, found in picked
        ]


async def _fetch_blocks(article_ids: list[int]) -> dict[int, list[Any]]:
    """Content blocks per article, in reading order, fetched in small pages."""
    from app.rag.chunking.news import Block
    from app.storage.mysql.engine import fetch_all

    blocks: dict[int, list[Any]] = defaultdict(list)
    for start in range(0, len(article_ids), 300):
        ids = article_ids[start : start + 300]
        names = ", ".join(f":i{k}" for k in range(len(ids)))
        rows = await fetch_all(
            "SELECT article_id, block_type, text_content, table_payload_json "
            "FROM news_article_content_blocks "
            f"WHERE article_id IN ({names}) ORDER BY article_id, block_order",
            {f"i{k}": v for k, v in enumerate(ids)},
        )
        for r in rows:
            blocks[r["article_id"]].append(
                Block(r["block_type"], r["text_content"] or "", r["table_payload_json"])
            )
    return blocks


def _unit(vector: array[float]) -> array[float]:
    """Scale to length 1, so a dot product is the cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return array("f", (x / norm for x in vector))


_dense: dict[tuple[date, date, str], DenseIndex] = {}


def _dense_retriever(
    mode: str,
    per_article: int = 1,
    rerank_pool: int = 0,
    method: str = "dense",
    reranker: str = "voyage",
    gate: float = 0.0,
) -> Retriever:
    async def retrieve(case: Case) -> list[Hit]:
        # One index per calendar month, shared by the cases of that month; each case then
        # keeps only its own period.
        first = case.date_from.replace(day=1)
        following = (first + timedelta(days=32)).replace(day=1)
        key = (first, max(case.date_to, following - timedelta(days=1)), mode)
        if key not in _dense:
            index = DenseIndex(*key)
            await index.load()
            _dense[key] = index
        return await _dense[key].search(
            case, per_article, rerank_pool, method, reranker, gate
        )

    return retrieve


RETRIEVERS: dict[str, Retriever] = {
    "like": like_retriever,
    # Voyage embedding of the question against article vectors, cosine order.
    "dense": _dense_retriever("summary"),
    "chunks": _dense_retriever("chunks"),
    # The same chunks, reading the best 2 or 3 of each article instead of only the best.
    "chunks-top2": _dense_retriever("chunks", 2),
    "chunks-top3": _dense_retriever("chunks", 3),
    # Vector search finds the N closest chunks, Voyage rerank re-scores them, top 2 per article.
    "rerank20": _dense_retriever("chunks", 2, 20),
    "rerank40": _dense_retriever("chunks", 2, 40),
    "rerank80": _dense_retriever("chunks", 2, 80),
    # Keywords only, and keywords + vectors merged by rank (RRF), without and with rerank.
    "bm25": _dense_retriever("chunks", 2, 0, "bm25"),
    "hybrid": _dense_retriever("chunks", 2, 0, "hybrid"),
    "hybrid-rerank80": _dense_retriever("chunks", 2, 80, "hybrid"),
    # The same 80 chunks re-scored by Jev: one yes/no question per chunk, or one choice over all.
    "jev-point80": _dense_retriever("chunks", 2, 80, "dense", "jev-point"),
    "jev-list80": _dense_retriever("chunks", 2, 80, "dense", "jev-list"),
    # Voyage rerank 80, then Jev drops the returned chunks that do not answer the question,
    # below the given probability.
    "gate10": _dense_retriever("chunks", 2, 80, "dense", "voyage", 0.1),
    "gate30": _dense_retriever("chunks", 2, 80, "dense", "voyage", 0.3),
    "gate50": _dense_retriever("chunks", 2, 80, "dense", "voyage", 0.5),
    "gate70": _dense_retriever("chunks", 2, 80, "dense", "voyage", 0.7),
    "structure": _dense_retriever("structure"),
    "structure-rerank80": _dense_retriever("structure", 2, 80),
    "semantic": _dense_retriever("semantic"),
}


async def run(retriever: Retriever, cases: list[Case]) -> dict[str, Any]:
    results = []
    for case in cases:
        s = score(case, await retriever(case))
        results.append((case, s))
        p5 = "-" if s.precision_at_5 is None else f"{s.precision_at_5:.2f}"
        print(
            f"{case.id:<26} {case.category:<10} hit@5={int(s.hit_at_5)} "
            f"p@5={p5:<4} mrr={s.mrr_at_10:.2f} unjudged={len(s.unjudged)}"
            + ("" if s.evidence_at_5 is None else f" evidence@5={int(s.evidence_at_5)}")
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
    ap.add_argument("--dataset", type=Path, default=DATASET)
    args = ap.parse_args()

    cases = load_cases(args.dataset)
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
