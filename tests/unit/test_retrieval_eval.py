from array import array
from datetime import date, datetime

import pytest

from evals.runners.retrieval import (
    Case,
    DenseIndex,
    Hit,
    load_cases,
    score,
    summarize,
)


def _case(
    relevant: set[str],
    irrelevant: set[str] | None = None,
    evidence: tuple[str, ...] = (),
) -> Case:
    return Case(
        id="c",
        category="lexical",
        question="q",
        query="q",
        date_from=date(2026, 10, 1),
        date_to=date(2026, 10, 31),
        relevant=frozenset(relevant),
        irrelevant=frozenset(irrelevant or ()),
        evidence=evidence,
    )


def _ranked(*uids: str) -> list[Hit]:
    return [(uid, f"title {uid}", f"text {uid}") for uid in uids]


def test_evidence_must_be_in_the_text_read_of_a_relevant_article() -> None:
    case = _case({"r1"}, evidence=("đạt 78 tỷ đồng",))
    cut = [("r1", "t", "Doanh thu bất động sản đạt 78")]  # chunk cut mid-evidence
    whole = [("r1", "t", "Doanh thu bất động sản\n đạt 78  tỷ đồng, tăng 52 lần")]
    other = [("x1", "t", "đạt 78 tỷ đồng")]  # right text, wrong article
    assert score(case, cut).evidence_at_5 is False
    assert score(case, whole).evidence_at_5 is True  # whitespace is ignored
    assert score(case, other).evidence_at_5 is False
    assert score(_case({"r1"}), whole).evidence_at_5 is None  # no evidence in the case


def test_evidence_is_averaged_over_detail_cases_only() -> None:
    hit = score(_case({"r1"}, evidence=("a",)), [("r1", "t", "a")])
    miss = score(_case({"r1"}, evidence=("a",)), [("r1", "t", "b")])
    plain = score(_case({"r1"}), _ranked("r1"))
    assert summarize([hit, miss, plain])["evidence@5"] == 0.5
    assert summarize([plain])["evidence@5"] is None


def test_first_relevant_at_rank_three() -> None:
    s = score(_case({"r1", "r2"}, {"x1", "x2"}), _ranked("x1", "x2", "r1", "r2", "u1"))
    assert s.hit_at_5 and s.mrr_at_10 == 1 / 3
    assert s.precision_at_5 == 0.5  # 2 relevant of 4 judged
    assert s.unjudged == [{"article_uid": "u1", "title": "title u1"}]


def test_unjudged_articles_do_not_count_as_wrong() -> None:
    s = score(_case({"r1"}), _ranked("u1", "u2", "r1"))
    assert s.precision_at_5 == 1.0 and len(s.unjudged) == 2


def test_relevant_only_after_top_five_is_no_hit() -> None:
    s = score(_case({"r1"}), _ranked("u1", "u2", "u3", "u4", "u5", "r1"))
    assert not s.hit_at_5 and s.mrr_at_10 == 1 / 6


def test_empty_result_scores_zero() -> None:
    s = score(_case({"r1"}), [])
    assert (s.hit_at_5, s.precision_at_5, s.mrr_at_10) == (False, None, 0.0)
    summary = summarize([s])
    assert summary["hit@5"] == 0 and summary["precision@5"] is None


def test_dataset_is_well_formed() -> None:
    cases = load_cases()
    assert len(cases) >= 20
    assert len({c.id for c in cases}) == len(cases)
    assert {c.category for c in cases} == {
        "lexical",
        "multiword",
        "paraphrase",
        "english",
        "detail",
    }
    for c in cases:
        assert c.relevant, c.id
        assert bool(c.evidence) == (c.category == "detail"), c.id
        assert not c.relevant & c.irrelevant, c.id
        assert c.date_from <= c.date_to, c.id


@pytest.mark.anyio
async def test_dense_search_reads_the_best_chunks_of_each_article(monkeypatch) -> None:
    from app.integrations.voyage import embeddings

    async def fake_embed(texts, input_type, **kwargs):
        return embeddings.Embeddings([[1.0, 0.0]], 1)

    monkeypatch.setattr(embeddings, "embed", fake_embed)
    index = DenseIndex(date(2026, 10, 1), date(2026, 10, 31), "chunks")
    index.uids, index.titles = ["a1", "a2"], ["A1", "A2"]
    index.published = [datetime(2026, 10, 15)] * 2
    # Article 0 has three chunks (similarity 0.6, 0.9, 0.2); article 1 has one (0.8).
    index.owner = [0, 0, 0, 1]
    index.texts = ["a-mid", "a-best", "a-low", "b-only"]
    index.vectors = [
        array("f", v) for v in ([0.6, 0.8], [0.9, 0.4], [0.2, 0.9], [0.8, 0.6])
    ]

    case = _case({"a1"})
    one = await index.search(case, per_article=1)
    two = await index.search(case, per_article=2)

    assert [uid for uid, _, _ in one] == ["a1", "a2"]  # ranked by best chunk
    assert one[0][2] == "a-best"
    assert two[0][2] == "a-best\n---\na-mid" and two[1][2] == "b-only"
