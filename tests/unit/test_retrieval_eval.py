from datetime import date

from evals.runners.retrieval import Case, load_cases, score, summarize


def _case(relevant: set[str], irrelevant: set[str] | None = None) -> Case:
    return Case(
        id="c",
        category="lexical",
        question="q",
        query="q",
        date_from=date(2026, 10, 1),
        date_to=date(2026, 10, 31),
        relevant=frozenset(relevant),
        irrelevant=frozenset(irrelevant or ()),
    )


def _ranked(*uids: str) -> list[tuple[str, str]]:
    return [(uid, f"title {uid}") for uid in uids]


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
    }
    for c in cases:
        assert c.relevant, c.id
        assert not c.relevant & c.irrelevant, c.id
        assert c.date_from <= c.date_to, c.id
