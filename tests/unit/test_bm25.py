from app.rag.retrieval.bm25 import Bm25Index, reciprocal_rank_fusion, tokenize

DOCS = [
    "Coteccons ký hợp đồng mới trị giá 60.307 tỉ đồng",
    "Hòa Phát lãi quý 3 tăng mạnh nhờ thép",
    "Số thành viên Super App đạt 19,6 triệu người",
    "Lãi suất cho vay giảm tại nhiều ngân hàng",
    "Lãi suất huy động giảm tại nhiều ngân hàng",
]


def test_numbers_stay_whole_and_pairs_of_syllables_are_tokens() -> None:
    tokens = tokenize("Đạt 19,6 triệu và 12.038 tỷ")
    assert "19,6" in tokens and "12.038" in tokens
    assert "19,6 triệu" in tokens and "triệu và" in tokens
    assert "đạt" in tokens  # lower-cased, accents kept


def test_a_rare_exact_term_finds_its_document() -> None:
    index = Bm25Index(DOCS)
    scores = index.scores("Coteccons ký được bao nhiêu hợp đồng?")
    assert max(scores, key=lambda d: scores[d]) == 0
    scores = index.scores("thành viên Super App là bao nhiêu")
    assert max(scores, key=lambda d: scores[d]) == 2


def test_word_pairs_tell_apart_loan_and_deposit_rates() -> None:
    index = Bm25Index(DOCS)
    scores = index.scores("lãi suất cho vay")
    assert scores[3] > scores[4]


def test_documents_without_a_shared_token_are_absent_and_filter_applies() -> None:
    index = Bm25Index(DOCS)
    assert index.scores("zzz không có") == {}
    assert set(index.scores("lãi suất", allowed=lambda d: d == 4)) == {4}


def test_rank_fusion_rewards_agreement_and_ignores_score_scale() -> None:
    dense = {0: 0.9, 1: 0.8, 2: 0.1}  # cosine similarities
    keyword = {1: 37.0, 2: 5.0}  # BM25 scores, on another scale
    fused = reciprocal_rank_fusion([dense, keyword])
    # 1 is high in both lists; 2 is low in one and second in the other; 0 is only in one.
    assert max(fused, key=lambda d: fused[d]) == 1
    assert fused[2] > fused[0]


def test_rank_fusion_depth_limits_each_list() -> None:
    fused = reciprocal_rank_fusion([{i: float(i) for i in range(10)}], depth=3)
    assert set(fused) == {9, 8, 7}
