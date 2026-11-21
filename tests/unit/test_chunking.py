from app.rag.chunking.news import (
    MAX_CHARS,
    MAX_CHUNKS,
    Block,
    build_chunks,
    chunk_article,
    sections,
    semantic_group,
    size_group,
    split_body,
    table_text,
)


def test_short_article_is_one_chunk_with_its_title() -> None:
    chunks = chunk_article("Tiêu đề", "Đoạn một.\n\nĐoạn hai.")
    assert chunks == ["Tiêu đề\nĐoạn một.\n\nĐoạn hai."]


def test_paragraphs_are_packed_without_being_split() -> None:
    para = "a" * 500
    chunks = split_body(f"{para}\n\n{para}\n\n{para}")
    # Two paragraphs fit in one chunk (1002 chars), the third starts the next.
    assert [len(c) for c in chunks] == [1002, 500]


def test_long_paragraph_is_cut_at_sentence_ends() -> None:
    sentence = "Câu này dài vừa phải." + " " * 0
    body = " ".join([sentence] * 200)  # one paragraph of ~4,400 characters
    chunks = split_body(body)
    assert len(chunks) > 1 and all(len(c) <= MAX_CHARS for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # no sentence cut in half


def test_text_without_any_break_is_hard_cut() -> None:
    chunks = split_body("x" * (MAX_CHARS * 2 + 10))
    assert [len(c) for c in chunks] == [MAX_CHARS, MAX_CHARS, 10]


def test_every_chunk_carries_the_title() -> None:
    body = "\n\n".join("đoạn " * 150 for _ in range(6))
    chunks = chunk_article("Hòa Phát lãi lớn", body)
    assert len(chunks) > 1
    assert all(c.startswith("Hòa Phát lãi lớn\n") for c in chunks)


def test_very_long_article_is_capped() -> None:
    body = "\n\n".join("y" * 1000 for _ in range(MAX_CHUNKS * 3))
    assert len(split_body(body)) == MAX_CHUNKS


def test_missing_body_falls_back_to_title_and_summary() -> None:
    assert chunk_article("Tiêu đề", None, "Tóm tắt.") == ["Tiêu đề\nTóm tắt."]
    assert chunk_article("Tiêu đề", "  \n\n ", None) == ["Tiêu đề"]
    assert split_body(None) == []


# --- structure and meaning ----------------------------------------------------------------


def test_headings_start_sections_and_media_is_dropped() -> None:
    blocks = [
        Block("image"),
        Block("paragraph", "Mở đầu."),
        Block("heading", "Tiền về ngân hàng"),
        Block("paragraph", "Đoạn một."),
        Block("line_break"),
        Block("list", "- ý a\n- ý b"),
        Block("video"),
    ]
    parts = sections(blocks)
    assert [(p.heading, p.paragraphs) for p in parts] == [
        ("", ["Mở đầu."]),
        ("Tiền về ngân hàng", ["Đoạn một.", "- ý a\n- ý b"]),
    ]


def test_table_becomes_readable_lines_and_its_own_chunk() -> None:
    table = {"rows": [["", ""], ["2-1-2012", "41.8"], ["20-8-2012", "42.58"]]}
    assert table_text(table) == "2-1-2012 | 41.8\n20-8-2012 | 42.58"
    assert table_text('{"rows": [["a", "b"]]}') == "a | b"  # as MySQL returns it
    assert table_text("not json") == ""
    parts = sections([Block("paragraph", "Giá vàng."), Block("table", table=table)])
    chunks = build_chunks("Vàng", parts)
    assert chunks == ["Vàng\nGiá vàng.", "Vàng\n2-1-2012 | 41.8\n20-8-2012 | 42.58"]


def test_chunks_carry_title_and_heading() -> None:
    parts = sections([Block("heading", "Mục A"), Block("paragraph", "Nội dung.")])
    assert build_chunks("Tiêu đề", parts) == ["Tiêu đề\nMục A\nNội dung."]


def test_no_text_blocks_falls_back_to_summary() -> None:
    chunks = build_chunks("Tiêu đề", sections([Block("image")]), summary="Tóm tắt.")
    assert chunks == ["Tiêu đề\nTóm tắt."]


def test_semantic_group_cuts_where_the_subject_changes() -> None:
    para = "x" * 300  # three of them fit one chunk (904 chars)
    paragraphs = [para] * 6
    # Three paragraphs about one thing (vector A), three about another (vector B).
    a, b = (1.0, 0.0), (0.0, 1.0)
    groups = semantic_group(paragraphs, [a, a, a, b, b, b])
    assert len(groups) == 2
    assert groups[0].count(para) == 3 and groups[1].count(para) == 3


def test_semantic_group_respects_min_and_max_size() -> None:
    short = "y" * 100
    a, b = (1.0, 0.0), (0.0, 1.0)
    # The subject changes after one short paragraph, but 100 chars is below MIN_CHARS.
    merged = semantic_group([short, short, short, short], [a, b, b, b])
    assert len(merged) == 1
    long = "z" * 700
    capped = semantic_group([long, long, long], [a, a, a])
    assert all(len(c) <= MAX_CHARS for c in capped) and len(capped) == 3


def test_semantic_group_with_few_paragraphs_is_size_based() -> None:
    paragraphs = ["a" * 500, "b" * 500]
    result = semantic_group(paragraphs, [(1.0, 0.0), (0.0, 1.0)])
    assert result == size_group(paragraphs)
