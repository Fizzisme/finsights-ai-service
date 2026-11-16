import json
from datetime import date
from decimal import Decimal

import pytest
from langchain_core.messages import ToolMessage

from app.graph.nodes.verifier import feedback_message, number_candidates, verify_answer

TODAY = date(2026, 11, 15)


def news(
    *titles: str, uid: str = "3f2a1b9c-1111-2222-3333-444455556666"
) -> ToolMessage:
    articles = [
        {
            "article_uid": uid,
            "title": t,
            "published_at": "2026-05-01T08:00:00",
            "url": "https://example.com/2026/05/123456",
        }
        for t in titles
    ]
    return ToolMessage(
        json.dumps({"articles": articles}), tool_call_id="n1", name="search_news"
    )


def calc(variables: dict[str, str], expression: str, result: str) -> ToolMessage:
    return ToolMessage(
        json.dumps(
            {"expression": expression, "variables": variables, "result": result}
        ),
        tool_call_id="c1",
        name="calculate",
    )


def check(answer: str, *messages: ToolMessage, question: str = "q"):
    return verify_answer(answer, messages, question=question, today=TODAY)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("12.990.000", {12990000}),
        ("1,234,567", {1234567}),
        ("1,5", {1.5}),
        ("1.5", {1.5}),
        ("1.500", {1500, 1.5}),
        ("27500", {27500}),
    ],
)
def test_number_candidates_cover_vi_and_en_formats(token, expected) -> None:
    assert number_candidates(token) == {Decimal(str(e)) for e in expected}


def test_figure_from_tool_result_passes_in_either_format() -> None:
    msg = news("VCB báo lãi 12.990 tỷ đồng, tăng 5,5%")
    assert check("VCB lãi 12.990 tỷ, tăng 5,5%.", msg).passed
    assert check("VCB lãi 12,990 tỷ, tăng 5.5%.", msg).passed


def test_invented_figure_is_reported() -> None:
    result = check("VCB lãi 15.000 tỷ.", news("VCB báo lãi 12.990 tỷ đồng"))
    assert not result.passed
    assert result.unsupported == ["15.000"]


def test_dates_are_compared_as_dates_across_formats() -> None:
    msg = news("Tin VCB")
    assert check("Ngày 01/05/2026 · CafeF", msg).passed
    assert check("1 May 2026", msg).passed
    wrong = check("Ngày 02/05/2026", msg)
    assert wrong.unsupported == ["2026-05-02"]


def test_urls_uuids_list_markers_and_tickers_are_not_claims() -> None:
    msg = news("VN30 giảm")
    answer = (
        "1. VN30 giảm "
        "([link](https://example.com/2026/05/123456)) 3f2a1b9c-1111-2222-3333-444455556666\n"
        "2. Tháng 5, quý 2 và Q3 không phải số liệu."
    )
    assert check(answer, msg).passed


def test_numbers_from_the_question_and_today_are_allowed() -> None:
    assert check(
        "Trong 10 ngày qua, đến 15/11/2026.", news("x"), question="10 ngày"
    ).passed


def test_article_count_is_allowed() -> None:
    assert check("Có 2 tin liên quan.", news("a", "b")).passed
    assert not check("Có 7 tin liên quan.", news("a", "b")).passed


def test_calculation_result_is_allowed_when_inputs_are_sourced() -> None:
    data = news("Giá từ 25000 lên 27500")
    ok = calc({"old": "25000", "new": "27500"}, "(new - old) / old * 100", "10.00")
    assert check("Giá tăng 10%.", data, ok).passed


def test_calculation_with_invented_inputs_fails_even_if_result_is_quoted() -> None:
    data = news("Giá từ 25000 lên 27500")
    bad = calc({"old": "20000", "new": "27500"}, "(new - old) / old * 100", "37.50")
    result = check("Giá tăng 37,5%.", data, bad)
    assert not result.passed
    assert result.calculation_issues


def test_calculation_does_not_launder_its_own_inputs() -> None:
    # Inputs echoed by the calculate tool itself are not a source.
    bad = calc({"x": "999"}, "x * 2", "1998")
    assert not check("Gấp đôi là 1998.", news("a"), bad).passed


def test_error_tool_results_are_not_sources() -> None:
    err = ToolMessage("giá 777", tool_call_id="e", name="search_news", status="error")
    assert not check("giá 777", err).passed


def test_feedback_message_lists_problems() -> None:
    text = feedback_message(check("lãi 15.000 tỷ", news("lãi 12.990 tỷ")))
    assert text.startswith("<verifier_feedback>") and "15.000" in text


def test_negative_calculation_result_matches_unsigned_figure_in_answer() -> None:
    data = news("Cổ phiếu giảm từ 50000 xuống 49550")
    result = calc({"old": "50000", "new": "49550"}, "(new - old) / old * 100", "-0.90")
    assert check("Giảm 0,9%.", data, result).passed
    assert check("Giảm -0.90%.", data, result).passed


def test_negative_calculation_inputs_are_checked_by_magnitude() -> None:
    data = news("Lỗ 250 tỷ")
    ok = calc({"loss": "-250"}, "loss * 2", "-500")
    assert check("Gấp đôi là 500.", data, ok).passed
    bad = calc({"loss": "-999"}, "loss * 2", "-1998")
    assert not check("Gấp đôi là 1998.", data, bad).passed
