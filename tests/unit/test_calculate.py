from decimal import Decimal

import pytest

from app.tools.calculator.calculate import CalculationError, calculate_tool, evaluate


def test_percent_change_is_exact() -> None:
    value = evaluate(
        "(new - old) / old * 100", {"old": Decimal(25000), "new": Decimal(27500)}
    )
    assert value == Decimal("10.00")


def test_decimal_arithmetic_has_no_float_error() -> None:
    assert evaluate("0.1 + 0.2", {}, decimals=1) == Decimal("0.3")


def test_rounding_uses_requested_places() -> None:
    assert evaluate("10 / 3", {}, decimals=4) == Decimal("3.3333")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('x')",
        "open('f')",
        "a.b",
        "[1, 2]",
        "'text'",
        "True + 1",
        "2 ** 11",
        "1 / 0",
        "unknown + 1",
        "1 +",
        "10 ** 10 * 10 ** 10",
    ],
)
def test_unsafe_or_invalid_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(CalculationError):
        evaluate(expression, {})


@pytest.mark.anyio
async def test_tool_returns_json_with_result_and_inputs() -> None:
    out = await calculate_tool.ainvoke(
        {"expression": "a / b", "variables": {"a": 5, "b": 4}, "decimals": 2}
    )
    assert out == {
        "expression": "a / b",
        "variables": {"a": "5", "b": "4"},
        "result": "1.25",
    }


@pytest.mark.anyio
async def test_tool_reports_errors_instead_of_raising() -> None:
    out = await calculate_tool.ainvoke({"expression": "1 / 0"})
    assert "error" in out


def test_schema_describes_every_field() -> None:
    assert all("description" in p for p in calculate_tool.args.values())
