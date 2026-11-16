"""Deterministic arithmetic for the agent: never eval(), exact Decimal maths."""

import ast
import operator
from collections.abc import Callable
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 10
MAX_ABS_RESULT = Decimal("1e18")

_BINARY: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


class CalculationError(ValueError):
    pass


class CalculateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_LENGTH,
        description=(
            "Arithmetic expression using + - * / ** and parentheses, e.g. "
            "'(new - old) / old * 100'. Put every figure that comes from a tool result "
            "in `variables` and refer to it by name; only plain constants such as 100 "
            "belong inside the expression."
        ),
    )
    variables: dict[
        Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,30}$")], Decimal
    ] = Field(
        default_factory=dict,
        description=(
            "Name -> number, copied exactly from tool results (for example "
            "{'old': 25000, 'new': 27500}). Do not invent or estimate values."
        ),
    )
    decimals: int = Field(
        default=2, ge=0, le=8, description="Decimal places to round the result to."
    )


def _eval(node: ast.AST, variables: dict[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("only numbers are allowed")
        return Decimal(str(node.value))
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise CalculationError(f"unknown variable {node.id!r}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand, variables)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _eval(node.left, variables)
        right = _eval(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculationError(f"exponent must be at most {MAX_EXPONENT}")
        result = _BINARY[type(node.op)](left, right)
        if abs(result) > MAX_ABS_RESULT:
            raise CalculationError("result is too large")
        return result
    raise CalculationError("unsupported syntax: use numbers, names, + - * / ** and ()")


def evaluate(
    expression: str, variables: dict[str, Decimal], decimals: int = 2
) -> Decimal:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise CalculationError("invalid expression") from None
    try:
        with localcontext() as ctx:
            ctx.prec = 40
            result = _eval(tree, variables)
            return result.quantize(Decimal(1).scaleb(-decimals))
    except DivisionByZero, InvalidOperation:
        raise CalculationError("division by zero or invalid operation") from None


@tool("calculate", args_schema=CalculateInput)
def calculate_tool(
    expression: str,
    variables: dict[str, Decimal] | None = None,
    decimals: int = 2,
) -> dict[str, Any]:
    """Compute an arithmetic expression exactly (percent change, ratio, sum, average).

    Always use this instead of doing arithmetic yourself, and state the returned
    `result` unchanged in your answer. Inputs must be numbers taken from tool results.
    """
    variables = variables or {}
    try:
        result = evaluate(expression, variables, decimals)
    except CalculationError as exc:
        return {"error": str(exc)}
    return {
        "expression": expression,
        "variables": {name: str(value) for name, value in variables.items()},
        "result": str(result),
    }
