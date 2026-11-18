"""Deterministic verifier: every number and date in the answer must trace to a tool result.

Supported = appears in a (non-error) tool result, in the user's question, in a fact of an
earlier verified turn of the same chat session, is the result of a `calculate` call whose
inputs are themselves supported, or is a trivial count of returned articles. Anything else is reported back so the agent can fix or drop it.

Known limits (by design, kept simple): no unit scaling ("1,2 tỷ" vs "1200 triệu"), no
check that a number is attached to the right entity, and no verification of prose claims.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from langchain_core.messages import BaseMessage, ToolMessage

CALCULATE_TOOL = "calculate"
# Plain constants a formula may contain without being sourced (percent, months, thousands).
SAFE_CONSTANTS = frozenset(Decimal(n) for n in (0, 1, 2, 3, 4, 12, 100, 1000))

_URL = re.compile(r"https?://\S+")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*]\s+)?\d+[.)]\s+")
_CALENDAR_WORD = re.compile(r"(?i)\b(?:tháng|quý|q)\s*\d{1,2}\b")
# No trailing \b: ISO timestamps continue with "T08:00:00".
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_VI_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_EN_DATE = re.compile(
    r"\b(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{4})\b"
)
_DAY_MONTH = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_NUMBER = re.compile(r"(?<![\w.,/])\d+(?:[.,]\d+)*(?![\w/])")
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

_DOT_THOUSANDS = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
_COMMA_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_SINGLE_DECIMAL = re.compile(r"^\d+[.,]\d+$")
_INTEGER = re.compile(r"^\d+$")


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    unsupported: list[str] = field(default_factory=list)
    calculation_issues: list[str] = field(default_factory=list)


def number_candidates(token: str) -> set[Decimal]:
    """All plausible values of a numeric token (vi '1.234,5' and en '1,234.5' both occur)."""
    values: list[str] = []
    if _INTEGER.match(token):
        values.append(token)
    if _DOT_THOUSANDS.match(token):
        values.append(token.replace(".", "").replace(",", "."))
    if _COMMA_THOUSANDS.match(token):
        values.append(token.replace(",", ""))
    if _SINGLE_DECIMAL.match(token):
        values.append(token.replace(",", "."))
    result: set[Decimal] = set()
    for value in values:
        try:
            result.add(Decimal(value))
        except InvalidOperation:
            continue
    return result


@dataclass
class _Extracted:
    tokens: list[str]
    dates: set[date]
    day_months: set[tuple[int, int]]


def _extract(text: str) -> _Extracted:
    text = _UUID.sub(" ", _URL.sub(" ", text))
    dates: set[date] = set()
    day_months: set[tuple[int, int]] = set()

    def take(pattern: re.Pattern[str], build) -> None:
        nonlocal text
        for m in pattern.finditer(text):
            try:
                dates.add(build(m))
            except ValueError:
                pass
        text = pattern.sub(" ", text)

    take(_ISO_DATE, lambda m: date(int(m[1]), int(m[2]), int(m[3])))
    take(_VI_DATE, lambda m: date(int(m[3]), int(m[2]), int(m[1])))
    take(
        _EN_DATE,
        lambda m: date(int(m[3]), _MONTHS.index(m[2]) + 1, int(m[1])),
    )
    for m in _DAY_MONTH.finditer(text):
        day_months.add((int(m[1]), int(m[2])))
    text = _DAY_MONTH.sub(" ", text)
    text = _CALENDAR_WORD.sub(" ", _LIST_MARKER.sub("", text))
    return _Extracted(_NUMBER.findall(text), dates, day_months)


def facts_of(text: str) -> dict[str, list[str]]:
    """Numbers and dates `text` states, kept so a later turn may reuse them as a source."""
    found = _extract(text)
    return {
        "numbers": list(dict.fromkeys(found.tokens)),
        "dates": sorted(d.isoformat() for d in found.dates),
        "day_months": sorted(f"{d:02d}/{m:02d}" for d, m in found.day_months),
    }


def _known_values(tokens: Sequence[str]) -> set[Decimal]:
    values: set[Decimal] = set()
    for token in tokens:
        values |= number_candidates(token)
    return values


def _tool_text(message: ToolMessage) -> str:
    return (
        message.content
        if isinstance(message.content, str)
        else json.dumps(message.content)
    )


def _article_count(text: str) -> int | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    articles = data.get("articles") if isinstance(data, dict) else None
    return len(articles) if isinstance(articles, list) else None


def _calculation_inputs(
    text: str,
) -> tuple[list[str], list[str], Decimal | None] | None:
    """(variable values, expression literals, result) of a calculate result, or None."""
    try:
        data = json.loads(text)
        result = Decimal(str(data["result"]))
        # Signs carry direction ("giảm"/"tăng"), so compare magnitudes only.
        variables = [str(v).lstrip("+-") for v in data.get("variables", {}).values()]
        literals = _extract(str(data["expression"])).tokens
    except ValueError, KeyError, TypeError, InvalidOperation, AttributeError:
        return None
    return variables, literals, result


def verify_answer(
    answer: str,
    messages: Sequence[BaseMessage],
    *,
    question: str,
    today: date,
    history_sources: Sequence[str] = (),
) -> VerifyResult:
    sources = [
        m for m in messages if isinstance(m, ToolMessage) and m.status != "error"
    ]
    data_texts = [_tool_text(m) for m in sources if m.name != CALCULATE_TOOL]
    base = [_extract(t) for t in [*data_texts, question, *history_sources]]

    allowed = _known_values([tok for e in base for tok in e.tokens])
    allowed_dates = {today, *(d for e in base for d in e.dates)}
    allowed_day_months = {(d.day, d.month) for d in allowed_dates} | {
        dm for e in base for dm in e.day_months
    }
    allowed |= {Decimal(d.year) for d in allowed_dates}
    allowed |= {Decimal(n) for t in data_texts if (n := _article_count(t)) is not None}

    calculation_issues: list[str] = []
    for message in sources:
        if message.name != CALCULATE_TOOL:
            continue
        parsed = _calculation_inputs(_tool_text(message))
        if parsed is None:
            continue
        variables, literals, result = parsed
        unsourced = [
            token
            for token in [*variables, *literals]
            if not number_candidates(token) & (allowed | SAFE_CONSTANTS)
        ]
        if unsourced:
            calculation_issues.append(
                f"inputs {sorted(set(unsourced))} of a calculation are not in any tool result"
            )
        elif result is not None:
            allowed.add(abs(result))

    found = _extract(answer)
    unsupported: list[str] = []
    for token in found.tokens:
        candidates = number_candidates(token)
        if candidates and not candidates & allowed and token not in unsupported:
            unsupported.append(token)
    for d in sorted(found.dates - allowed_dates):
        unsupported.append(d.isoformat())
    for day, month in sorted(found.day_months - allowed_day_months):
        unsupported.append(f"{day:02d}/{month:02d}")

    return VerifyResult(
        passed=not unsupported and not calculation_issues,
        unsupported=unsupported,
        calculation_issues=calculation_issues,
    )


def feedback_message(result: VerifyResult) -> str:
    """Instruction sent back to the agent when verification fails."""
    lines = ["<verifier_feedback>", "Your last answer failed number verification."]
    if result.unsupported:
        lines.append(
            "These figures/dates are not in any tool result: "
            + ", ".join(result.unsupported)
            + ". Remove them, or obtain them with a tool, or compute them with `calculate`."
        )
    lines.extend(result.calculation_issues)
    lines.append("Write the corrected final answer now.")
    lines.append("</verifier_feedback>")
    return "\n".join(lines)
