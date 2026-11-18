from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from app.schemas.news import ResolveSecurityInput, ResolveSecurityResult
from app.tools.sql import resolve_security as mod


def test_ticker_like_name_also_matches_symbol() -> None:
    sql, params = mod._build_query(ResolveSecurityInput(name="hpg"))
    assert "symbol_at_publication = :symbol" in sql
    assert params["symbol"] == "HPG" and params["kw"] == "%hpg%"


def test_phrase_with_space_matches_alias_only() -> None:
    sql, params = mod._build_query(ResolveSecurityInput(name="Hòa Phát"))
    assert ":symbol" not in sql and "symbol" not in params


def test_wildcards_escaped_and_bound() -> None:
    sql, params = mod._build_query(ResolveSecurityInput(name="a%_b"))
    assert params["kw"] == "%a\\%\\_b%" and "a%" not in sql


@pytest.mark.anyio
async def test_groups_aliases_by_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        return [
            {"symbol": "VCB", "alias": "Vietcombank", "n": 9},
            {"symbol": "VCB", "alias": "VCB", "n": 5},  # ticker echo is dropped
            {"symbol": "VCB", "alias": "Ngân hàng TMCP Ngoại thương Việt Nam", "n": 1},
            {"symbol": "TCB", "alias": "Techcombank", "n": 1},
        ]

    monkeypatch.setattr(mod, "fetch_all", fake)
    out = await mod.resolve_security(ResolveSecurityInput(name="ngân hàng"))
    assert [m.symbol for m in out.matches] == ["VCB", "TCB"]
    assert out.matches[0].aliases == [
        "Vietcombank",
        "Ngân hàng TMCP Ngoại thương Việt Nam",
    ]


@pytest.mark.anyio
async def test_no_match_is_empty_with_coverage_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(sql: str, params: dict[str, Any] | None = None) -> list[dict]:
        return []

    monkeypatch.setattr(mod, "fetch_all", fake)
    out = await mod.resolve_security_tool.ainvoke({"name": "Vingroup"})
    assert out["matches"] == [] and "news database" in out["note"]


@pytest.mark.anyio
async def test_db_failure_returns_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(args: ResolveSecurityInput) -> ResolveSecurityResult:
        raise OperationalError("SELECT secret", {}, Exception("host 1.2.3.4 denied"))

    monkeypatch.setattr(mod, "resolve_security", boom)
    out = await mod.resolve_security_tool.ainvoke({"name": "VCB"})
    assert out["matches"] == []
    assert "1.2.3.4" not in str(out) and "secret" not in str(out)
