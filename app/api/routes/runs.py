"""Stateless runs: health, routing check, one run, and a run's trace."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
