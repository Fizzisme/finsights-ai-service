"""The caller's token quota: how much of each window is used and when it resets."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes.sessions import UNAVAILABLE, User, quota_limits
from app.schemas.usage import QuotaWindow, UsageResponse
from app.storage.mysql import usage_store

router = APIRouter(prefix="/usage", tags=["usage"])
logger = logging.getLogger(__name__)


@router.get("", response_model=UsageResponse)
async def get_usage(user: User) -> Any:
    try:
        row = await usage_store.get_usage(user)
    except SQLAlchemyError, RuntimeError:
        logger.exception("loading token usage failed")
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from None
    windows = usage_store.windows(row, usage_store.utc_now(), quota_limits())
    return UsageResponse(
        **{
            name: QuotaWindow(used=w.used, limit=w.limit or None, resets_at=w.resets_at)
            for name, w in windows.items()
        }
    )
