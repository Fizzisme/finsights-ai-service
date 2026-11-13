"""HTTP endpoints: read the request and the identity headers, hand the work to the services."""

from fastapi import APIRouter

from app.api.routes import chat, runs

router = APIRouter()
router.include_router(runs.router)
router.include_router(chat.router)
