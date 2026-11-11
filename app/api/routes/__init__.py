"""HTTP endpoints: read the request and the identity headers, hand the work to the services."""

from fastapi import APIRouter

from app.api.routes import runs


router = APIRouter()
router.include_router(runs.router)