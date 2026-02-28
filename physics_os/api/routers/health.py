"""Ontic API — Health endpoint.

GET /v1/health — Liveness & readiness probe.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

import physics_os

_BOOT_TIME = time.time()

router = APIRouter(prefix="/v1", tags=["health"])


@router.get(
    "/health",
    summary="Health check",
    description="Liveness probe. Returns runtime versions and uptime.",
)
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _BOOT_TIME, 2),
        "versions": {
            "api": physics_os.API_VERSION,
            "runtime": physics_os.RUNTIME_VERSION,
            "schema": physics_os.SCHEMA_VERSION,
        },
    }
