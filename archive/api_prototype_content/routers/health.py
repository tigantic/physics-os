"""Ontic API — Health endpoint.

``GET /v1/health`` returns server status, GPU info, and uptime.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from ..config import settings
from ..models import HealthResponse

router = APIRouter(prefix="/v1", tags=["health"])

_start_time = time.monotonic()


def _gpu_info() -> tuple[str | None, int | None]:
    """Detect GPU name and memory without leaking driver internals."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
            return name, mem
    except Exception:
        pass
    return None, None


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Server health check",
    description="Returns server status, GPU availability, and uptime.",
)
async def health() -> HealthResponse:
    gpu_name, gpu_mem = _gpu_info()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=settings.device,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        uptime_s=round(time.monotonic() - _start_time, 2),
        domains_available=7,
    )
