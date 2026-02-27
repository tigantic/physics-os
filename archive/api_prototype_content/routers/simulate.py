"""HyperTensor API — Simulation endpoint.

``POST /v1/simulate`` accepts a physics problem specification and
returns computed observables with conservation diagnostics.

All tensor-train internals are stripped before the response is sent.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_api_key
from ..config import settings
from ..models import (
    ErrorResponse,
    SimulationRequest,
    SimulationResponse,
)
from ..services.serializer import serialize_result
from ..services.solver import execute_simulation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["simulate"])


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        504: {"model": ErrorResponse, "description": "Simulation timeout"},
    },
    summary="Run a physics simulation",
    description=(
        "Submit a physics problem specification and receive computed "
        "physical observables.  The server compiles the problem into "
        "a compressed numerical program, executes it on GPU, and "
        "returns field values with conservation diagnostics.\n\n"
        "**No internal solver state is exposed** — only physical "
        "quantities and verification metrics are returned."
    ),
)
async def simulate(
    request: SimulationRequest,
    api_key: Annotated[str, Depends(require_api_key)],
) -> SimulationResponse:
    job_id = str(uuid.uuid4())

    # ── Validate resolution bounds ──────────────────────────────────
    if request.resolution.n_bits > settings.max_n_bits:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"n_bits={request.resolution.n_bits} exceeds server maximum "
                f"of {settings.max_n_bits}."
            ),
        )
    if request.resolution.n_steps > settings.max_n_steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"n_steps={request.resolution.n_steps} exceeds server maximum "
                f"of {settings.max_n_steps}."
            ),
        )
    if request.resolution.max_rank > settings.max_rank:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"max_rank={request.resolution.max_rank} exceeds server maximum "
                f"of {settings.max_rank}."
            ),
        )

    logger.info(
        "Job %s: domain=%s n_bits=%d n_steps=%d (key=…%s)",
        job_id,
        request.domain.value,
        request.resolution.n_bits,
        request.resolution.n_steps,
        api_key[-6:] if len(api_key) > 6 else api_key,
    )

    # ── Execute in thread pool (non-blocking) ───────────────────────
    loop = asyncio.get_running_loop()
    try:
        result, domain_str, domain_label, params_echo = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(
                    execute_simulation,
                    domain=request.domain,
                    resolution=request.resolution,
                    parameters=request.parameters,
                ),
            ),
            timeout=settings.job_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error("Job %s timed out after %.0f s", job_id, settings.job_timeout_s)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Simulation timed out after {settings.job_timeout_s:.0f} seconds.",
        )
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {type(exc).__name__}: {exc}",
        )

    # ── Serialize (strip TT internals) ──────────────────────────────
    if not result.success:
        logger.warning("Job %s completed with error: %s", job_id, result.error)

    response = serialize_result(
        job_id=job_id,
        result=result,
        domain=domain_str,
        domain_label=domain_label,
        params_echo=params_echo,
        return_fields=request.return_fields,
        return_coordinates=request.return_coordinates,
    )

    logger.info(
        "Job %s completed: wall=%.2fs conservation=%s",
        job_id,
        response.performance.wall_time_s,
        response.conservation.status if response.conservation else "n/a",
    )

    return response
