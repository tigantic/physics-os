"""FlowBox API — FastAPI router.

Endpoints:
    POST /v1/flowbox/run                   → Submit a FlowBox job
    GET  /v1/flowbox/presets               → List presets + tiers
    GET  /v1/flowbox/jobs/{job_id}/render   → Download MP4/GIF

FlowBox jobs use the same job store as general jobs, so status and
result retrieval work via the existing ``/v1/jobs/*`` endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from ...core.certificates import issue_certificate
from ...core.evidence import generate_claims, generate_validation_report
from ...core.hasher import content_hash
from ...core.physics_qoi import extract_physics_qoi
from ...jobs.models import (
    ErrorCode,
    InvalidTransition,
    Job,
    JobError,
    JobInput,
    JobState,
    JobType,
)
from ...jobs.store import store
from ..auth import require_api_key
from ..config import settings
from ...flowbox import __version__ as flowbox_version, PRODUCT_KEY
from ...flowbox.contract import (
    FlowBoxConfig,
    FlowBoxRequest,
    FlowBoxTier,
    list_presets,
    list_tiers,
    resolve,
)
from ...flowbox.executor import FlowBoxResult, run_flowbox
from ...flowbox.render import generate_render
from ...flowbox.sanitizer import sanitize_flowbox

import physics_os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/flowbox", tags=["flowbox"])

# In-memory render store (job_id → video bytes).
# In production, this maps to object storage.
_render_store: dict[str, bytes] = {}


# ═══════════════════════════════════════════════════════════════════
# GET /v1/flowbox/presets — List presets + tier caps
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/presets",
    summary="List FlowBox presets and tier caps",
    description=(
        "Returns available IC presets and per-tier parameter limits.  "
        "No authentication required."
    ),
)
async def get_presets() -> dict[str, Any]:
    return {
        "product": PRODUCT_KEY,
        "product_version": flowbox_version,
        "presets": list_presets(),
        "tiers": list_tiers(),
        "poisson_profiles": ["fast", "balanced", "accurate"],
        "grids": [512, 1024],
    }


# ═══════════════════════════════════════════════════════════════════
# POST /v1/flowbox/run — Submit a FlowBox job
# ═══════════════════════════════════════════════════════════════════


@router.post(
    "/run",
    status_code=status.HTTP_201_CREATED,
    summary="Submit a FlowBox simulation",
    description=(
        "Submit a Navier-Stokes 2D simulation with a preset IC.  "
        "Returns immediately with job status.  The simulation executes "
        "asynchronously.  Poll GET /v1/jobs/{job_id} for status."
    ),
    responses={
        201: {"description": "Job created"},
        400: {"description": "Invalid request or tier limit exceeded"},
        409: {"description": "Idempotency key collision"},
    },
)
async def submit_run(
    request: FlowBoxRequest,
    api_key: Annotated[str, Depends(require_api_key)],
    x_idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    # ── Idempotency check ───────────────────────────────────────
    if x_idempotency_key:
        existing = store.get_by_idempotency_key(x_idempotency_key)
        if existing is not None:
            return existing.to_status()

    # ── Resolve tier (alpha: default to explorer) ───────────────
    tier = _resolve_tier(api_key)

    # ── Resolve and validate config ─────────────────────────────
    try:
        config = resolve(request, tier=tier)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E002",
                "message": str(exc),
                "retryable": False,
            },
        )

    # ── Create job ──────────────────────────────────────────────
    job_input = JobInput(
        job_type=JobType.FULL_PIPELINE,
        domain="navier_stokes_2d",
        n_bits=config.n_bits,
        n_steps=config.steps,
        dt=config.dt,
        max_rank=64,
        parameters={
            "_product": PRODUCT_KEY,
            "_flowbox": {
                "preset": config.preset,
                "viscosity": config.viscosity,
                "seed": config.seed,
                "poisson_profile": config.poisson.precond,
                "render": config.render,
                "render_colormap": config.render_colormap,
                "render_fps": config.render_fps,
                "render_watermark": config.render_watermark,
                "output_cadence": config.output_cadence,
                "output_fields": list(config.output_fields),
                "tier": config.tier.value,
            },
            "viscosity": config.viscosity,
            "ic_type": config.preset_spec.ic_type if config.preset_spec.ic_type != "custom" else "taylor_green",
            "poisson_precond": config.poisson.precond,
            "poisson_tol": config.poisson.tol,
            "poisson_max_iters": config.poisson.max_iters,
        },
        return_fields=True,
        return_coordinates=True,
    )

    input_hash = content_hash(job_input.model_dump())

    job = Job(
        job_type=JobType.FULL_PIPELINE,
        input=job_input,
        input_manifest_hash=input_hash,
        idempotency_key=x_idempotency_key,
        api_key_suffix=api_key[-6:] if len(api_key) > 6 else api_key,
        artifact_hashes={
            "input": input_hash,
            "result": None,
            "validation": None,
            "certificate": None,
        },
    )
    store.create(job)

    logger.info(
        "FlowBox job %s created: preset=%s grid=%d steps=%d (key=…%s)",
        job.job_id, config.preset, config.grid, config.steps,
        job.api_key_suffix,
    )

    # ── Execute asynchronously ──────────────────────────────────
    asyncio.get_running_loop().create_task(
        _execute_flowbox_job(job.job_id, config)
    )

    resp = job.to_status()
    resp["product"] = PRODUCT_KEY
    resp["product_version"] = flowbox_version
    resp["preset"] = config.preset
    resp["grid"] = config.grid
    return resp


# ═══════════════════════════════════════════════════════════════════
# GET /v1/flowbox/jobs/{job_id}/render — Download render
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}/render",
    summary="Download FlowBox render",
    description=(
        "Returns the MP4 (or GIF) render for a completed FlowBox job.  "
        "The render is only available after the job reaches 'attested' state."
    ),
    responses={
        200: {"content": {"video/mp4": {}, "image/gif": {}}},
        404: {"description": "Job or render not found"},
        409: {"description": "Job not yet completed"},
    },
)
async def get_render(
    job_id: str,
    api_key: Annotated[str, Depends(require_api_key)],
) -> Response:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "E004", "message": f"Job {job_id} not found.", "retryable": False},
        )

    if job.state in (JobState.QUEUED, JobState.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "E005", "message": f"Job is still {job.state.value}.", "retryable": True},
        )

    video_bytes = _render_store.get(job_id)
    if video_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "E004", "message": "Render not available for this job.", "retryable": False},
        )

    # Determine format from job result
    render_meta = {}
    if job.result and isinstance(job.result, dict):
        render_meta = job.result.get("render", {})
    fmt = render_meta.get("format", "mp4")
    media_type = "video/mp4" if fmt == "mp4" else "image/gif"
    ext = fmt

    return Response(
        content=video_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="flowbox_{job_id[:8]}.{ext}"',
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Background execution pipeline
# ═══════════════════════════════════════════════════════════════════


async def _execute_flowbox_job(job_id: str, config: FlowBoxConfig) -> None:
    """Execute a FlowBox job through the full pipeline.

    queued → running → succeeded → validated → attested

    Steps:
    1. QTT simulation via standard VM execute()
    2. FlowBox sanitization (tighter whitelist)
    3. Dense spectral render (MP4 generation)
    4. Validation
    5. Metering
    6. Attestation (Ed25519 certificate)
    """
    job = store.get(job_id)
    if job is None:
        return

    try:
        job.transition(JobState.RUNNING)
        store.update(job)
    except InvalidTransition:
        return

    loop = asyncio.get_running_loop()

    try:
        # ── 1. Execute in thread pool ───────────────────────────
        fb_result: FlowBoxResult = await asyncio.wait_for(
            loop.run_in_executor(None, partial(run_flowbox, config)),
            timeout=settings.job_timeout_s,
        )

        if not fb_result.raw_result.success:
            job.error = JobError(
                code=ErrorCode.DIVERGED.value,
                message="Simulation diverged or failed.",
                retryable=False,
            )
            job.transition(JobState.FAILED)
            store.update(job)
            return

        # ── 2. Sanitize with FlowBox whitelist ──────────────────
        render_meta: dict[str, Any] | None = None

        # ── 3. Generate render (if requested) ───────────────────
        if config.render and fb_result.render_frames:
            render_result = await loop.run_in_executor(
                None,
                partial(
                    generate_render,
                    fb_result.render_frames,
                    fb_result.render_frame_times,
                    fps=config.render_fps,
                    colormap=config.render_colormap,
                    watermark=config.render_watermark,
                    preset_label=config.preset_spec.label,
                    grid=config.grid,
                    viscosity=config.viscosity,
                ),
            )
            render_meta = render_result.metadata
            if render_result.available:
                _render_store[job_id] = render_result.data
                logger.info(
                    "FlowBox render stored: job=%s format=%s size=%d",
                    job_id, render_result.format, render_result.size_bytes,
                )

        sanitized = sanitize_flowbox(
            raw_result=fb_result.raw_result,
            config=config,
            physics_qoi=fb_result.physics_qoi,
            render_metadata=render_meta,
        )

        # ── 4. Build product result ─────────────────────────────
        from ...core.registry import get_domain
        domain_spec = get_domain("navier_stokes_2d")

        result_payload: dict[str, Any] = {
            "product": PRODUCT_KEY,
            "product_version": flowbox_version,
            "preset": config.preset,
            "preset_label": config.preset_spec.label,
            "domain": domain_spec.key,
            "domain_label": domain_spec.label,
            "equation": domain_spec.equation,
            "parameters": {
                "viscosity": config.viscosity,
                "dt": config.dt,
                "steps": config.steps,
                "seed": config.seed,
                "poisson_profile": config.poisson.precond,
            },
            **sanitized,
        }

        job.result = result_payload
        result_hash = content_hash(result_payload)
        job.artifact_hashes["result"] = result_hash

        # ── 5. Metering ─────────────────────────────────────────
        from physics_os.billing.meter import ledger as _billing_ledger
        from physics_os.billing.stripe_billing import get_billing

        wall_time = sanitized.get("performance", {}).get("wall_time_s", 0.0)
        meter_record = _billing_ledger.record(
            job_id=job.job_id,
            api_key_suffix=job.api_key_suffix,
            domain="navier_stokes_2d",
            device_class=settings.device,
            wall_time_s=wall_time,
        )

        try:
            stripe_billing = get_billing()
            stripe_billing.report_usage(
                api_key=job.api_key_suffix,
                compute_units=meter_record.compute_units if meter_record else 0.0,
                job_id=job.job_id,
            )
        except Exception:
            logger.warning(
                "Stripe usage reporting failed for FlowBox job %s (non-fatal)",
                job.job_id,
            )

        job.transition(JobState.SUCCEEDED)
        store.update(job)

        # ── 6. Validate ─────────────────────────────────────────
        validation = generate_validation_report(sanitized, "navier_stokes_2d")
        job.validation = validation
        validation_hash = content_hash(validation)
        job.artifact_hashes["validation"] = validation_hash
        job.transition(JobState.VALIDATED)
        store.update(job)

        # ── 7. Attest ───────────────────────────────────────────
        claims = generate_claims(sanitized, "navier_stokes_2d")
        certificate = issue_certificate(
            job_id=job.job_id,
            claims=claims,
            input_manifest_hash=job.input_manifest_hash,
            result_hash=result_hash,
            config_hash=content_hash(result_payload.get("parameters", {})),
            runtime_version=physics_os.RUNTIME_VERSION,
            device_class=settings.device,
        )
        job.certificate = certificate
        cert_hash = content_hash(certificate)
        job.artifact_hashes["certificate"] = cert_hash
        job.transition(JobState.ATTESTED)
        store.update(job)

        logger.info(
            "FlowBox job %s attested: preset=%s %d claims",
            job.job_id, config.preset, len(claims),
        )

    except asyncio.TimeoutError:
        job.error = JobError(
            code=ErrorCode.TIMEOUT.value,
            message=f"FlowBox execution timed out after {settings.job_timeout_s:.0f}s.",
            retryable=True,
        )
        job.transition(JobState.FAILED)
        store.update(job)

    except Exception as exc:
        logger.exception("FlowBox job %s failed: %s", job_id, exc)
        job.error = JobError(
            code=ErrorCode.INTERNAL.value,
            message=f"FlowBox execution failed: {type(exc).__name__}",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)


# ═══════════════════════════════════════════════════════════════════
# Tier resolution
# ═══════════════════════════════════════════════════════════════════


def _resolve_tier(api_key: str) -> FlowBoxTier:
    """Resolve the billing tier for an API key.

    Alpha: all users are Explorer.
    Production: query billing module for Stripe subscription tier.
    """
    try:
        from physics_os.billing.stripe_billing import get_billing, Tier

        billing = get_billing()
        # Look up customer by API key suffix
        customer = billing.get_customer(api_key[-6:] if len(api_key) > 6 else api_key)
        if customer and customer.tier:
            tier_map = {
                Tier.EXPLORER: FlowBoxTier.EXPLORER,
                Tier.BUILDER: FlowBoxTier.BUILDER,
                Tier.PROFESSIONAL: FlowBoxTier.PROFESSIONAL,
            }
            return tier_map.get(customer.tier, FlowBoxTier.EXPLORER)
    except Exception:
        pass

    return FlowBoxTier.EXPLORER
