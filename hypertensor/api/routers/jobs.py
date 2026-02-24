"""HyperTensor API — Job endpoints.

POST /v1/jobs                  → Submit a job
GET  /v1/jobs/{job_id}         → Get job status
GET  /v1/jobs/{job_id}/result  → Get result payload
GET  /v1/jobs/{job_id}/validation    → Get validation report
GET  /v1/jobs/{job_id}/certificate   → Get trust certificate
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ...core.certificates import issue_certificate
from ...core.evidence import generate_claims, generate_validation_report
from ...core.executor import ExecutionConfig, execute
from ...core.hasher import content_hash
from ...core.registry import DOMAINS, get_domain
from ...core.sanitizer import sanitize_result
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
from .. import __version__ as api_version
from ..auth import require_api_key
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["jobs"])

import hypertensor


# ═══════════════════════════════════════════════════════════════════
# POST /v1/jobs — Submit a job
# ═══════════════════════════════════════════════════════════════════


class _SubmitRequest:
    """Thin extraction of Submit parameters from the raw body."""


from pydantic import BaseModel, Field


class SubmitJobRequest(BaseModel):
    """Job submission request."""

    job_type: JobType = Field(
        ..., description="Type of job to run."
    )
    domain: str | None = Field(
        default=None,
        description="Physics domain (required for execution jobs).",
    )
    n_bits: int = Field(default=8, ge=4, le=14)
    n_steps: int = Field(default=100, ge=1, le=10_000)
    dt: float | None = Field(default=None, gt=0.0)
    max_rank: int = Field(default=64, ge=2, le=128)
    parameters: dict[str, Any] = Field(default_factory=dict)
    return_fields: bool = True
    return_coordinates: bool = True
    # For validation/attestation jobs
    artifact_bundle: dict[str, Any] | None = None


@router.post(
    "/jobs",
    status_code=status.HTTP_201_CREATED,
    summary="Submit a job",
    description=(
        "Submit a physics simulation, benchmark, validation, or attestation "
        "job.  Returns immediately with job status.  The job executes "
        "asynchronously.  Poll GET /v1/jobs/{job_id} for status."
    ),
    responses={
        201: {"description": "Job created"},
        400: {"description": "Invalid request"},
        409: {"description": "Idempotency key collision"},
    },
)
async def submit_job(
    request: SubmitJobRequest,
    api_key: Annotated[str, Depends(require_api_key)],
    x_idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    # ── Idempotency check ───────────────────────────────────────────
    if x_idempotency_key:
        existing = store.get_by_idempotency_key(x_idempotency_key)
        if existing is not None:
            return existing.to_status()

    # ── Validate domain for execution jobs ──────────────────────────
    if request.job_type in (
        JobType.PHYSICS_VM_EXECUTION,
        JobType.RANK_ATLAS_DIAGNOSTIC,
    ):
        if not request.domain:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "E001",
                    "message": "domain is required for execution jobs.",
                    "retryable": False,
                },
            )
        if request.domain not in DOMAINS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "E001",
                    "message": f"Unknown domain: {request.domain!r}.  Available: {list(DOMAINS)}",
                    "retryable": False,
                },
            )

    # ── Validate resolution limits ──────────────────────────────────
    if request.n_bits > settings.max_n_bits:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E003",
                "message": f"n_bits={request.n_bits} exceeds limit {settings.max_n_bits}.",
                "retryable": False,
            },
        )

    # ── Create job ──────────────────────────────────────────────────
    job_input = JobInput(
        job_type=request.job_type,
        domain=request.domain,
        n_bits=request.n_bits,
        n_steps=request.n_steps,
        dt=request.dt,
        max_rank=request.max_rank,
        parameters=request.parameters,
        return_fields=request.return_fields,
        return_coordinates=request.return_coordinates,
        artifact_bundle=request.artifact_bundle,
    )

    input_hash = content_hash(job_input.model_dump())

    job = Job(
        job_type=request.job_type,
        input=job_input,
        input_manifest_hash=input_hash,
        idempotency_key=x_idempotency_key,
        api_key_suffix=api_key[-6:] if len(api_key) > 6 else api_key,
        artifact_hashes={"input": input_hash, "result": None, "validation": None, "certificate": None},
    )
    store.create(job)

    logger.info(
        "Job %s created: type=%s domain=%s (key=…%s)",
        job.job_id, job.job_type.value, request.domain, job.api_key_suffix,
    )

    # ── Execute asynchronously ──────────────────────────────────────
    asyncio.get_running_loop().create_task(
        _execute_job(job.job_id)
    )

    return job.to_status()


# ═══════════════════════════════════════════════════════════════════
# GET /v1/jobs/{job_id} — Job status
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}",
    summary="Get job status",
    description="Returns job metadata and state.  Does not include result payload.",
)
async def get_job(
    job_id: str,
    api_key: Annotated[str, Depends(require_api_key)],
) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "E004", "message": f"Job {job_id} not found.", "retryable": False},
        )
    return job.to_status()


# ═══════════════════════════════════════════════════════════════════
# GET /v1/jobs/{job_id}/result — Result payload
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}/result",
    summary="Get job result",
    description="Returns the full result payload (envelope format).",
)
async def get_result(
    job_id: str,
    api_key: Annotated[str, Depends(require_api_key)],
) -> dict[str, Any]:
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
    return job.to_envelope()


# ═══════════════════════════════════════════════════════════════════
# GET /v1/jobs/{job_id}/validation — Validation report
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}/validation",
    summary="Get validation report",
    description="Returns the structured validation report for a completed job.",
)
async def get_validation(
    job_id: str,
    api_key: Annotated[str, Depends(require_api_key)],
) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "E004", "message": f"Job {job_id} not found.", "retryable": False},
        )
    if job.validation is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "E005", "message": "Job not yet validated.", "retryable": True},
        )
    return job.validation


# ═══════════════════════════════════════════════════════════════════
# GET /v1/jobs/{job_id}/certificate — Trust certificate
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}/certificate",
    summary="Get trust certificate",
    description="Returns the signed trust certificate for an attested job.",
)
async def get_certificate(
    job_id: str,
    api_key: Annotated[str, Depends(require_api_key)],
) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "E004", "message": f"Job {job_id} not found.", "retryable": False},
        )
    if job.certificate is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "E005", "message": "Job not yet attested.", "retryable": True},
        )
    return job.certificate


# ═══════════════════════════════════════════════════════════════════
# Background job execution
# ═══════════════════════════════════════════════════════════════════


async def _execute_job(job_id: str) -> None:
    """Execute a job through the full lifecycle:
    queued → running → succeeded → validated → attested
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
        if job.job_type in (
            JobType.PHYSICS_VM_EXECUTION,
            JobType.FULL_PIPELINE,
        ):
            await _run_physics_execution(job, loop)
        elif job.job_type == JobType.VALIDATION:
            await _run_validation(job)
        elif job.job_type == JobType.ATTESTATION:
            await _run_attestation(job)
        else:
            # Benchmark jobs use the same physics execution path
            await _run_physics_execution(job, loop)

    except asyncio.TimeoutError:
        job.error = JobError(
            code=ErrorCode.TIMEOUT.value,
            message=f"Execution timed out after {settings.job_timeout_s:.0f}s.",
            retryable=True,
        )
        job.transition(JobState.FAILED)
        store.update(job)

    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        job.error = JobError(
            code=ErrorCode.INTERNAL.value,
            message=f"Execution failed: {type(exc).__name__}",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)


async def _run_physics_execution(job: Job, loop: asyncio.AbstractEventLoop) -> None:
    """Execute a physics VM job through the full lifecycle."""
    inp = job.input

    config = ExecutionConfig(
        domain=inp.domain or "",
        n_bits=inp.n_bits,
        n_steps=inp.n_steps,
        dt=inp.dt,
        max_rank=inp.max_rank,
        truncation_tol=settings.truncation_tol,
        parameters=inp.parameters,
    )

    # Execute in thread pool
    raw_result = await asyncio.wait_for(
        loop.run_in_executor(None, partial(execute, config)),
        timeout=settings.job_timeout_s,
    )

    if not raw_result.success:
        job.error = JobError(
            code=ErrorCode.DIVERGED.value,
            message="Simulation diverged or failed.",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return

    # Sanitize (strip TT internals)
    domain_spec = get_domain(inp.domain or "")
    sanitized = sanitize_result(
        execution_result=raw_result,
        domain_key=inp.domain or "",
        precision=settings.field_precision,
        max_field_points=settings.max_field_points,
        include_fields=inp.return_fields,
        include_coordinates=inp.return_coordinates,
    )

    # Build public result
    result_payload = {
        "domain": domain_spec.key,
        "domain_label": domain_spec.label,
        "equation": domain_spec.equation,
        "parameters": config.merged_parameters,
        **sanitized,
    }

    job.result = result_payload
    result_hash = content_hash(result_payload)
    job.artifact_hashes["result"] = result_hash
    job.transition(JobState.SUCCEEDED)
    store.update(job)

    # ── Auto-validate ───────────────────────────────────────────────
    validation = generate_validation_report(sanitized, inp.domain or "")
    job.validation = validation
    validation_hash = content_hash(validation)
    job.artifact_hashes["validation"] = validation_hash
    job.transition(JobState.VALIDATED)
    store.update(job)

    # ── Auto-attest ─────────────────────────────────────────────────
    claims = generate_claims(sanitized, inp.domain or "")
    certificate = issue_certificate(
        job_id=job.job_id,
        claims=claims,
        input_manifest_hash=job.input_manifest_hash,
        result_hash=result_hash,
        config_hash=content_hash(config.merged_parameters),
        runtime_version=hypertensor.RUNTIME_VERSION,
        device_class=settings.device,
    )
    job.certificate = certificate
    cert_hash = content_hash(certificate)
    job.artifact_hashes["certificate"] = cert_hash
    job.transition(JobState.ATTESTED)
    store.update(job)

    logger.info("Job %s attested: %d claims", job.job_id, len(claims))


async def _run_validation(job: Job) -> None:
    """Run a validation-only job against an artifact bundle."""
    bundle = job.input.artifact_bundle
    if not bundle or "result" not in bundle:
        job.error = JobError(
            code=ErrorCode.INVALID_ARTIFACT.value,
            message="artifact_bundle with 'result' key is required for validation jobs.",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return

    domain = bundle.get("domain", job.input.domain or "unknown")
    validation = generate_validation_report(bundle["result"], domain)
    job.result = {"validated_bundle": True}
    job.validation = validation
    job.artifact_hashes["result"] = content_hash(bundle["result"])
    job.artifact_hashes["validation"] = content_hash(validation)
    job.transition(JobState.SUCCEEDED)
    store.update(job)
    job.transition(JobState.VALIDATED)
    store.update(job)


async def _run_attestation(job: Job) -> None:
    """Run an attestation-only job to generate a certificate."""
    bundle = job.input.artifact_bundle
    if not bundle or "result" not in bundle:
        job.error = JobError(
            code=ErrorCode.INVALID_ARTIFACT.value,
            message="artifact_bundle with 'result' key is required for attestation jobs.",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return

    domain = bundle.get("domain", job.input.domain or "unknown")
    result_data = bundle["result"]

    # Validate first
    validation = generate_validation_report(result_data, domain)
    job.validation = validation

    if not validation["valid"]:
        job.error = JobError(
            code=ErrorCode.VALIDATION_FAILED.value,
            message="Cannot attest: validation failed.",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return

    # Generate claims and certificate
    claims = generate_claims(result_data, domain)
    result_hash = content_hash(result_data)
    certificate = issue_certificate(
        job_id=job.job_id,
        claims=claims,
        input_manifest_hash=job.input_manifest_hash,
        result_hash=result_hash,
        config_hash=content_hash(bundle.get("parameters", {})),
        runtime_version=hypertensor.RUNTIME_VERSION,
        device_class=settings.device,
    )

    job.result = {"attested_bundle": True}
    job.certificate = certificate
    job.artifact_hashes["result"] = result_hash
    job.artifact_hashes["validation"] = content_hash(validation)
    job.artifact_hashes["certificate"] = content_hash(certificate)

    job.transition(JobState.SUCCEEDED)
    store.update(job)
    job.transition(JobState.VALIDATED)
    store.update(job)
    job.transition(JobState.ATTESTED)
    store.update(job)
