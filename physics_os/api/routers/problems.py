"""Ontic API — Problem Template endpoints.

POST /v1/problems              → Compile & submit a high-level physics problem
GET  /v1/templates             → List available problem templates
GET  /v1/templates/{class_key} → Get template details
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from ...core.certificates import issue_certificate
from ...core.evidence import generate_claims, generate_validation_report
from ...core.executor import ExecutionConfig, execute
from ...core.hasher import content_hash
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
from ...templates.compiler import compile_problem
from ...templates.models import (
    BoundarySpec,
    FlowConditions,
    GeometrySpec,
    GeometryType,
    ProblemClass,
    ProblemResult,
    ProblemSpec,
)
from ...templates.registry import TemplateRegistry
from ..auth import require_api_key
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["templates"])

import physics_os


# ═══════════════════════════════════════════════════════════════════
# Request / response models
# ═══════════════════════════════════════════════════════════════════


class ProblemRequest(BaseModel):
    """High-level problem submission — no raw PDE parameters needed.

    The user specifies what they want to simulate (shape, flow, material)
    and the Problem Compiler resolves everything else.
    """

    problem_class: str = Field(
        ...,
        description=(
            "Physics problem type.  One of: external_flow, internal_flow, "
            "heat_transfer, wave_propagation, natural_convection, "
            "boundary_layer, vortex_dynamics, channel_flow."
        ),
    )
    geometry: dict[str, Any] = Field(
        ...,
        description=(
            'Geometry specification.  Must contain a "shape" key (e.g. '
            '"circle", "naca4", "rectangle") and a "params" dict with '
            "shape-specific parameters."
        ),
    )
    flow: dict[str, Any] = Field(
        ...,
        description=(
            'Flow conditions.  Must contain "velocity" (m/s).  Optional: '
            '"fluid" (default "air"), "temperature" (K), "pressure" (Pa).'
        ),
    )
    boundaries: dict[str, str] | None = Field(
        default=None,
        description="Boundary conditions (keys: inlet, outlet, walls, top, bottom).",
    )
    quality: str = Field(
        default="standard",
        description="Quality tier: quick, standard, high, or maximum.",
    )
    t_end: float | None = Field(
        default=None, gt=0.0,
        description="Simulation end time in seconds (auto-estimated if omitted).",
    )
    domain_multiplier: float = Field(
        default=10.0, gt=1.0,
        description="Domain size as multiple of characteristic length.",
    )
    max_rank: int = Field(
        default=64, ge=2, le=128,
        description="Maximum tensor-train rank.",
    )
    wait: bool = Field(
        default=True,
        description="If True, block until simulation completes.",
    )


class TemplateResponse(BaseModel):
    """Template discovery response."""

    problem_class: str
    label: str
    description: str
    supported_geometries: list[str]
    default_geometry: str
    required_flow_fields: list[str]
    optional_flow_fields: list[str]
    example_params: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# GET /v1/templates — List all templates
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/templates",
    summary="List problem templates",
    description=(
        "Returns all available problem templates with supported geometries, "
        "required/optional fields, and example parameters.  "
        "No authentication required."
    ),
)
async def list_templates() -> dict[str, Any]:
    registry = TemplateRegistry()
    templates: list[dict[str, Any]] = []
    for pc in ProblemClass:
        info = registry.get(pc)
        if info is None:
            continue
        templates.append({
            "problem_class": info.problem_class.value,
            "label": info.label,
            "description": info.description,
            "supported_geometries": [g.value for g in info.supported_geometries],
            "default_geometry": info.default_geometry.value,
            "required_flow_fields": info.required_flow_fields,
            "optional_flow_fields": info.optional_flow_fields,
            "example_params": info.example_params,
        })
    return {
        "template_count": len(templates),
        "templates": templates,
    }


# ═══════════════════════════════════════════════════════════════════
# GET /v1/templates/{class_key} — Template detail
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/templates/{class_key}",
    summary="Get template details",
    description="Returns details for a specific problem template.",
)
async def get_template(class_key: str) -> dict[str, Any]:
    try:
        pc = ProblemClass(class_key)
    except ValueError:
        available = [p.value for p in ProblemClass]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "E001",
                "message": f"Unknown template: {class_key!r}.  Available: {available}",
                "retryable": False,
            },
        )
    registry = TemplateRegistry()
    info = registry.get(pc)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "E001", "message": f"Template {class_key!r} not registered."},
        )
    return {
        "problem_class": info.problem_class.value,
        "label": info.label,
        "description": info.description,
        "supported_geometries": [g.value for g in info.supported_geometries],
        "default_geometry": info.default_geometry.value,
        "required_flow_fields": info.required_flow_fields,
        "optional_flow_fields": info.optional_flow_fields,
        "example_params": info.example_params,
    }


# ═══════════════════════════════════════════════════════════════════
# POST /v1/problems — Compile & submit
# ═══════════════════════════════════════════════════════════════════


@router.post(
    "/problems",
    status_code=status.HTTP_201_CREATED,
    summary="Submit a physics problem",
    description=(
        "Submit a high-level physics problem specification.  The Problem "
        "Compiler resolves the geometry, fluid properties, dimensionless "
        "numbers, and optimal resolution — then runs the simulation.  "
        "No raw PDE parameters needed."
    ),
    responses={
        201: {"description": "Problem compiled and job created"},
        400: {"description": "Invalid problem specification"},
    },
)
async def submit_problem(
    request: ProblemRequest,
    api_key: Annotated[str, Depends(require_api_key)],
    x_idempotency_key: str | None = Header(default=None),
) -> dict[str, Any]:
    # ── Idempotency ─────────────────────────────────────────────────
    if x_idempotency_key:
        existing = store.get_by_idempotency_key(x_idempotency_key)
        if existing is not None:
            return existing.to_status()

    # ── Parse the problem spec ──────────────────────────────────────
    try:
        pc = ProblemClass(request.problem_class)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E001",
                "message": (
                    f"Unknown problem_class: {request.problem_class!r}.  "
                    f"Available: {[p.value for p in ProblemClass]}"
                ),
                "retryable": False,
            },
        )

    # Build geometry spec
    geo_shape = request.geometry.get("shape")
    if not geo_shape:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E001",
                "message": "geometry.shape is required.",
                "retryable": False,
            },
        )

    try:
        geo_type = GeometryType(geo_shape)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E001",
                "message": (
                    f"Unknown geometry shape: {geo_shape!r}.  "
                    f"Available: {[g.value for g in GeometryType]}"
                ),
                "retryable": False,
            },
        )

    geometry = GeometrySpec(
        shape=geo_type,
        params=request.geometry.get("params", {}),
    )

    flow = FlowConditions(
        velocity=request.flow.get("velocity", 1.0),
        fluid=request.flow.get("fluid", "air"),
        temperature=request.flow.get("temperature"),
        pressure=request.flow.get("pressure"),
    )

    boundaries = None
    if request.boundaries:
        boundaries = BoundarySpec(**request.boundaries)

    spec = ProblemSpec(
        problem_class=pc,
        geometry=geometry,
        flow=flow,
        boundaries=boundaries,
        quality=request.quality,
        t_end=request.t_end,
        domain_multiplier=request.domain_multiplier,
        max_rank=request.max_rank,
    )

    # ── Compile ─────────────────────────────────────────────────────
    try:
        compiled: ProblemResult = compile_problem(spec)
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E001",
                "message": f"Compilation error: {exc}",
                "retryable": False,
            },
        )

    # ── Validate resolution limits ──────────────────────────────────
    if compiled.n_bits > settings.max_n_bits:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E003",
                "message": (
                    f"Compiled n_bits={compiled.n_bits} exceeds server "
                    f"limit {settings.max_n_bits}."
                ),
                "retryable": False,
            },
        )

    # ── Create job from compiled result ─────────────────────────────
    job_input = JobInput(
        job_type=JobType.FULL_PIPELINE,
        domain=compiled.domain,
        n_bits=compiled.n_bits,
        n_steps=compiled.n_steps,
        dt=compiled.dt,
        max_rank=compiled.max_rank,
        parameters=compiled.parameters,
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
        "Problem compiled → Job %s: domain=%s n_bits=%d n_steps=%d Re=%.1f",
        job.job_id,
        compiled.domain,
        compiled.n_bits,
        compiled.n_steps,
        compiled.reynolds_number,
    )

    # ── Execute ─────────────────────────────────────────────────────
    asyncio.get_running_loop().create_task(
        _execute_problem_job(job.job_id)
    )

    response: dict[str, Any] = {
        **job.to_status(),
        "compilation": {
            "domain": compiled.domain,
            "n_bits": compiled.n_bits,
            "n_steps": compiled.n_steps,
            "dt": compiled.dt,
            "max_rank": compiled.max_rank,
            "reynolds_number": compiled.reynolds_number,
            "mach_number": compiled.mach_number,
            "characteristic_length": compiled.characteristic_length,
            "fluid_name": compiled.fluid_name,
            "geometry_type": compiled.geometry_type,
            "quality_tier": compiled.quality_tier,
            "warnings": compiled.warnings,
        },
    }
    return response


# ═══════════════════════════════════════════════════════════════════
# Background execution (reuses jobs infrastructure)
# ═══════════════════════════════════════════════════════════════════


async def _execute_problem_job(job_id: str) -> None:
    """Execute a problem-compiled job through full lifecycle."""
    job = store.get(job_id)
    if job is None:
        return

    try:
        job.transition(JobState.RUNNING)
        store.update(job)
    except InvalidTransition:
        return

    loop = asyncio.get_running_loop()
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

    try:
        raw_result = await asyncio.wait_for(
            loop.run_in_executor(None, partial(execute, config)),
            timeout=settings.job_timeout_s,
        )
    except asyncio.TimeoutError:
        job.error = JobError(
            code=ErrorCode.TIMEOUT.value,
            message=f"Execution timed out after {settings.job_timeout_s:.0f}s.",
            retryable=True,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return
    except Exception as exc:
        logger.exception("Problem job %s failed: %s", job_id, exc)
        job.error = JobError(
            code=ErrorCode.INTERNAL.value,
            message=f"Execution failed: {type(exc).__name__}",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return

    if not raw_result.success:
        job.error = JobError(
            code=ErrorCode.DIVERGED.value,
            message="Simulation diverged or failed.",
            retryable=False,
        )
        job.transition(JobState.FAILED)
        store.update(job)
        return

    from ...core.registry import get_domain

    domain_spec = get_domain(inp.domain or "")
    sanitized = sanitize_result(
        execution_result=raw_result,
        domain_key=inp.domain or "",
        precision=settings.field_precision,
        max_field_points=settings.max_field_points,
        include_fields=inp.return_fields,
        include_coordinates=inp.return_coordinates,
    )

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

    # Validate
    validation = generate_validation_report(sanitized, inp.domain or "")
    job.validation = validation
    job.artifact_hashes["validation"] = content_hash(validation)
    job.transition(JobState.VALIDATED)
    store.update(job)

    # Attest
    claims = generate_claims(sanitized, inp.domain or "")
    certificate = issue_certificate(
        job_id=job.job_id,
        claims=claims,
        input_manifest_hash=job.input_manifest_hash,
        result_hash=result_hash,
        config_hash=content_hash(config.merged_parameters),
        runtime_version=physics_os.RUNTIME_VERSION,
        device_class=settings.device,
    )
    job.certificate = certificate
    job.artifact_hashes["certificate"] = content_hash(certificate)
    job.transition(JobState.ATTESTED)
    store.update(job)

    logger.info("Problem job %s attested: %d claims", job.job_id, len(claims))
