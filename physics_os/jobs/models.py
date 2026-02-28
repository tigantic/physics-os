"""Job model and state machine.

State transitions::

    queued → running → succeeded → validated → attested
                     ↘ failed

Each transition is validated.  Invalid transitions raise
``InvalidTransition``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════


class JobType(str, Enum):
    PHYSICS_VM_EXECUTION = "physics_vm_execution"
    RANK_ATLAS_BENCHMARK = "rank_atlas_benchmark"
    RANK_ATLAS_DIAGNOSTIC = "rank_atlas_diagnostic"
    VALIDATION = "validation"
    ATTESTATION = "attestation"
    FULL_PIPELINE = "full_pipeline"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    VALIDATED = "validated"
    ATTESTED = "attested"


# Valid state transitions
_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {JobState.RUNNING},
    JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED},
    JobState.SUCCEEDED: {JobState.VALIDATED},
    JobState.FAILED: set(),  # terminal
    JobState.VALIDATED: {JobState.ATTESTED},
    JobState.ATTESTED: set(),  # terminal
}

TERMINAL_STATES = {JobState.FAILED, JobState.ATTESTED}


class InvalidTransition(Exception):
    """Raised when a state transition is invalid."""

    def __init__(self, current: JobState, target: JobState) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid transition: {current.value} → {target.value}.  "
            f"Allowed from {current.value}: "
            f"{[s.value for s in _TRANSITIONS.get(current, set())]}"
        )


# ═══════════════════════════════════════════════════════════════════
# Error codes
# ═══════════════════════════════════════════════════════════════════


class ErrorCode(str, Enum):
    INVALID_DOMAIN = "E001"
    PARAM_OUT_OF_RANGE = "E002"
    RESOLUTION_LIMIT = "E003"
    JOB_NOT_FOUND = "E004"
    INVALID_STATE = "E005"
    DIVERGED = "E006"
    TIMEOUT = "E007"
    VALIDATION_FAILED = "E008"
    INVALID_ARTIFACT = "E009"
    RATE_LIMIT = "E010"
    AUTH_FAILED = "E011"
    INTERNAL = "E012"


# ═══════════════════════════════════════════════════════════════════
# Job model
# ═══════════════════════════════════════════════════════════════════


class JobError(BaseModel):
    """Machine-readable error.  Never contains stack traces."""

    code: str = Field(..., pattern=r"^E\d{3}$")
    message: str
    retryable: bool = False


class JobInput(BaseModel):
    """Normalized job input specification."""

    job_type: JobType
    domain: str | None = None
    n_bits: int = 8
    n_steps: int = 100
    dt: float | None = None
    max_rank: int = 64
    parameters: dict[str, Any] = Field(default_factory=dict)
    return_fields: bool = True
    return_coordinates: bool = True
    # For validation/attestation jobs
    artifact_bundle: dict[str, Any] | None = None


class Job(BaseModel):
    """Full job record."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType
    state: JobState = JobState.QUEUED
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    idempotency_key: str | None = None
    api_key_suffix: str = ""

    # Input
    input: JobInput
    input_manifest_hash: str = ""

    # Results (populated as job progresses)
    result: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    certificate: dict[str, Any] | None = None

    # Artifact hashes
    artifact_hashes: dict[str, str | None] = Field(
        default_factory=lambda: {
            "input": None,
            "result": None,
            "validation": None,
            "certificate": None,
        }
    )

    # Versions
    versions: dict[str, str] = Field(
        default_factory=lambda: {
            "api_version": "1.0.0",
            "schema_version": "1.0.0",
            "runtime_version": "3.1.0",
        }
    )

    # Error (only if failed)
    error: JobError | None = None

    def transition(self, target: JobState) -> None:
        """Transition to a new state.  Raises InvalidTransition on failure."""
        allowed = _TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise InvalidTransition(self.state, target)
        self.state = target
        if target in TERMINAL_STATES or target == JobState.SUCCEEDED:
            self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_envelope(self) -> dict[str, Any]:
        """Serialize to the v1 artifact envelope format."""
        return {
            "envelope_version": "1.0.0",
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.state.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "idempotency_key": self.idempotency_key,
            "input_manifest_hash": self.input_manifest_hash,
            "result": self.result,
            "validation": self.validation,
            "certificate": self.certificate,
            "artifact_hashes": self.artifact_hashes,
            "versions": self.versions,
            "error": self.error.model_dump() if self.error else None,
        }

    def to_status(self) -> dict[str, Any]:
        """Lightweight status response (no payloads)."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "state": self.state.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "input_manifest_hash": self.input_manifest_hash,
            "versions": self.versions,
            "error": self.error.model_dump() if self.error else None,
        }
