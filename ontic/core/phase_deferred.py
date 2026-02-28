"""
Phase-Deferred Error System
===========================

Provides explicit marking of intentionally deferred functionality,
replacing generic NotImplementedError with auditable phase tracking.

This enables:
- Clear visibility into what is implemented vs. deferred
- Explicit dependency tracking between features
- Automatic documentation of the implementation roadmap
- Reviewer confidence that stubs are deliberate, not forgotten

Usage:
    raise PhaseDeferredError(
        phase="24",
        reason="Adjoint gradient support",
        depends_on=["stable TDVP-CFD", "validated WENO-TT"]
    )

Constitution Compliance: Article I (Transparency), Article III (Honesty)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field


@dataclass
class PhaseDeferredError(Exception):
    """
    Raised when functionality is intentionally deferred to a future phase.

    This is NOT a bug or missing implementation - it's a deliberate
    architectural decision to defer complexity.

    Attributes:
        phase: The phase number where this will be implemented (e.g., "24", "25")
        reason: Human-readable explanation of what is deferred
        depends_on: List of features that must be complete first
        ticket: Optional issue/ticket reference
        eta: Optional estimated completion
    """

    phase: str
    reason: str
    depends_on: list[str] = field(default_factory=list)
    ticket: str | None = None
    eta: str | None = None

    def __post_init__(self):
        # Build the message
        msg_parts = [
            f"Phase {self.phase} deferred: {self.reason}",
        ]
        if self.depends_on:
            msg_parts.append(f"Dependencies: {', '.join(self.depends_on)}")
        if self.ticket:
            msg_parts.append(f"Ticket: {self.ticket}")
        if self.eta:
            msg_parts.append(f"ETA: {self.eta}")

        super().__init__("\n".join(msg_parts))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase,
            "reason": self.reason,
            "depends_on": self.depends_on,
            "ticket": self.ticket,
            "eta": self.eta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhaseDeferredError:
        """Create from dictionary."""
        return cls(
            phase=data["phase"],
            reason=data["reason"],
            depends_on=data.get("depends_on", []),
            ticket=data.get("ticket"),
            eta=data.get("eta"),
        )


# =============================================================================
# Common Phase-Deferred Patterns
# =============================================================================


def phase_24_deferred(feature: str, depends_on: list[str] | None = None):
    """Helper for Phase 24 deferred features."""
    return PhaseDeferredError(
        phase="24",
        reason=feature,
        depends_on=depends_on or ["stable TDVP-CFD", "validated WENO-TT"],
    )


def phase_25_deferred(feature: str, depends_on: list[str] | None = None):
    """Helper for Phase 25 deferred features."""
    return PhaseDeferredError(
        phase="25",
        reason=feature,
        depends_on=depends_on or ["Phase 24 complete", "hardware validation"],
    )


def adjoint_not_implemented():
    """Standard deferred error for adjoint gradients."""
    return PhaseDeferredError(
        phase="24",
        reason="Adjoint gradient computation",
        depends_on=["stable forward solver", "memory-efficient checkpointing"],
        ticket="HYPER-101",
    )


def realtime_not_implemented():
    """Standard deferred error for real-time guarantees."""
    return PhaseDeferredError(
        phase="25",
        reason="Real-time execution guarantees",
        depends_on=["Jetson validation", "WCET analysis"],
        ticket="HYPER-102",
    )


def hardware_not_implemented():
    """Standard deferred error for hardware-specific optimization."""
    return PhaseDeferredError(
        phase="25",
        reason="Hardware-specific optimization",
        depends_on=["GPU backend validated", "memory profiling complete"],
        ticket="HYPER-103",
    )


# =============================================================================
# Registry for Tracking All Deferred Features
# =============================================================================


class DeferredFeatureRegistry:
    """
    Central registry of all phase-deferred features.

    Used by generate_truth_boundary.py to auto-document
    what is implemented vs. deferred.
    """

    _instance = None
    _features: list[dict] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._features = []
        return cls._instance

    @classmethod
    def register(cls, error: PhaseDeferredError, location: str):
        """Register a deferred feature."""
        cls._features.append(
            {
                **error.to_dict(),
                "location": location,
            }
        )

    @classmethod
    def get_all(cls) -> list[dict]:
        """Get all registered features."""
        return cls._features.copy()

    @classmethod
    def get_by_phase(cls, phase: str) -> list[dict]:
        """Get features for a specific phase."""
        return [f for f in cls._features if f["phase"] == phase]

    @classmethod
    def to_json(cls) -> str:
        """Export registry as JSON."""
        return json.dumps(cls._features, indent=2)

    @classmethod
    def clear(cls):
        """Clear registry (for testing)."""
        cls._features = []


def register_deferred(error: PhaseDeferredError, location: str | None = None):
    """
    Register a deferred feature and raise the error.

    Usage:
        register_deferred(
            PhaseDeferredError(phase="24", reason="Feature X"),
            location="ontic/module.py:123"
        )
    """
    if location is None:
        # Auto-detect location
        frame = sys._getframe(1)
        location = f"{frame.f_code.co_filename}:{frame.f_lineno}"

    DeferredFeatureRegistry.register(error, location)
    raise error


# =============================================================================
# Decorator for Deferred Methods
# =============================================================================


def phase_deferred(phase: str, reason: str, depends_on: list[str] | None = None):
    """
    Decorator to mark a method as phase-deferred.

    Usage:
        @phase_deferred("24", "Adjoint gradients", ["forward solver"])
        def compute_adjoint(self):
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            raise PhaseDeferredError(
                phase=phase,
                reason=f"{func.__name__}: {reason}",
                depends_on=depends_on or [],
            )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = f"[Phase {phase} Deferred] {reason}\n\n{func.__doc__ or ''}"
        wrapper._phase_deferred = True
        wrapper._phase = phase
        wrapper._reason = reason
        return wrapper

    return decorator


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "PhaseDeferredError",
    "phase_24_deferred",
    "phase_25_deferred",
    "adjoint_not_implemented",
    "realtime_not_implemented",
    "hardware_not_implemented",
    "DeferredFeatureRegistry",
    "register_deferred",
    "phase_deferred",
]
