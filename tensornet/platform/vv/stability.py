"""
Numerical Stability Monitors
==============================

Runtime monitors that detect stability violations during simulation:

- **CFL checker** — verifies the Courant-Friedrichs-Lewy condition.
- **Blow-up detector** — catches NaN, ±Inf, and exponential growth.
- **Stiffness estimator** — estimates the spectral radius of the Jacobian
  to flag stiff regimes that need implicit treatment.

These monitors can be used inline (checked every N steps) or post-hoc
on recorded observable histories.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# StabilityCheck base
# ═══════════════════════════════════════════════════════════════════════════════


class StabilityCheck(ABC):
    """
    Base class for a stability check that can be evaluated on a SimulationState.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def check(self, state: SimulationState, dt: float) -> "StabilityVerdict":
        """Run the check and return a verdict."""
        ...


@dataclass(frozen=True)
class StabilityVerdict:
    """Result of a single stability check."""

    check_name: str
    passed: bool
    metric_value: float
    threshold: float
    message: str = ""

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"StabilityVerdict({self.check_name}: {status}, "
            f"value={self.metric_value:.4e}, threshold={self.threshold:.4e})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CFL Checker
# ═══════════════════════════════════════════════════════════════════════════════


class CFLChecker(StabilityCheck):
    """
    Verifies the CFL condition for explicit time integration.

    For a scalar advection equation:  CFL = |u_max| * dt / dx
    For a diffusion equation:        CFL = alpha * dt / dx^2

    The checker supports both modes:

    - ``mode='advection'``: CFL = max(|field|) * dt / dx
    - ``mode='diffusion'``: CFL = coeff * dt / dx^2

    Parameters
    ----------
    field_name : str
        The field to compute wave speed from (advection mode)
        or ignored (diffusion mode, uses *coeff*).
    mode : str
        ``'advection'`` or ``'diffusion'``.
    max_cfl : float
        Maximum allowable CFL number.
    coeff : float
        Diffusion coefficient (only used in diffusion mode).
    """

    def __init__(
        self,
        field_name: str = "u",
        mode: str = "advection",
        max_cfl: float = 1.0,
        coeff: float = 1.0,
    ) -> None:
        if mode not in ("advection", "diffusion"):
            raise ValueError(f"mode must be 'advection' or 'diffusion', got {mode!r}")
        self._field_name = field_name
        self._mode = mode
        self._max_cfl = max_cfl
        self._coeff = coeff

    @property
    def name(self) -> str:
        return f"CFL({self._mode})"

    def compute_cfl(self, state: SimulationState, dt: float) -> float:
        """Compute the CFL number for the current state."""
        mesh = state.mesh
        if not isinstance(mesh, StructuredMesh):
            raise NotImplementedError("CFL checker requires StructuredMesh")

        dx = min(mesh.dx)

        if self._mode == "advection":
            field = state.get_field(self._field_name)
            u_max = field.data.abs().max().item()
            return u_max * dt / dx
        else:  # diffusion
            return self._coeff * dt / (dx ** 2)

    def check(self, state: SimulationState, dt: float) -> StabilityVerdict:
        cfl = self.compute_cfl(state, dt)
        passed = cfl <= self._max_cfl
        if not passed:
            logger.warning(
                "CFL violation: %s = %.4e > %.4e at t=%.4e",
                self.name, cfl, self._max_cfl, state.t,
            )
        return StabilityVerdict(
            check_name=self.name,
            passed=passed,
            metric_value=cfl,
            threshold=self._max_cfl,
            message="" if passed else f"CFL={cfl:.4e} exceeds limit={self._max_cfl:.4e}",
        )

    def max_stable_dt(
        self, state: SimulationState, safety: float = 0.9
    ) -> float:
        """
        Compute the maximum stable timestep for the current state.

        Parameters
        ----------
        state : SimulationState
        safety : float
            Safety factor (< 1) to keep CFL below the limit.
        """
        mesh = state.mesh
        if not isinstance(mesh, StructuredMesh):
            raise NotImplementedError("CFL checker requires StructuredMesh")

        dx = min(mesh.dx)

        if self._mode == "advection":
            field = state.get_field(self._field_name)
            u_max = field.data.abs().max().item()
            if u_max < 1e-30:
                return 1.0  # effectively no constraint
            return safety * self._max_cfl * dx / u_max
        else:  # diffusion
            return safety * self._max_cfl * dx ** 2 / self._coeff


# ═══════════════════════════════════════════════════════════════════════════════
# Blow-up Detector
# ═══════════════════════════════════════════════════════════════════════════════


class BlowupDetector(StabilityCheck):
    """
    Detects numerical blow-up in the solution fields.

    Checks for:
    1. NaN values in any monitored field.
    2. ±Inf values.
    3. Maximum absolute value exceeding a threshold (e.g. 1e15).
    4. Growth rate exceeding a threshold between consecutive checks.

    Parameters
    ----------
    field_names : sequence of str
        Fields to monitor.
    max_value : float
        Absolute value threshold.
    max_growth_rate : float
        Max ratio of consecutive L∞ norms before flagging growth.
    """

    def __init__(
        self,
        field_names: Sequence[str] = ("u",),
        max_value: float = 1e15,
        max_growth_rate: float = 100.0,
    ) -> None:
        self._field_names = list(field_names)
        self._max_value = max_value
        self._max_growth_rate = max_growth_rate
        self._prev_norms: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "BlowupDetector"

    def reset(self) -> None:
        """Reset internal state (for reuse across simulations)."""
        self._prev_norms.clear()

    def check(self, state: SimulationState, dt: float) -> StabilityVerdict:
        worst_value = 0.0
        worst_ratio = 0.0
        has_nan = False
        has_inf = False
        messages: List[str] = []

        for fname in self._field_names:
            try:
                field = state.get_field(fname)
            except KeyError:
                continue

            data = field.data

            # Check NaN
            nan_count = torch.isnan(data).sum().item()
            if nan_count > 0:
                has_nan = True
                messages.append(f"{fname}: {int(nan_count)} NaN values")

            # Check Inf
            inf_count = torch.isinf(data).sum().item()
            if inf_count > 0:
                has_inf = True
                messages.append(f"{fname}: {int(inf_count)} Inf values")

            # Check max value
            max_abs = data.abs().max().item() if not (has_nan or has_inf) else float("inf")
            worst_value = max(worst_value, max_abs)
            if max_abs > self._max_value:
                messages.append(
                    f"{fname}: max|u|={max_abs:.4e} > threshold={self._max_value:.4e}"
                )

            # Check growth rate
            if fname in self._prev_norms and self._prev_norms[fname] > 1e-30:
                ratio = max_abs / self._prev_norms[fname]
                worst_ratio = max(worst_ratio, ratio)
                if ratio > self._max_growth_rate:
                    messages.append(
                        f"{fname}: growth ratio={ratio:.4e} > limit={self._max_growth_rate:.4e}"
                    )
            self._prev_norms[fname] = max_abs

        passed = not (
            has_nan
            or has_inf
            or worst_value > self._max_value
            or worst_ratio > self._max_growth_rate
        )

        if not passed:
            logger.warning(
                "Blow-up detected at t=%.6e: %s",
                state.t, "; ".join(messages),
            )

        return StabilityVerdict(
            check_name=self.name,
            passed=passed,
            metric_value=worst_value,
            threshold=self._max_value,
            message="; ".join(messages),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Stiffness Estimator
# ═══════════════════════════════════════════════════════════════════════════════


class StiffnessEstimator(StabilityCheck):
    """
    Estimates the spectral radius of the RHS Jacobian using power iteration.

    A large spectral radius (relative to 1/dt) indicates stiffness and
    suggests that implicit or IMEX methods are needed.

    Parameters
    ----------
    rhs : callable(state, t) → {field: Tensor}
        The right-hand side function.
    field_name : str
        Which field to perturb for Jacobian estimation.
    n_iterations : int
        Number of power iterations.
    epsilon : float
        Finite-difference perturbation magnitude.
    stiffness_threshold : float
        spectral_radius * dt threshold above which the problem is flagged stiff.
    """

    def __init__(
        self,
        rhs: Callable[[SimulationState, float], Dict[str, Tensor]],
        field_name: str = "u",
        n_iterations: int = 10,
        epsilon: float = 1e-6,
        stiffness_threshold: float = 2.0,
    ) -> None:
        self._rhs = rhs
        self._field_name = field_name
        self._n_iter = n_iterations
        self._eps = epsilon
        self._threshold = stiffness_threshold

    @property
    def name(self) -> str:
        return "StiffnessEstimator"

    def estimate_spectral_radius(
        self, state: SimulationState, dt: float
    ) -> float:
        """
        Estimate the spectral radius of the spatially-discrete RHS Jacobian.

        Uses power iteration with finite-difference Jacobian-vector products.
        """
        field = state.get_field(self._field_name)
        n = field.data.numel()

        # Random initial vector
        v = torch.randn(n, dtype=field.data.dtype, device=field.data.device)
        v = v / torch.norm(v)

        f0 = self._rhs(state, state.t)[self._field_name].flatten()

        eigenvalue_est = 0.0
        for _ in range(self._n_iter):
            # Perturb state in direction v
            perturbed_data = field.data.flatten() + self._eps * v
            perturbed_field = FieldData(
                name=self._field_name,
                data=perturbed_data.reshape(field.data.shape),
                mesh=state.mesh,
                components=field.components,
                units=field.units,
            )
            perturbed_state = state.with_fields(
                **{self._field_name: perturbed_field}
            )

            f1 = self._rhs(perturbed_state, state.t)[self._field_name].flatten()
            Jv = (f1 - f0) / self._eps

            eigenvalue_est = torch.norm(Jv).item()
            if eigenvalue_est < 1e-30:
                break
            v = Jv / eigenvalue_est

        return eigenvalue_est

    def check(self, state: SimulationState, dt: float) -> StabilityVerdict:
        rho = self.estimate_spectral_radius(state, dt)
        stiffness_number = rho * dt
        passed = stiffness_number <= self._threshold

        if not passed:
            logger.warning(
                "Stiffness detected at t=%.6e: spectral_radius=%.4e, "
                "stiffness_number=%.4e (threshold=%.4e)",
                state.t, rho, stiffness_number, self._threshold,
            )

        return StabilityVerdict(
            check_name=self.name,
            passed=passed,
            metric_value=stiffness_number,
            threshold=self._threshold,
            message=(
                f"spectral_radius={rho:.4e}, dt*rho={stiffness_number:.4e}"
                if not passed else ""
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Stability Report — aggregate
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StabilityReport:
    """
    Aggregate stability report from running all checks over a simulation.

    Call ``add_verdict`` at each monitored step, then ``finalize`` to
    compute summary statistics.
    """

    verdicts: List[StabilityVerdict] = dc_field(default_factory=list)
    _finalized: bool = dc_field(default=False, repr=False)
    _n_violations: int = dc_field(default=0, repr=False)
    _first_violation_step: Optional[int] = dc_field(default=None, repr=False)

    def add_verdict(self, verdict: StabilityVerdict, step: int = 0) -> None:
        """Record a verdict at a given step."""
        self.verdicts.append(verdict)
        if not verdict.passed:
            self._n_violations += 1
            if self._first_violation_step is None:
                self._first_violation_step = step

    @property
    def passed(self) -> bool:
        """True if no stability violations were recorded."""
        return self._n_violations == 0

    @property
    def n_violations(self) -> int:
        return self._n_violations

    @property
    def first_violation_step(self) -> Optional[int]:
        return self._first_violation_step

    def summary(self) -> str:
        """Human-readable summary of all stability checks."""
        lines = [
            "Stability Report",
            "=" * 60,
            f"  Total checks:          {len(self.verdicts)}",
            f"  Violations:            {self._n_violations}",
            f"  First violation step:  {self._first_violation_step}",
            f"  Overall verdict:       {'PASS' if self.passed else 'FAIL'}",
        ]
        if self._n_violations > 0:
            lines.append("")
            lines.append("  Violations:")
            for v in self.verdicts:
                if not v.passed:
                    lines.append(f"    - {v}")
        return "\n".join(lines)


def run_stability_checks(
    state: SimulationState,
    dt: float,
    checks: Sequence[StabilityCheck],
    step: int = 0,
    report: Optional[StabilityReport] = None,
) -> StabilityReport:
    """
    Convenience function: run all checks on *state* and accumulate into *report*.

    Parameters
    ----------
    state : SimulationState
    dt : float
    checks : sequence of StabilityCheck
    step : int
        Current step index (for violation tracking).
    report : StabilityReport, optional
        Accumulator; a new one is created if not provided.

    Returns
    -------
    StabilityReport
    """
    if report is None:
        report = StabilityReport()
    for chk in checks:
        verdict = chk.check(state, dt)
        report.add_verdict(verdict, step=step)
    return report
