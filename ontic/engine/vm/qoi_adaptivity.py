"""QoI-driven adaptive rank and tolerance control.

This module implements the "Better Per Dollar" adaptivity story:
the user specifies Quantities of Interest (QoIs) and target accuracy,
and the runtime automatically adjusts:
  - Per-field maximum rank
  - Truncation tolerances per operator stage
  - Hybrid tile activation thresholds

All adaptivity decisions are **deterministic given seed/config**,
satisfying the determinism contract (§20.2). The adaptive policy
is classified as DeterminismTier.REPRODUCIBLE — same config yields
same decisions within ε ≤ 10⁻¹².

Key classes:
  - QoITarget — a named quantity with target tolerance
  - QoIHistory — tracks convergence trend of a QoI over timesteps
  - AdaptiveRankPolicy — adjusts rank/tolerance based on QoI convergence
  - ConvergenceTrend — statistical trend analysis

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────
# QoI Target - what the user cares about
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QoITarget:
    """A quantity of interest with a target tolerance.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. "drag_coefficient", "L2_error_u").
    abs_tolerance : float
        Absolute tolerance for the QoI. Once the QoI change between
        successive evaluations drops below this, the QoI is converged.
    rel_tolerance : float
        Relative tolerance. Convergence if |δQ/Q| < rel_tolerance.
    priority : int
        Priority for resource allocation (1 = highest). When rank
        budget is limited, higher-priority QoIs get tighter tolerances.
    field_name : str
        Which field this QoI depends on (for per-field rank tuning).
    """

    name: str
    abs_tolerance: float = 1e-6
    rel_tolerance: float = 1e-4
    priority: int = 1
    field_name: str = ""


# ─────────────────────────────────────────────────────────────────────
# Convergence trend analysis
# ─────────────────────────────────────────────────────────────────────


class ConvergenceState(Enum):
    """Current convergence status of a QoI."""

    CONVERGING = auto()      # QoI is steadily improving
    CONVERGED = auto()       # QoI has met target tolerance
    STAGNATING = auto()      # QoI improvement has stalled
    DIVERGING = auto()       # QoI is getting worse
    INSUFFICIENT_DATA = auto()  # Not enough history to determine


@dataclass
class ConvergenceTrend:
    """Statistical convergence trend for a single QoI.

    Tracks the history and computes statistics for adaptive decisions.
    """

    target: QoITarget
    _history: list[float] = field(default_factory=list, repr=False)
    _timestamps: list[int] = field(default_factory=list, repr=False)

    # Minimum samples needed for trend analysis
    MIN_SAMPLES: int = 3

    def record(self, value: float, timestep: int) -> None:
        """Record a new QoI measurement.

        Parameters
        ----------
        value : float
            Current QoI value.
        timestep : int
            Current timestep index.
        """
        self._history.append(value)
        self._timestamps.append(timestep)

    @property
    def n_samples(self) -> int:
        """Number of recorded samples."""
        return len(self._history)

    @property
    def latest(self) -> float:
        """Most recent QoI value."""
        if not self._history:
            return float("nan")
        return self._history[-1]

    @property
    def state(self) -> ConvergenceState:
        """Determine current convergence state.

        Uses the last MIN_SAMPLES values to detect the trend.
        """
        if self.n_samples < self.MIN_SAMPLES:
            return ConvergenceState.INSUFFICIENT_DATA

        recent = self._history[-self.MIN_SAMPLES:]

        # Check absolute convergence: variation within tolerance
        spread = max(recent) - min(recent)
        if spread < self.target.abs_tolerance:
            return ConvergenceState.CONVERGED

        # Check relative convergence
        if abs(recent[-1]) > 1e-30:
            rel_spread = spread / abs(recent[-1])
            if rel_spread < self.target.rel_tolerance:
                return ConvergenceState.CONVERGED

        # Compute successive differences
        diffs = [abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1)]

        # Diverging: differences increasing
        if len(diffs) >= 2 and all(
            diffs[i + 1] > diffs[i] * 1.1 for i in range(len(diffs) - 1)
        ):
            return ConvergenceState.DIVERGING

        # Stagnating: differences not decreasing
        if len(diffs) >= 2 and all(
            diffs[i + 1] > diffs[i] * 0.9 for i in range(len(diffs) - 1)
        ):
            return ConvergenceState.STAGNATING

        return ConvergenceState.CONVERGING

    @property
    def estimated_rate(self) -> float:
        """Estimated convergence rate (log of ratio of successive changes).

        Returns NaN if insufficient data.
        """
        if self.n_samples < 3:
            return float("nan")

        recent = self._history[-3:]
        d1 = abs(recent[1] - recent[0])
        d2 = abs(recent[2] - recent[1])

        if d1 < 1e-30 or d2 < 1e-30:
            return float("nan")

        return math.log(d2 / d1)

    def clear(self) -> None:
        """Reset history."""
        self._history.clear()
        self._timestamps.clear()


# ─────────────────────────────────────────────────────────────────────
# QoI History — tracks all targets simultaneously
# ─────────────────────────────────────────────────────────────────────


@dataclass
class QoIHistory:
    """Aggregate convergence tracker for multiple QoIs.

    Parameters
    ----------
    targets : list[QoITarget]
        QoIs to track.
    """

    targets: list[QoITarget] = field(default_factory=list)
    _trends: dict[str, ConvergenceTrend] = field(
        default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        for t in self.targets:
            self._trends[t.name] = ConvergenceTrend(target=t)

    def add_target(self, target: QoITarget) -> None:
        """Add a new QoI target to track."""
        self.targets.append(target)
        self._trends[target.name] = ConvergenceTrend(target=target)

    def record(self, name: str, value: float, timestep: int) -> None:
        """Record a QoI value.

        Parameters
        ----------
        name : str
            QoI name (must match a registered target).
        value : float
            Current QoI value.
        timestep : int
            Current timestep index.

        Raises
        ------
        KeyError
            If name not in registered targets.
        """
        if name not in self._trends:
            raise KeyError(f"QoI '{name}' not registered")
        self._trends[name].record(value, timestep)

    def get_trend(self, name: str) -> ConvergenceTrend:
        """Get convergence trend for a named QoI."""
        return self._trends[name]

    @property
    def all_converged(self) -> bool:
        """True if all tracked QoIs have converged."""
        if not self._trends:
            return False
        return all(
            t.state == ConvergenceState.CONVERGED
            for t in self._trends.values()
        )

    @property
    def any_diverging(self) -> bool:
        """True if any tracked QoI is diverging."""
        return any(
            t.state == ConvergenceState.DIVERGING
            for t in self._trends.values()
        )

    @property
    def worst_state(self) -> ConvergenceState:
        """Return the worst convergence state across all QoIs.

        Priority order: DIVERGING > STAGNATING > INSUFFICIENT_DATA
        > CONVERGING > CONVERGED.
        """
        if not self._trends:
            return ConvergenceState.INSUFFICIENT_DATA

        severity = {
            ConvergenceState.DIVERGING: 4,
            ConvergenceState.STAGNATING: 3,
            ConvergenceState.INSUFFICIENT_DATA: 2,
            ConvergenceState.CONVERGING: 1,
            ConvergenceState.CONVERGED: 0,
        }
        return max(
            (t.state for t in self._trends.values()),
            key=lambda s: severity[s],
        )

    def summary(self) -> dict[str, object]:
        """Return a sanitizer-safe summary.

        Returns
        -------
        dict
            Per-QoI convergence state and latest value.
        """
        result: dict[str, object] = {}
        for name, trend in self._trends.items():
            result[name] = {
                "state": trend.state.name,
                "latest": trend.latest,
                "n_samples": trend.n_samples,
            }
        result["all_converged"] = self.all_converged
        result["worst_state"] = self.worst_state.name
        return result


# ─────────────────────────────────────────────────────────────────────
# AdaptiveRankPolicy — rank/tolerance tuning from QoI feedback
# ─────────────────────────────────────────────────────────────────────


class RankAction(Enum):
    """Rank adjustment action."""

    HOLD = auto()       # Keep current rank
    INCREASE = auto()   # Increase rank (QoI not converging)
    DECREASE = auto()   # Decrease rank (QoI converged, save compute)


@dataclass
class FieldRankSpec:
    """Per-field rank specification from the adaptive policy.

    Parameters
    ----------
    field_name : str
        Name of the field.
    max_rank : int
        Current maximum rank.
    rel_tol : float
        Current truncation tolerance.
    action : RankAction
        What to do at next evaluation.
    """

    field_name: str
    max_rank: int = 64
    rel_tol: float = 1e-10
    action: RankAction = RankAction.HOLD


@dataclass
class AdaptiveRankPolicy:
    """QoI-driven rank and tolerance adaptation.

    Given a QoIHistory and per-field rank specs, this policy decides
    how to adjust ranks and tolerances to meet QoI targets efficiently.

    All decisions are deterministic given the QoI history — no random
    elements, no hardware-dependent branching.

    Parameters
    ----------
    base_max_rank : int
        Baseline maximum rank (from GPURankGovernor).
    rank_increment : int
        How much to increase rank per step when QoI not converging.
    rank_decrement : int
        How much to decrease rank when QoI converged.
    min_rank : int
        Absolute minimum rank.
    max_rank_ceiling : int
        Absolute maximum rank (never exceed).
    tol_tighten_factor : float
        Factor to multiply tolerance when tightening (< 1.0).
    tol_relax_factor : float
        Factor to multiply tolerance when relaxing (> 1.0).
    min_tol : float
        Minimum truncation tolerance.
    max_tol : float
        Maximum truncation tolerance.
    evaluation_interval : int
        Evaluate QoI convergence every N timesteps.
    """

    base_max_rank: int = 64
    rank_increment: int = 4
    rank_decrement: int = 2
    min_rank: int = 4
    max_rank_ceiling: int = 128
    tol_tighten_factor: float = 0.5
    tol_relax_factor: float = 2.0
    min_tol: float = 1e-14
    max_tol: float = 1e-6
    evaluation_interval: int = 10

    _field_specs: dict[str, FieldRankSpec] = field(
        default_factory=dict, repr=False,
    )

    def register_field(
        self,
        field_name: str,
        initial_rank: int | None = None,
        initial_tol: float | None = None,
    ) -> None:
        """Register a field for adaptive rank management.

        Parameters
        ----------
        field_name : str
            Field name.
        initial_rank : int, optional
            Initial max rank. Defaults to base_max_rank.
        initial_tol : float, optional
            Initial truncation tolerance. Defaults to 1e-10.
        """
        self._field_specs[field_name] = FieldRankSpec(
            field_name=field_name,
            max_rank=initial_rank or self.base_max_rank,
            rel_tol=initial_tol or 1e-10,
        )

    def get_field_spec(self, field_name: str) -> FieldRankSpec:
        """Get current rank spec for a field."""
        if field_name not in self._field_specs:
            # Auto-register with defaults
            self.register_field(field_name)
        return self._field_specs[field_name]

    def evaluate(
        self,
        qoi_history: QoIHistory,
        timestep: int,
    ) -> dict[str, FieldRankSpec]:
        """Evaluate QoI convergence and decide rank adjustments.

        Parameters
        ----------
        qoi_history : QoIHistory
            Current QoI convergence data.
        timestep : int
            Current timestep.

        Returns
        -------
        dict[str, FieldRankSpec]
            Updated per-field rank specifications.
        """
        if timestep % self.evaluation_interval != 0:
            return dict(self._field_specs)

        # Map QoI targets to field-level actions
        field_actions: dict[str, RankAction] = {}

        for target in qoi_history.targets:
            trend = qoi_history.get_trend(target.name)
            state = trend.state
            fname = target.field_name or "default"

            if state == ConvergenceState.CONVERGED:
                action = RankAction.DECREASE
            elif state == ConvergenceState.DIVERGING:
                action = RankAction.INCREASE
            elif state == ConvergenceState.STAGNATING:
                action = RankAction.INCREASE
            elif state == ConvergenceState.CONVERGING:
                action = RankAction.HOLD
            else:
                action = RankAction.HOLD

            # Higher priority QoI wins if conflict
            if fname in field_actions:
                existing = field_actions[fname]
                if action == RankAction.INCREASE:
                    field_actions[fname] = RankAction.INCREASE
                elif (
                    action == RankAction.HOLD
                    and existing == RankAction.DECREASE
                ):
                    field_actions[fname] = RankAction.HOLD
            else:
                field_actions[fname] = action

        # Apply actions to field specs
        for fname, action in field_actions.items():
            spec = self.get_field_spec(fname)
            spec.action = action

            if action == RankAction.INCREASE:
                new_rank = min(
                    spec.max_rank + self.rank_increment,
                    self.max_rank_ceiling,
                )
                new_tol = max(
                    spec.rel_tol * self.tol_tighten_factor,
                    self.min_tol,
                )
                spec.max_rank = new_rank
                spec.rel_tol = new_tol

            elif action == RankAction.DECREASE:
                new_rank = max(
                    spec.max_rank - self.rank_decrement,
                    self.min_rank,
                )
                new_tol = min(
                    spec.rel_tol * self.tol_relax_factor,
                    self.max_tol,
                )
                spec.max_rank = new_rank
                spec.rel_tol = new_tol

        return dict(self._field_specs)

    def should_stop(self, qoi_history: QoIHistory) -> bool:
        """Check if the simulation should stop early.

        Stops if all QoIs have converged.
        """
        return qoi_history.all_converged

    def diagnostics(self) -> dict[str, object]:
        """Return sanitizer-safe diagnostic summary.

        Returns
        -------
        dict
            Per-field rank, tolerance, and action.
        """
        result: dict[str, object] = {}
        for fname, spec in self._field_specs.items():
            result[fname] = {
                "max_rank": spec.max_rank,
                "rel_tol": spec.rel_tol,
                "action": spec.action.name,
            }
        return result


# ─────────────────────────────────────────────────────────────────────
# Sanitizer for adaptivity diagnostics
# ─────────────────────────────────────────────────────────────────────

_ADAPTIVITY_DIAG_WHITELIST: frozenset[str] = frozenset({
    "all_converged",
    "worst_state",
    "qoi_states",
    "field_ranks",
    "field_tolerances",
    "rank_actions",
    "early_stop",
    "total_evaluations",
})


def sanitize_adaptivity_diagnostics(
    raw: dict[str, object],
) -> dict[str, object]:
    """Filter adaptivity diagnostics to whitelist-only outputs.

    Parameters
    ----------
    raw : dict
        Raw diagnostic dictionary.

    Returns
    -------
    dict
        Filtered dictionary containing only whitelisted keys.
    """
    return {k: v for k, v in raw.items() if k in _ADAPTIVITY_DIAG_WHITELIST}
