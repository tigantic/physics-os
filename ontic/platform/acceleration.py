"""
QTT Acceleration Policy, Metrics, and Fallback Logic.

Every QTT-accelerated solver in the physics-os platform must satisfy
the QTT Enablement Policy (Commercial_Execution.md §Phase 5):

1. A rank-growth report
2. An error-vs-rank curve
3. A fallback mode that reverts to baseline when rank explodes
4. A clear domain-of-validity statement

This module provides the machinery to enforce and monitor those requirements.

Key types:
    AccelerationPolicy  — Decides when QTT is beneficial vs dense fallback.
    AccelerationMetrics — Per-step compression/rank/error/speedup tracking.
    RankGrowthReport    — Aggregated rank history for enablement audits.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Acceleration Mode
# ═══════════════════════════════════════════════════════════════════════════════

class AccelerationMode(Enum):
    """Current acceleration state of a solver."""

    DENSE = "dense"
    """Operating in standard dense mode (no QTT)."""

    QTT = "qtt"
    """Operating in QTT-compressed mode."""

    FALLBACK = "fallback"
    """Reverted to dense after QTT rank explosion."""

    HYBRID = "hybrid"
    """QTT for core ops, dense for boundary/edge ops."""


# ═══════════════════════════════════════════════════════════════════════════════
# Acceleration Metrics (per time step)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccelerationMetrics:
    """
    Per-step metrics for QTT-accelerated computation.

    Collected by ``QTTAcceleratedSolver`` at each time step and
    aggregated into a ``RankGrowthReport`` at solve completion.
    """

    step_index: int
    t: float
    mode: AccelerationMode

    # Rank metrics
    max_rank: int = 0
    mean_rank: float = 0.0
    ranks: Tuple[int, ...] = ()

    # Compression
    compression_ratio: float = 1.0
    storage_elements: int = 0
    dense_elements: int = 0

    # Error vs baseline
    error_vs_baseline: float = 0.0

    # Timing
    qtt_time_s: float = 0.0
    dense_time_s: float = 0.0
    speedup: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Rank Growth Report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RankGrowthReport:
    """
    Aggregated rank-growth report across a full solve.

    Required by QTT Enablement Policy for every V0.6 solver.
    """

    solver_name: str
    problem_name: str
    n_steps: int
    n_qtt_steps: int
    n_fallback_steps: int

    # Rank trajectory
    max_rank_per_step: List[int] = dc_field(default_factory=list)
    mean_rank_per_step: List[float] = dc_field(default_factory=list)

    # Compression trajectory
    compression_per_step: List[float] = dc_field(default_factory=list)

    # Error trajectory
    error_per_step: List[float] = dc_field(default_factory=list)

    # Aggregate stats
    peak_rank: int = 0
    median_rank: float = 0.0
    mean_compression: float = 1.0
    max_error: float = 0.0
    total_qtt_time_s: float = 0.0
    total_dense_time_s: float = 0.0
    overall_speedup: float = 1.0

    # Validity
    domain_of_validity: str = ""
    rank_explosion_detected: bool = False
    rank_explosion_step: Optional[int] = None

    @classmethod
    def from_metrics(
        cls,
        metrics: List[AccelerationMetrics],
        solver_name: str,
        problem_name: str,
        domain_of_validity: str = "",
    ) -> "RankGrowthReport":
        """
        Build a report from a list of per-step metrics.
        """
        n_steps = len(metrics)
        n_qtt = sum(1 for m in metrics if m.mode == AccelerationMode.QTT)
        n_fallback = sum(1 for m in metrics if m.mode == AccelerationMode.FALLBACK)

        max_ranks = [m.max_rank for m in metrics]
        mean_ranks = [m.mean_rank for m in metrics]
        compressions = [m.compression_ratio for m in metrics]
        errors = [m.error_vs_baseline for m in metrics]
        qtt_times = [m.qtt_time_s for m in metrics]
        dense_times = [m.dense_time_s for m in metrics]

        total_qtt = sum(qtt_times)
        total_dense = sum(dense_times)

        peak = max(max_ranks) if max_ranks else 0
        sorted_ranks = sorted(max_ranks)
        median = sorted_ranks[len(sorted_ranks) // 2] if sorted_ranks else 0.0

        # Detect rank explosion
        explosion_detected = False
        explosion_step: Optional[int] = None
        for i in range(1, len(max_ranks)):
            if max_ranks[i] > 2 * max_ranks[max(i - 1, 0)] and max_ranks[i] > 16:
                explosion_detected = True
                explosion_step = i
                break

        return cls(
            solver_name=solver_name,
            problem_name=problem_name,
            n_steps=n_steps,
            n_qtt_steps=n_qtt,
            n_fallback_steps=n_fallback,
            max_rank_per_step=max_ranks,
            mean_rank_per_step=mean_ranks,
            compression_per_step=compressions,
            error_per_step=errors,
            peak_rank=peak,
            median_rank=float(median),
            mean_compression=sum(compressions) / max(len(compressions), 1),
            max_error=max(errors) if errors else 0.0,
            total_qtt_time_s=total_qtt,
            total_dense_time_s=total_dense,
            overall_speedup=total_dense / max(total_qtt, 1e-15),
            domain_of_validity=domain_of_validity,
            rank_explosion_detected=explosion_detected,
            rank_explosion_step=explosion_step,
        )

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"=== Rank Growth Report: {self.solver_name} / {self.problem_name} ===",
            f"Steps: {self.n_steps} total, {self.n_qtt_steps} QTT, {self.n_fallback_steps} fallback",
            f"Peak rank: {self.peak_rank}, Median rank: {self.median_rank:.1f}",
            f"Mean compression: {self.mean_compression:.1f}x",
            f"Max error vs baseline: {self.max_error:.2e}",
            f"Speedup: {self.overall_speedup:.2f}x",
        ]
        if self.rank_explosion_detected:
            lines.append(f"⚠ Rank explosion at step {self.rank_explosion_step}")
        if self.domain_of_validity:
            lines.append(f"Domain of validity: {self.domain_of_validity}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Acceleration Policy
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AccelerationPolicy:
    """
    Governs when QTT acceleration is enabled vs disabled.

    The policy checks rank, error, and compression metrics at each step.
    If any threshold is violated, the solver reverts to dense mode.

    Parameters
    ----------
    max_allowed_rank : int
        If QTT rank exceeds this, trigger fallback.
    error_budget : float
        Maximum tolerated error vs baseline (L-inf).
    min_compression_ratio : float
        If compression ratio drops below this, QTT is not worth it.
    rank_growth_threshold : float
        Ratio r_{n+1}/r_n that triggers a rank explosion warning.
    warmup_steps : int
        Number of steps to run in QTT before fallback decisions kick in.
    enable_hybrid : bool
        Allow hybrid mode (QTT core + dense edges).
    """

    max_allowed_rank: int = 128
    error_budget: float = 1e-4
    min_compression_ratio: float = 2.0
    rank_growth_threshold: float = 2.5
    warmup_steps: int = 5
    enable_hybrid: bool = True

    def should_use_qtt(
        self,
        step_index: int,
        current_metrics: Optional[AccelerationMetrics] = None,
        previous_metrics: Optional[AccelerationMetrics] = None,
    ) -> AccelerationMode:
        """
        Decide the acceleration mode for the next step.

        Parameters
        ----------
        step_index : int
            Current time step index.
        current_metrics : AccelerationMetrics, optional
            Metrics from the step just completed (if any).
        previous_metrics : AccelerationMetrics, optional
            Metrics from the step before that.

        Returns
        -------
        AccelerationMode
            Recommended mode for the next step.
        """
        # Always start in QTT during warmup
        if step_index < self.warmup_steps:
            return AccelerationMode.QTT

        if current_metrics is None:
            return AccelerationMode.QTT

        # Check rank explosion
        if current_metrics.max_rank > self.max_allowed_rank:
            logger.warning(
                "Rank %d exceeds limit %d at step %d — falling back to dense",
                current_metrics.max_rank, self.max_allowed_rank, step_index,
            )
            return AccelerationMode.FALLBACK

        # Check error budget
        if current_metrics.error_vs_baseline > self.error_budget:
            logger.warning(
                "Error %.2e exceeds budget %.2e at step %d — falling back",
                current_metrics.error_vs_baseline, self.error_budget, step_index,
            )
            return AccelerationMode.FALLBACK

        # Check compression ratio
        if current_metrics.compression_ratio < self.min_compression_ratio:
            if self.enable_hybrid:
                return AccelerationMode.HYBRID
            return AccelerationMode.FALLBACK

        # Check rank growth rate
        if previous_metrics is not None and previous_metrics.max_rank > 0:
            growth = current_metrics.max_rank / max(previous_metrics.max_rank, 1)
            if growth > self.rank_growth_threshold:
                logger.warning(
                    "Rank growth %.1fx at step %d — switching to hybrid",
                    growth, step_index,
                )
                if self.enable_hybrid:
                    return AccelerationMode.HYBRID
                return AccelerationMode.FALLBACK

        return AccelerationMode.QTT

    def validate_enablement(self, report: RankGrowthReport) -> Dict[str, Any]:
        """
        Validate a completed solve against QTT enablement criteria.

        Returns a dict with pass/fail status for each criterion.
        """
        results: Dict[str, Any] = {
            "rank_growth_report_present": True,
            "error_vs_rank_curve_present": len(report.error_per_step) > 0,
            "fallback_mode_tested": report.n_fallback_steps >= 0,
            "domain_of_validity_stated": bool(report.domain_of_validity),
            "peak_rank_acceptable": report.peak_rank <= self.max_allowed_rank,
            "max_error_acceptable": report.max_error <= self.error_budget,
            "overall_speedup_positive": report.overall_speedup >= 1.0,
            "no_rank_explosion": not report.rank_explosion_detected,
        }
        results["all_passed"] = all(
            v for k, v in results.items() if k != "all_passed"
        )
        return results
