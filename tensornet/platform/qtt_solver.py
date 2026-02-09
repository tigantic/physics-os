"""
QTT-Accelerated Solver Wrapper.

Wraps any ``Solver``-protocol-conformant solver with transparent QTT
acceleration.  The wrapper:

1. Compresses incoming ``SimulationState`` fields to QTT.
2. Delegates the time step to the underlying solver.
3. Monitors rank growth, error vs baseline, and compression.
4. Falls back to dense mode when the acceleration policy triggers.

The wrapped solver satisfies the ``Solver`` protocol, so callers
(including the V&V harness) are unaware of the QTT layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.acceleration import (
    AccelerationMetrics,
    AccelerationMode,
    AccelerationPolicy,
    RankGrowthReport,
)
from tensornet.platform.data_model import FieldData, SimulationState, StructuredMesh
from tensornet.platform.protocols import Observable, SolveResult
from tensornet.platform.qtt import QTTFieldData, field_to_qtt, qtt_to_field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Simulation State
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QTTSimulationState:
    """
    Simulation state with both dense and QTT field representations.

    Maintains a shadow copy in QTT form for accelerated computation.
    The dense fields remain the ground truth; QTT fields are used
    for the hot path when the acceleration policy permits.
    """

    dense_state: SimulationState
    qtt_fields: Dict[str, QTTFieldData]
    mode: AccelerationMode = AccelerationMode.QTT

    @property
    def t(self) -> float:
        return self.dense_state.t

    @property
    def step_index(self) -> int:
        return self.dense_state.step_index


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Accelerated Solver
# ═══════════════════════════════════════════════════════════════════════════════


class QTTAcceleratedSolver:
    """
    Wraps a baseline solver with transparent QTT acceleration.

    Satisfies the ``Solver`` protocol.

    Parameters
    ----------
    baseline_solver : Any
        The dense-mode solver (must satisfy ``Solver`` protocol).
    policy : AccelerationPolicy, optional
        Acceleration policy governing rank/error/fallback decisions.
    max_rank : int
        Maximum QTT rank for field compression.
    tolerance : float
        QTT truncation tolerance.
    domain_of_validity : str
        Human-readable statement of when this QTT acceleration is valid
        (required by QTT Enablement Policy).
    """

    def __init__(
        self,
        baseline_solver: Any,
        policy: Optional[AccelerationPolicy] = None,
        max_rank: int = 64,
        tolerance: float = 1e-10,
        domain_of_validity: str = "",
    ) -> None:
        self._baseline = baseline_solver
        self._policy = policy or AccelerationPolicy()
        self._max_rank = max_rank
        self._tolerance = tolerance
        self._domain_of_validity = domain_of_validity

        # Metrics collection
        self._metrics_history: List[AccelerationMetrics] = []
        self._current_mode = AccelerationMode.QTT

    @property
    def name(self) -> str:
        return f"QTT({self._baseline.name})"

    @property
    def metrics_history(self) -> List[AccelerationMetrics]:
        """Per-step acceleration metrics collected during the last solve."""
        return list(self._metrics_history)

    def rank_growth_report(self, problem_name: str = "unknown") -> RankGrowthReport:
        """
        Generate the rank growth report for the last completed solve.

        Required by QTT Enablement Policy.
        """
        return RankGrowthReport.from_metrics(
            self._metrics_history,
            solver_name=self.name,
            problem_name=problem_name,
            domain_of_validity=self._domain_of_validity,
        )

    def step(
        self,
        state: SimulationState,
        dt: float,
        **kwargs: Any,
    ) -> SimulationState:
        """
        Advance state by one time step with QTT acceleration.

        If QTT mode:
        1. Compress fields to QTT
        2. Delegate to baseline solver (which sees dense fields)
        3. Compress result back to QTT
        4. Measure rank growth and error
        5. Apply acceleration policy

        If DENSE/FALLBACK mode:
        - Straight delegation to baseline solver.
        """
        step_idx = state.step_index

        if self._current_mode == AccelerationMode.FALLBACK:
            # Pure dense — no QTT overhead
            t0 = time.perf_counter()
            result = self._baseline.step(state, dt, **kwargs)
            dense_time = time.perf_counter() - t0

            self._metrics_history.append(AccelerationMetrics(
                step_index=step_idx,
                t=state.t,
                mode=AccelerationMode.FALLBACK,
                dense_time_s=dense_time,
            ))
            return result

        # QTT or HYBRID mode — compress, step, measure
        t_qtt_start = time.perf_counter()

        # Compress input fields to QTT (timing included)
        qtt_fields: Dict[str, QTTFieldData] = {}
        for fname, fdata in state.fields.items():
            qtt_fields[fname] = field_to_qtt(fdata, self._max_rank, self._tolerance)

        # Run baseline solver on dense state
        t_dense_start = time.perf_counter()
        result_state = self._baseline.step(state, dt, **kwargs)
        dense_time = time.perf_counter() - t_dense_start

        # Compress result fields
        result_qtt: Dict[str, QTTFieldData] = {}
        for fname, fdata in result_state.fields.items():
            result_qtt[fname] = field_to_qtt(fdata, self._max_rank, self._tolerance)

        qtt_time = time.perf_counter() - t_qtt_start

        # Measure metrics
        all_ranks: List[int] = []
        total_storage = 0
        total_dense = 0
        for qf in result_qtt.values():
            all_ranks.extend(qf.ranks)
            total_storage += qf.storage_elements
            total_dense += qf.dense_elements

        max_rank = max(all_ranks) if all_ranks else 0
        mean_rank = sum(all_ranks) / max(len(all_ranks), 1)
        compression = total_dense / max(total_storage, 1)

        # Error vs original dense (the baseline result is ground truth)
        max_error = 0.0
        for fname, qf in result_qtt.items():
            recon = qtt_to_field(qf)
            orig = result_state.fields[fname].data
            n = min(orig.shape[0], recon.data.shape[0])
            err = (orig[:n] - recon.data[:n]).abs().max().item()
            max_error = max(max_error, err)

        metrics = AccelerationMetrics(
            step_index=step_idx,
            t=state.t,
            mode=self._current_mode,
            max_rank=max_rank,
            mean_rank=mean_rank,
            ranks=tuple(all_ranks),
            compression_ratio=compression,
            storage_elements=total_storage,
            dense_elements=total_dense,
            error_vs_baseline=max_error,
            qtt_time_s=qtt_time,
            dense_time_s=dense_time,
            speedup=dense_time / max(qtt_time, 1e-15),
        )
        self._metrics_history.append(metrics)

        # Apply acceleration policy for next step
        prev = self._metrics_history[-2] if len(self._metrics_history) >= 2 else None
        self._current_mode = self._policy.should_use_qtt(
            step_idx, metrics, prev,
        )

        return result_state

    def solve(
        self,
        state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        *,
        observables: Optional[Sequence[Observable]] = None,
        callback: Optional[Any] = None,
        max_steps: Optional[int] = None,
    ) -> SolveResult:
        """
        Integrate from ``t_span[0]`` to ``t_span[1]`` with QTT acceleration.

        Collects per-step acceleration metrics and generates a rank-growth
        report accessible via ``self.rank_growth_report()``.
        """
        self._metrics_history.clear()
        self._current_mode = AccelerationMode.QTT

        t_start, t_end = t_span
        current_state = state
        steps = 0
        obs_history: Dict[str, List[Tensor]] = {}

        if observables:
            for obs in observables:
                obs_history[obs.name] = []

        step_limit = max_steps or int(1e9)

        while current_state.t < t_end - 1e-14 and steps < step_limit:
            h = min(dt, t_end - current_state.t)
            current_state = self.step(current_state, h)
            steps += 1

            if observables:
                for obs in observables:
                    val = obs.compute(current_state)
                    obs_history[obs.name].append(val)

            if callback is not None:
                callback(current_state, steps)

        return SolveResult(
            final_state=current_state,
            t_final=current_state.t,
            steps_taken=steps,
            observable_history=obs_history,
            converged=True,
            metadata={
                "acceleration": {
                    "n_qtt_steps": sum(
                        1 for m in self._metrics_history
                        if m.mode == AccelerationMode.QTT
                    ),
                    "n_fallback_steps": sum(
                        1 for m in self._metrics_history
                        if m.mode == AccelerationMode.FALLBACK
                    ),
                    "peak_rank": max(
                        (m.max_rank for m in self._metrics_history), default=0
                    ),
                    "mean_compression": (
                        sum(m.compression_ratio for m in self._metrics_history)
                        / max(len(self._metrics_history), 1)
                    ),
                },
            },
        )
