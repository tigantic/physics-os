"""
Performance Harness
====================

Profiling, memory tracking, and scaling-study infrastructure.

- ``PerformanceHarness`` — wraps a solver run with wall-time, peak-memory,
  and per-step timing instrumentation.
- ``ScalingStudy`` — runs the same problem at multiple sizes (N, P) and
  measures strong / weak scaling efficiency.
- ``PerformanceReport`` — structured summary of a single run.
"""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.solvers import RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TimingResult:
    """Timing breakdown for a single solver run."""

    total_wall_s: float
    total_steps: int
    mean_step_s: float
    min_step_s: float
    max_step_s: float
    median_step_s: float
    step_times_s: Tuple[float, ...]

    @property
    def throughput_steps_per_s(self) -> float:
        return self.total_steps / self.total_wall_s if self.total_wall_s > 0 else 0.0


@dataclass(frozen=True)
class MemorySnapshot:
    """Memory usage snapshot."""

    peak_rss_bytes: int
    current_rss_bytes: int
    tensor_allocated_bytes: int
    tensor_reserved_bytes: int

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def tensor_allocated_mb(self) -> float:
        return self.tensor_allocated_bytes / (1024 * 1024)


@dataclass(frozen=True)
class PerformanceReport:
    """Full performance report for a single instrumented run."""

    problem_name: str
    n_cells: int
    timing: TimingResult
    memory: MemorySnapshot
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Performance Report: {self.problem_name}\n"
            f"  Grid cells:            {self.n_cells:,}\n"
            f"  Total wall time:       {self.timing.total_wall_s:.3f} s\n"
            f"  Steps:                 {self.timing.total_steps}\n"
            f"  Mean step time:        {self.timing.mean_step_s:.6f} s\n"
            f"  Min / Max step time:   {self.timing.min_step_s:.6f} / "
            f"{self.timing.max_step_s:.6f} s\n"
            f"  Throughput:            {self.timing.throughput_steps_per_s:.1f} steps/s\n"
            f"  Peak RSS:              {self.memory.peak_rss_mb:.1f} MB\n"
            f"  Tensor allocated:      {self.memory.tensor_allocated_mb:.1f} MB"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory usage."""
    # RSS from /proc (Linux) or psutil fallback
    rss = _get_rss_bytes()
    peak_rss = _get_peak_rss_bytes()

    # PyTorch GPU memory (if available)
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    else:
        allocated = 0
        reserved = 0

    return MemorySnapshot(
        peak_rss_bytes=peak_rss,
        current_rss_bytes=rss,
        tensor_allocated_bytes=allocated,
        tensor_reserved_bytes=reserved,
    )


def _get_rss_bytes() -> int:
    """Get current RSS in bytes (Linux /proc/self/status)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0


def _get_peak_rss_bytes() -> int:
    """Get peak RSS in bytes (Linux /proc/self/status)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Harness
# ═══════════════════════════════════════════════════════════════════════════════


class PerformanceHarness:
    """
    Wraps a solver run with per-step timing and memory instrumentation.

    Usage::

        harness = PerformanceHarness(problem_name="1D-Heat")
        report = harness.run(
            integrator=RK4(),
            state=state0,
            rhs=heat_rhs,
            t_span=(0.0, 1.0),
            dt=1e-4,
        )
        print(report.summary())
    """

    def __init__(
        self,
        problem_name: str = "unnamed",
        warmup_steps: int = 0,
    ) -> None:
        self._problem_name = problem_name
        self._warmup = warmup_steps

    def run(
        self,
        integrator: TimeIntegrator,
        state: SimulationState,
        rhs: RHSCallable,
        t_span: Tuple[float, float],
        dt: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceReport:
        """
        Run the solver with instrumentation.

        Unlike ``TimeIntegrator.solve``, this records per-step wall times.
        """
        gc.collect()
        mem_before = _get_memory_snapshot()

        t0_global, tf = t_span
        if state.t != t0_global:
            state = SimulationState(
                t=t0_global,
                fields=state.fields,
                mesh=state.mesh,
                metadata=state.metadata,
                step_index=state.step_index,
            )

        step_times: List[float] = []
        steps = 0

        while state.t < tf - 1e-14 * abs(dt):
            actual_dt = min(dt, tf - state.t)
            t0_step = time.perf_counter()
            state = integrator.step(state, rhs, actual_dt)
            t_step = time.perf_counter() - t0_step
            steps += 1

            if steps > self._warmup:
                step_times.append(t_step)

        total_wall = sum(step_times) if step_times else 0.0
        mem_after = _get_memory_snapshot()

        if step_times:
            sorted_times = sorted(step_times)
            n = len(sorted_times)
            median = (
                sorted_times[n // 2]
                if n % 2 == 1
                else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2.0
            )
            timing = TimingResult(
                total_wall_s=total_wall,
                total_steps=steps,
                mean_step_s=total_wall / len(step_times),
                min_step_s=sorted_times[0],
                max_step_s=sorted_times[-1],
                median_step_s=median,
                step_times_s=tuple(step_times),
            )
        else:
            timing = TimingResult(
                total_wall_s=0.0,
                total_steps=steps,
                mean_step_s=0.0,
                min_step_s=0.0,
                max_step_s=0.0,
                median_step_s=0.0,
                step_times_s=(),
            )

        return PerformanceReport(
            problem_name=self._problem_name,
            n_cells=state.mesh.n_cells,
            timing=timing,
            memory=MemorySnapshot(
                peak_rss_bytes=max(mem_before.peak_rss_bytes, mem_after.peak_rss_bytes),
                current_rss_bytes=mem_after.current_rss_bytes,
                tensor_allocated_bytes=mem_after.tensor_allocated_bytes,
                tensor_reserved_bytes=mem_after.tensor_reserved_bytes,
            ),
            metadata=metadata or {},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Scaling Study
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScalingResult:
    """Result of a scaling study at multiple problem sizes."""

    study_type: str  # 'strong' or 'weak'
    sizes: Tuple[int, ...]
    wall_times: Tuple[float, ...]
    efficiencies: Tuple[float, ...]
    throughputs: Tuple[float, ...]

    def summary(self) -> str:
        lines = [
            f"Scaling Study ({self.study_type})",
            "=" * 60,
            f"  {'N':>10}  {'wall_s':>10}  {'efficiency':>12}  {'throughput':>14}",
        ]
        for n, w, e, t in zip(self.sizes, self.wall_times, self.efficiencies, self.throughputs):
            lines.append(f"  {n:>10}  {w:>10.4f}  {e:>12.4f}  {t:>14.1f}")
        return "\n".join(lines)


SolverRunFactory = Callable[
    [int],
    Tuple[TimeIntegrator, SimulationState, RHSCallable, Tuple[float, float], float],
]
"""
Factory: (N,) → (integrator, state0, rhs, t_span, dt)
"""


class ScalingStudy:
    """
    Runs the same problem at multiple grid sizes to measure scaling.

    Usage::

        def factory(N):
            mesh = StructuredMesh((N,), ((0., 1.),))
            ...
            return integrator, state0, rhs, (0., 0.1), dt

        study = ScalingStudy(factory)
        result = study.run_strong([64, 128, 256, 512, 1024])
        print(result.summary())
    """

    def __init__(self, factory: SolverRunFactory) -> None:
        self._factory = factory

    def run_strong(
        self,
        sizes: Sequence[int],
        n_repeats: int = 1,
    ) -> ScalingResult:
        """
        Strong scaling: same total work, measure how wall time changes with N.

        Ideal strong scaling: T(N) ∝ N  (linear in grid size).
        """
        wall_times: List[float] = []

        for N in sizes:
            best = float("inf")
            for _ in range(n_repeats):
                integrator, state0, rhs, t_span, dt = self._factory(N)
                t0 = time.perf_counter()
                integrator.solve(state0, rhs, t_span=t_span, dt=dt)
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
            wall_times.append(best)

        # Efficiency relative to smallest problem
        base_time = wall_times[0]
        base_n = sizes[0]
        efficiencies = [
            (base_time * (n / base_n)) / t if t > 0 else 0.0
            for n, t in zip(sizes, wall_times)
        ]
        throughputs = [n / t if t > 0 else 0.0 for n, t in zip(sizes, wall_times)]

        return ScalingResult(
            study_type="strong",
            sizes=tuple(sizes),
            wall_times=tuple(wall_times),
            efficiencies=tuple(efficiencies),
            throughputs=tuple(throughputs),
        )

    def run_weak(
        self,
        sizes: Sequence[int],
        n_repeats: int = 1,
    ) -> ScalingResult:
        """
        Weak scaling: problem size grows proportionally with grid size.

        Ideal weak scaling: T(N) = const.
        """
        wall_times: List[float] = []

        for N in sizes:
            best = float("inf")
            for _ in range(n_repeats):
                integrator, state0, rhs, t_span, dt = self._factory(N)
                t0 = time.perf_counter()
                integrator.solve(state0, rhs, t_span=t_span, dt=dt)
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
            wall_times.append(best)

        base_time = wall_times[0]
        efficiencies = [
            base_time / t if t > 0 else 0.0 for t in wall_times
        ]
        throughputs = [n / t if t > 0 else 0.0 for n, t in zip(sizes, wall_times)]

        return ScalingResult(
            study_type="weak",
            sizes=tuple(sizes),
            wall_times=tuple(wall_times),
            efficiencies=tuple(efficiencies),
            throughputs=tuple(throughputs),
        )
