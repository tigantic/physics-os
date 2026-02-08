"""
Benchmark Registry
===================

A registry of canonical benchmark problems with:

- Reference / analytical solutions (golden outputs).
- Acceptance thresholds for error norms.
- Metadata (source paper, domain, V-state tier).

Any solver that passes a registered benchmark earns the corresponding
ledger advancement (V0.4 → V0.5).

Usage::

    from tensornet.platform.vv.benchmarks import get_benchmark_registry
    registry = get_benchmark_registry()
    registry.register(my_benchmark)

    result = registry.run("heat_1d_dirichlet", solver_factory)
    assert result.passed
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.solvers import RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# Golden Output
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GoldenOutput:
    """
    Reference solution for a benchmark problem at a specific resolution.

    Can be either an analytical function or a stored tensor with hash.
    """

    resolution: int
    t_final: float
    linf_error_bound: float
    l2_error_bound: float
    exact_fn: Optional[Callable[[Tensor, float], Tensor]] = None
    stored_data: Optional[Tensor] = None
    data_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.exact_fn is None and self.stored_data is None:
            raise ValueError(
                "GoldenOutput must have either exact_fn or stored_data"
            )
        if self.stored_data is not None and self.data_hash is None:
            # Auto-compute hash
            h = hashlib.sha256(self.stored_data.numpy().tobytes()).hexdigest()
            object.__setattr__(self, "data_hash", h)

    def reference(self, x: Tensor, t: float) -> Tensor:
        """Get the reference solution at coordinates *x* and time *t*."""
        if self.exact_fn is not None:
            return self.exact_fn(x, t)
        if self.stored_data is not None:
            return self.stored_data.clone()
        raise RuntimeError("No reference available")

    def verify_hash(self) -> bool:
        """Verify integrity of stored data against recorded hash."""
        if self.stored_data is None or self.data_hash is None:
            return True  # no stored data to verify
        computed = hashlib.sha256(
            self.stored_data.numpy().tobytes()
        ).hexdigest()
        return computed == self.data_hash


# ═══════════════════════════════════════════════════════════════════════════════
# BenchmarkProblem
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkProblem:
    """
    A canonical benchmark problem with reference solutions and acceptance criteria.

    Parameters
    ----------
    name : str
        Unique identifier (e.g. ``'heat_1d_dirichlet'``).
    description : str
        Human-readable description.
    domain_pack : str
        Domain pack ID (e.g. ``'I'``, ``'II'``).
    golden_outputs : dict
        ``{resolution: GoldenOutput}``
    setup_fn : callable
        ``(mesh, golden) → (integrator, state0, rhs, t_span, dt)``
        Sets up the solver for the given mesh and golden output.
    default_resolutions : tuple of int
        Default grid sizes for running the benchmark.
    convergence_order : int
        Expected convergence order.
    source : str
        Citation / reference.
    tags : list of str
        Classification tags.
    """

    name: str
    description: str
    domain_pack: str
    golden_outputs: Dict[int, GoldenOutput]
    setup_fn: Callable[
        [StructuredMesh, GoldenOutput],
        Tuple[TimeIntegrator, SimulationState, RHSCallable, Tuple[float, float], float],
    ]
    default_resolutions: Tuple[int, ...] = (32, 64, 128, 256)
    convergence_order: int = 2
    source: str = ""
    tags: List[str] = dc_field(default_factory=list)

    def available_resolutions(self) -> List[int]:
        return sorted(self.golden_outputs.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# BenchmarkResult
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of running a single benchmark at one or more resolutions."""

    benchmark_name: str
    resolution_results: List["ResolutionResult"]
    observed_order_linf: float
    observed_order_l2: float
    passed: bool

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.benchmark_name}",
            "=" * 60,
            f"  Observed order (L∞): {self.observed_order_linf:.3f}",
            f"  Observed order (L2): {self.observed_order_l2:.3f}",
            f"  Verdict:             {'PASS' if self.passed else 'FAIL'}",
            "",
            f"  {'N':>6}  {'L∞ err':>12}  {'L∞ bound':>12}  "
            f"{'L2 err':>12}  {'L2 bound':>12}  {'status':>8}",
        ]
        for r in self.resolution_results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  {r.resolution:>6}  {r.linf_error:>12.4e}  "
                f"{r.linf_bound:>12.4e}  {r.l2_error:>12.4e}  "
                f"{r.l2_bound:>12.4e}  {status:>8}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ResolutionResult:
    """Result at a single resolution."""

    resolution: int
    linf_error: float
    l2_error: float
    linf_bound: float
    l2_bound: float
    passed: bool
    wall_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BenchmarkRegistry
# ═══════════════════════════════════════════════════════════════════════════════


class BenchmarkRegistry:
    """
    Central registry of benchmark problems.

    Benchmarks are registered by name and can be run individually or as a suite.
    """

    def __init__(self) -> None:
        self._benchmarks: Dict[str, BenchmarkProblem] = {}

    def register(self, benchmark: BenchmarkProblem) -> None:
        """Register a benchmark problem."""
        if benchmark.name in self._benchmarks:
            raise ValueError(
                f"Benchmark '{benchmark.name}' already registered"
            )
        self._benchmarks[benchmark.name] = benchmark

    def get(self, name: str) -> BenchmarkProblem:
        """Retrieve a registered benchmark."""
        if name not in self._benchmarks:
            raise KeyError(
                f"Benchmark '{name}' not found. "
                f"Available: {list(self._benchmarks.keys())}"
            )
        return self._benchmarks[name]

    def list_benchmarks(self) -> List[str]:
        """List all registered benchmark names."""
        return sorted(self._benchmarks.keys())

    def list_by_domain(self, domain_pack: str) -> List[str]:
        """List benchmarks for a specific domain pack."""
        return sorted(
            name
            for name, b in self._benchmarks.items()
            if b.domain_pack == domain_pack
        )

    def list_by_tag(self, tag: str) -> List[str]:
        """List benchmarks with a specific tag."""
        return sorted(
            name
            for name, b in self._benchmarks.items()
            if tag in b.tags
        )

    def run(
        self,
        name: str,
        resolutions: Optional[Sequence[int]] = None,
        field_name: str = "u",
    ) -> BenchmarkResult:
        """
        Run a benchmark at one or more resolutions.

        Parameters
        ----------
        name : str
            Benchmark name.
        resolutions : optional sequence of int
            Grid sizes to run at. Defaults to the benchmark's
            ``default_resolutions``.
        field_name : str
            Which field to compare against the golden output.

        Returns
        -------
        BenchmarkResult
        """
        import time

        benchmark = self.get(name)
        res_list = resolutions or benchmark.default_resolutions

        resolution_results: List[ResolutionResult] = []
        dxs: List[float] = []
        linf_errors: List[float] = []
        l2_errors: List[float] = []

        for N in res_list:
            # Get golden output — use the closest available resolution
            golden = _find_golden(benchmark, N)
            domain = ((0.0, 1.0),)  # default 1-D domain
            mesh = StructuredMesh(shape=(N,), domain=domain)
            dx = mesh.dx[0]

            integrator, state0, rhs, t_span, dt = benchmark.setup_fn(mesh, golden)

            t0 = time.perf_counter()
            result = integrator.solve(state0, rhs, t_span=t_span, dt=dt)
            wall = time.perf_counter() - t0

            # Compare against golden
            x = mesh.cell_centers().squeeze(-1)
            u_num = result.final_state.get_field(field_name).data
            u_ref = golden.reference(x, golden.t_final)

            err = u_num - u_ref
            linf = err.abs().max().item()
            l2 = math.sqrt((err ** 2 * dx).sum().item())

            passed = (
                linf <= golden.linf_error_bound
                and l2 <= golden.l2_error_bound
            )

            resolution_results.append(ResolutionResult(
                resolution=N,
                linf_error=linf,
                l2_error=l2,
                linf_bound=golden.linf_error_bound,
                l2_bound=golden.l2_error_bound,
                passed=passed,
                wall_time_s=wall,
            ))

            dxs.append(dx)
            linf_errors.append(linf)
            l2_errors.append(l2)

        # Compute convergence orders
        obs_linf = _compute_order(dxs, linf_errors) if len(dxs) >= 2 else 0.0
        obs_l2 = _compute_order(dxs, l2_errors) if len(dxs) >= 2 else 0.0

        all_passed = all(r.passed for r in resolution_results)

        return BenchmarkResult(
            benchmark_name=name,
            resolution_results=resolution_results,
            observed_order_linf=obs_linf,
            observed_order_l2=obs_l2,
            passed=all_passed,
        )

    def run_suite(
        self,
        names: Optional[Sequence[str]] = None,
        domain_pack: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run multiple benchmarks. Filter by name list, domain, or tag."""
        if names is not None:
            targets = list(names)
        elif domain_pack is not None:
            targets = self.list_by_domain(domain_pack)
        elif tag is not None:
            targets = self.list_by_tag(tag)
        else:
            targets = self.list_benchmarks()

        results: Dict[str, BenchmarkResult] = {}
        for name in targets:
            results[name] = self.run(name)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _find_golden(benchmark: BenchmarkProblem, N: int) -> GoldenOutput:
    """
    Find the best golden output for resolution *N*.

    If an exact golden exists for *N*, use it.  Otherwise, use the
    golden at the highest available resolution that has an ``exact_fn``
    (analytical goldens work at any resolution).
    """
    if N in benchmark.golden_outputs:
        return benchmark.golden_outputs[N]

    # Fallback: find any golden with exact_fn
    for res in sorted(benchmark.golden_outputs.keys()):
        golden = benchmark.golden_outputs[res]
        if golden.exact_fn is not None:
            return golden

    raise ValueError(
        f"No golden output available for benchmark '{benchmark.name}' "
        f"at resolution N={N}. Available: {benchmark.available_resolutions()}"
    )


def _compute_order(params: List[float], errors: List[float]) -> float:
    """Least-squares log-log slope."""
    valid = [
        (math.log(p), math.log(e))
        for p, e in zip(params, errors)
        if p > 0 and e > 0
    ]
    if len(valid) < 2:
        return 0.0
    n = len(valid)
    sx = sum(v[0] for v in valid)
    sy = sum(v[1] for v in valid)
    sxx = sum(v[0] ** 2 for v in valid)
    sxy = sum(v[0] * v[1] for v in valid)
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-30:
        return 0.0
    return (n * sxy - sx * sy) / denom


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton registry
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_REGISTRY: Optional[BenchmarkRegistry] = None


def get_benchmark_registry() -> BenchmarkRegistry:
    """Get (or create) the global benchmark registry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = BenchmarkRegistry()
    return _GLOBAL_REGISTRY
