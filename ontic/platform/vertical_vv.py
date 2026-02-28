"""
Vertical Slice #3 — V&V Harness Demonstration (Phase 2 Exit Gate)
===================================================================

Exercises **every** Phase 2 V&V module on the 1-D heat equation, proving that
the harness is wired end-to-end and produces actionable pass/fail verdicts.

Modules exercised:
  1. MMS  — manufactured solution with forced source term
  2. Convergence  — grid and timestep refinement studies
  3. Conservation — mass integral monitor
  4. Stability    — CFL checker + blow-up detector
  5. Performance  — instrumented run with per-step timing
  6. Benchmarks   — registered golden-output benchmark

Success criteria:
  • MMS convergence study observes order ≥ 1.7 (formal 2).
  • Grid refinement study observes order ≥ 1.7.
  • Timestep refinement study observes order ≥ 0.7 (Forward Euler is O(1)).
  • Conservation: mass integral monotone decay, relative drift < 0.5 (diffusion).
  • Stability: CFL satisfied, no blow-up.
  • Performance: report generated with > 0 steps.
  • Benchmark: golden output comparison passes.

Usage::

    python3 -m ontic.platform.vertical_vv
"""

from __future__ import annotations

import math
import sys
from typing import Any, Callable, Dict, Tuple

import torch
from torch import Tensor

# Phase 1 modules
from ontic.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.solvers import ForwardEuler, RK4, RHSCallable, TimeIntegrator

# Phase 2 modules
from ontic.platform.vv.mms import (
    ManufacturedSolution,
    MMSProblem,
    mms_convergence_study,
)
from ontic.platform.vv.convergence import (
    RefinementStudy,
    grid_refinement_study,
    timestep_refinement_study,
)
from ontic.platform.vv.conservation import (
    ConservationMonitor,
    MassIntegral,
    LpNormQuantity,
)
from ontic.platform.vv.stability import (
    CFLChecker,
    BlowupDetector,
    StabilityReport,
    run_stability_checks,
)
from ontic.platform.vv.performance import PerformanceHarness
from ontic.platform.vv.benchmarks import (
    BenchmarkProblem,
    BenchmarkRegistry,
    GoldenOutput,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Heat equation infrastructure (reused from Phase 1 vertical_pde)
# ═══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.01
L = 1.0
PI = math.pi


def exact_heat(x: Tensor, t: float) -> Tensor:
    """Exact: u(x,t) = sin(πx/L) exp(−α(π/L)²t)."""
    return torch.sin(PI * x / L) * math.exp(-ALPHA * (PI / L) ** 2 * t)


def _laplacian_rhs(state: SimulationState, t: float) -> Dict[str, Tensor]:
    """du/dt = α ∇²u  with antisymmetric ghost-cell Dirichlet BCs."""
    u = state.get_field("u").data
    mesh = state.mesh
    assert isinstance(mesh, StructuredMesh)
    dx = mesh.dx[0]
    dx2 = dx ** 2

    d2u = torch.zeros_like(u)
    d2u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2
    d2u[0] = (u[1] - 2.0 * u[0] + (-u[0])) / dx2
    d2u[-1] = ((-u[-1]) - 2.0 * u[-1] + u[-2]) / dx2

    return {"u": ALPHA * d2u}


def _make_ic(mesh: StructuredMesh, field_name: str = "u") -> SimulationState:
    """IC: u(x,0) = sin(πx/L)."""
    x = mesh.cell_centers().squeeze(-1)
    u0 = torch.sin(PI * x / L)
    return SimulationState(
        t=0.0,
        fields={field_name: FieldData(name=field_name, data=u0, mesh=mesh)},
        mesh=mesh,
    )


def _solver_factory(mesh: StructuredMesh) -> Tuple[TimeIntegrator, RHSCallable, float]:
    """Standard solver setup: RK4 + CFL-safe dt."""
    dx = mesh.dx[0]
    dt = 0.4 * dx ** 2 / ALPHA
    return RK4(), _laplacian_rhs, dt


def _euler_factory(mesh: StructuredMesh) -> Tuple[TimeIntegrator, RHSCallable, float]:
    """Forward Euler for timestep-refinement studies."""
    dx = mesh.dx[0]
    dt = 0.4 * dx ** 2 / ALPHA
    return ForwardEuler(), _laplacian_rhs, dt


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MMS VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def _test_mms(verbose: bool = True) -> bool:
    """
    Manufactured solution: u = sin(2πx) exp(−t).

    Source: S = du/dt − α d²u/dx² = −sin(2πx)exp(−t) + α(2π)²sin(2πx)exp(−t)
            = (α(2π)² − 1) sin(2πx) exp(−t)
    """

    def exact(x: Tensor, t: float) -> Tensor:
        return torch.sin(2 * PI * x) * math.exp(-t)

    def source(x: Tensor, t: float) -> Tensor:
        coeff = ALPHA * (2 * PI) ** 2 - 1.0
        return coeff * torch.sin(2 * PI * x) * math.exp(-t)

    ms = ManufacturedSolution(
        exact=exact,
        source=source,
        name="heat_mms_sin2pi",
    )

    def mms_factory(
        mesh: StructuredMesh, ms_: ManufacturedSolution
    ) -> Tuple[Any, Callable, float]:
        dx = mesh.dx[0]
        dt = 0.4 * dx ** 2 / ALPHA
        base_rhs = _laplacian_rhs
        mms_prob = MMSProblem(ms_, mesh, base_rhs, field_name="u")
        return RK4(), mms_prob.rhs, dt

    result = mms_convergence_study(
        ms=ms,
        solver_factory=mms_factory,
        resolutions=[32, 64, 128, 256],
        t_final=0.1,
        formal_order=2,
        tolerance=0.3,
    )

    if verbose:
        print(result.summary())
        print()

    return result.passed


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONVERGENCE HARNESS
# ═══════════════════════════════════════════════════════════════════════════════


def _test_grid_convergence(verbose: bool = True) -> bool:
    """Grid refinement: vary N with RK4."""
    result = grid_refinement_study(
        exact_solution=exact_heat,
        ic_factory=_make_ic,
        solver_factory=_solver_factory,
        resolutions=[32, 64, 128, 256],
        t_final=0.5,
        formal_order=2,
        tolerance=0.3,
    )
    if verbose:
        print(result.summary())
        print()
    return result.passed


def _test_timestep_convergence(verbose: bool = True) -> bool:
    """
    Timestep refinement: vary dt with Forward Euler on a pure ODE.

    For the diffusion PDE, CFL constrains dt ~ dx^2, so temporal and spatial
    errors scale identically and cannot be separated.  Instead we use a scalar
    exponential-decay ODE:  du/dt = -u, u(0)=1, u(t) = exp(-t).

    Forward Euler is O(dt) in time.  By wrapping the ODE on a 1-cell mesh,
    we exercise the generic convergence harness with zero spatial error.
    """

    def ode_exact(x: Tensor, t: float) -> Tensor:
        return torch.full_like(x, math.exp(-t))

    def ode_ic(mesh: StructuredMesh) -> SimulationState:
        u0 = torch.ones(mesh.n_cells, dtype=torch.float64)
        return SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u0, mesh=mesh)},
            mesh=mesh,
        )

    def ode_factory(
        mesh: StructuredMesh,
    ) -> Tuple[TimeIntegrator, RHSCallable, float]:
        def rhs(state: SimulationState, t: float) -> Dict[str, Tensor]:
            return {"u": -state.get_field("u").data}
        return ForwardEuler(), rhs, 0.1  # default dt (overridden)

    result = timestep_refinement_study(
        exact_solution=ode_exact,
        ic_factory=ode_ic,
        solver_factory=ode_factory,
        N=1,
        dt_values=[0.1, 0.05, 0.025, 0.0125],
        t_final=1.0,
        formal_order=1,
        tolerance=0.3,
    )
    if verbose:
        print(result.summary())
        print()
    return result.passed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSERVATION MONITOR
# ═══════════════════════════════════════════════════════════════════════════════


def _test_conservation(verbose: bool = True) -> bool:
    """
    Mass integral should decay monotonically for the heat equation with
    homogeneous Dirichlet BCs (solution decays to zero).

    We check that the L2 norm is monotone-decaying (no spurious growth).
    """
    mesh = StructuredMesh(shape=(128,), domain=((0.0, L),))
    state = _make_ic(mesh)

    mass_mon = MassIntegral(field_name="u")
    l2_mon = LpNormQuantity(field_name="u", p=2)
    monitor = ConservationMonitor(
        quantities=[mass_mon, l2_mon],
        threshold=0.5,  # diffusion allows up to 50% decay, not growth
    )

    integrator = RK4()
    dt = 0.4 * mesh.dx[0] ** 2 / ALPHA
    monitor.record(state)

    t_final = 0.5
    while state.t < t_final - 1e-14:
        actual_dt = min(dt, t_final - state.t)
        state = integrator.step(state, _laplacian_rhs, actual_dt)
        monitor.record(state)

    reports = monitor.reports()

    # For diffusion: mass decays. Check monotone decay of L2 norm.
    l2_report = None
    for r in reports:
        if r.quantity_name.startswith("L2"):
            l2_report = r
            break

    if l2_report is None:
        if verbose:
            print("  Conservation: No L2 report found. FAIL")
        return False

    # Check monotone decay
    series = l2_report.time_series
    monotone = all(
        series[i] >= series[i + 1] - 1e-12
        for i in range(len(series) - 1)
    )

    if verbose:
        for r in reports:
            print(r.summary())
            print()
        print(f"  L2 norm monotone decay: {'PASS' if monotone else 'FAIL'}")
        print()

    return monotone


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STABILITY
# ═══════════════════════════════════════════════════════════════════════════════


def _test_stability(verbose: bool = True) -> bool:
    """CFL check + blow-up detector on a stable run."""
    mesh = StructuredMesh(shape=(128,), domain=((0.0, L),))
    state = _make_ic(mesh)

    dx = mesh.dx[0]
    dt = 0.4 * dx ** 2 / ALPHA  # CFL-safe

    cfl = CFLChecker(mode="diffusion", max_cfl=0.5, coeff=ALPHA)
    blowup = BlowupDetector(field_names=["u"])

    checks = [cfl, blowup]
    report = StabilityReport()

    integrator = RK4()
    t_final = 0.1
    step = 0
    while state.t < t_final - 1e-14:
        actual_dt = min(dt, t_final - state.t)
        report = run_stability_checks(state, actual_dt, checks, step=step, report=report)
        state = integrator.step(state, _laplacian_rhs, actual_dt)
        step += 1

    if verbose:
        print(report.summary())
        print()

    return report.passed


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERFORMANCE HARNESS
# ═══════════════════════════════════════════════════════════════════════════════


def _test_performance(verbose: bool = True) -> bool:
    """Instrumented run with per-step timing."""
    mesh = StructuredMesh(shape=(256,), domain=((0.0, L),))
    state = _make_ic(mesh)

    harness = PerformanceHarness(problem_name="heat_1d_vv")
    dx = mesh.dx[0]
    dt = 0.4 * dx ** 2 / ALPHA

    report = harness.run(
        integrator=RK4(),
        state=state,
        rhs=_laplacian_rhs,
        t_span=(0.0, 0.1),
        dt=dt,
    )

    if verbose:
        print(report.summary())
        print()

    return report.timing.total_steps > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BENCHMARK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


def _register_heat_benchmark(registry: BenchmarkRegistry) -> None:
    """Register the 1-D heat equation as a canonical benchmark."""

    def golden_exact(x: Tensor, t: float) -> Tensor:
        return exact_heat(x, t)

    golden_outputs = {}
    for N in (32, 64, 128, 256):
        golden_outputs[N] = GoldenOutput(
            resolution=N,
            t_final=0.5,
            linf_error_bound=_linf_bound(N),
            l2_error_bound=_l2_bound(N),
            exact_fn=golden_exact,
        )

    def setup_fn(
        mesh: StructuredMesh, golden: GoldenOutput
    ) -> Tuple[TimeIntegrator, SimulationState, RHSCallable, Tuple[float, float], float]:
        state0 = _make_ic(mesh)
        integrator, rhs, dt = _solver_factory(mesh)
        return integrator, state0, rhs, (0.0, golden.t_final), dt

    benchmark = BenchmarkProblem(
        name="heat_1d_dirichlet",
        description="1-D heat equation with Dirichlet BCs and sinusoidal IC",
        domain_pack="I",
        golden_outputs=golden_outputs,
        setup_fn=setup_fn,
        default_resolutions=(32, 64, 128, 256),
        convergence_order=2,
        source="Ontic Platform V&V Suite",
        tags=["parabolic", "diffusion", "1D", "analytical"],
    )
    registry.register(benchmark)


def _linf_bound(N: int) -> float:
    """Conservative L∞ error bound for the heat benchmark at resolution N."""
    # 2nd-order FVM: error ~ C * dx^2 = C / N^2
    # With safety margin
    return 5.0 / N ** 2


def _l2_bound(N: int) -> float:
    """Conservative L2 error bound."""
    return 2.0 / N ** 2


def _test_benchmark(verbose: bool = True) -> bool:
    """Register and run the heat benchmark."""
    registry = BenchmarkRegistry()
    _register_heat_benchmark(registry)

    result = registry.run("heat_1d_dirichlet")

    if verbose:
        print(result.summary())
        print()

    return result.passed


# ═══════════════════════════════════════════════════════════════════════════════
# Full V&V vertical slice
# ═══════════════════════════════════════════════════════════════════════════════


def run_vv_vertical_slice(
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Run all Phase 2 V&V modules on the 1-D heat equation.

    Returns a dict of ``{test_name: passed}`` for each module.
    """
    results: Dict[str, bool] = {}

    tests = [
        ("MMS convergence", _test_mms),
        ("Grid refinement", _test_grid_convergence),
        ("Timestep refinement", _test_timestep_convergence),
        ("Conservation monitor", _test_conservation),
        ("Stability checks", _test_stability),
        ("Performance harness", _test_performance),
        ("Benchmark registry", _test_benchmark),
    ]

    for name, fn in tests:
        if verbose:
            print("─" * 72)
            print(f"  {name}")
            print("─" * 72)
        try:
            passed = fn(verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  EXCEPTION: {exc}")
            passed = False
        results[name] = passed

    if verbose:
        print("=" * 72)
        print("  V&V VERTICAL SLICE — PHASE 2 EXIT GATE")
        print("=" * 72)
        all_pass = all(results.values())
        for name, ok in results.items():
            mark = "✓" if ok else "✗"
            print(f"  [{mark}] {name}")
        print()
        print(f"  RESULT: {'PHASE 2 VALIDATED' if all_pass else 'FAILED'}")
        print("=" * 72)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_vv_vertical_slice()
    sys.exit(0 if all(results.values()) else 1)
