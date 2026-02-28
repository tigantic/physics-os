"""
V&V Harness Tests — Phase 2
============================

Comprehensive tests for all six V&V modules:

- MMS
- Convergence
- Conservation
- Stability
- Performance
- Benchmarks

Plus integration tests for the vertical_vv slice.

Run:  pytest tests/test_vv.py -v
"""

from __future__ import annotations

import math
import tempfile
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytest
import torch
from torch import Tensor

from ontic.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.solvers import ForwardEuler, RK4, RHSCallable, TimeIntegrator


# ═══════════════════════════════════════════════════════════════════════════════
# Shared heat-equation fixtures
# ═══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.01
L = 1.0
PI = math.pi


def _exact_heat(x: Tensor, t: float) -> Tensor:
    return torch.sin(PI * x / L) * math.exp(-ALPHA * (PI / L) ** 2 * t)


def _make_ic(mesh: StructuredMesh) -> SimulationState:
    x = mesh.cell_centers().squeeze(-1)
    u0 = torch.sin(PI * x / L)
    return SimulationState(
        t=0.0,
        fields={"u": FieldData(name="u", data=u0, mesh=mesh)},
        mesh=mesh,
    )


def _heat_rhs(state: SimulationState, t: float) -> Dict[str, Tensor]:
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


def _solver_factory(mesh: StructuredMesh) -> Tuple[TimeIntegrator, RHSCallable, float]:
    dx = mesh.dx[0]
    dt = 0.4 * dx ** 2 / ALPHA
    return RK4(), _heat_rhs, dt


# ═══════════════════════════════════════════════════════════════════════════════
# MMS Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMMS:
    """Tests for ontic.platform.vv.mms."""

    def test_manufactured_solution_exact_field(self) -> None:
        from ontic.platform.vv.mms import ManufacturedSolution

        def exact(x: Tensor, t: float) -> Tensor:
            return torch.sin(x) * math.exp(-t)

        def source(x: Tensor, t: float) -> Tensor:
            return torch.zeros_like(x)

        ms = ManufacturedSolution(exact=exact, source=source, name="test")
        mesh = StructuredMesh(shape=(10,), domain=((0.0, PI),))
        field = ms.exact_field(mesh, t=0.5, field_name="u")
        assert field.name == "u"
        assert field.data.shape == (10,)
        x = mesh.cell_centers().squeeze(-1)
        expected = torch.sin(x) * math.exp(-0.5)
        assert torch.allclose(field.data, expected, atol=1e-12)

    def test_manufactured_solution_source_field(self) -> None:
        from ontic.platform.vv.mms import ManufacturedSolution

        def exact(x: Tensor, t: float) -> Tensor:
            return x * t

        def source(x: Tensor, t: float) -> Tensor:
            return x + t * torch.ones_like(x)

        ms = ManufacturedSolution(exact=exact, source=source)
        mesh = StructuredMesh(shape=(5,), domain=((0.0, 1.0),))
        src = ms.source_field(mesh, t=1.0)
        assert src.data.shape == (5,)

    def test_mms_problem_adds_source(self) -> None:
        from ontic.platform.vv.mms import ManufacturedSolution, MMSProblem

        def exact(x: Tensor, t: float) -> Tensor:
            return torch.ones_like(x)

        def source(x: Tensor, t: float) -> Tensor:
            return 2.0 * torch.ones_like(x)

        ms = ManufacturedSolution(exact=exact, source=source)
        mesh = StructuredMesh(shape=(4,), domain=((0.0, 1.0),))

        def base_rhs(state: SimulationState, t: float) -> Dict[str, Tensor]:
            return {"u": torch.ones(4, dtype=torch.float64)}

        mms_prob = MMSProblem(ms, mesh, base_rhs, field_name="u")
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=torch.ones(4, dtype=torch.float64), mesh=mesh)},
            mesh=mesh,
        )
        result = mms_prob.rhs(state, 0.0)
        # base returns 1.0, source is 2.0, total should be 3.0
        assert torch.allclose(result["u"], torch.full((4,), 3.0, dtype=torch.float64), atol=1e-12)

    def test_mms_convergence_study_heat(self) -> None:
        from ontic.platform.vv.mms import (
            ManufacturedSolution,
            MMSProblem,
            mms_convergence_study,
        )

        def exact(x: Tensor, t: float) -> Tensor:
            return torch.sin(2 * PI * x) * math.exp(-t)

        def source(x: Tensor, t: float) -> Tensor:
            coeff = ALPHA * (2 * PI) ** 2 - 1.0
            return coeff * torch.sin(2 * PI * x) * math.exp(-t)

        ms = ManufacturedSolution(exact=exact, source=source, name="heat_mms")

        def factory(mesh: StructuredMesh, ms_: ManufacturedSolution) -> Tuple:
            dx = mesh.dx[0]
            dt = 0.4 * dx ** 2 / ALPHA
            mms_prob = MMSProblem(ms_, mesh, _heat_rhs, field_name="u")
            return RK4(), mms_prob.rhs, dt

        result = mms_convergence_study(
            ms=ms,
            solver_factory=factory,
            resolutions=[32, 64, 128],
            t_final=0.1,
            formal_order=2,
            tolerance=0.3,
        )
        assert result.passed
        assert result.observed_order_linf >= 1.7
        assert len(result.points) == 3

    def test_mms_convergence_summary(self) -> None:
        from ontic.platform.vv.mms import ConvergencePoint, MMSConvergenceResult

        pts = [
            ConvergencePoint(N=32, dx=0.03125, dt=1e-4, linf_error=1e-4,
                             l1_error=1e-4, l2_error=1e-4, steps=100),
            ConvergencePoint(N=64, dx=0.015625, dt=2.5e-5, linf_error=2.5e-5,
                             l1_error=2.5e-5, l2_error=2.5e-5, steps=400),
        ]
        result = MMSConvergenceResult(
            points=pts, observed_order_linf=2.0, observed_order_l2=2.0,
            formal_order=2, passed=True, tolerance=0.3,
        )
        s = result.summary()
        assert "PASS" in s
        assert "MMS Convergence Study" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvergence:
    """Tests for ontic.platform.vv.convergence."""

    def test_grid_refinement_passes(self) -> None:
        from ontic.platform.vv.convergence import grid_refinement_study

        result = grid_refinement_study(
            exact_solution=_exact_heat,
            ic_factory=_make_ic,
            solver_factory=_solver_factory,
            resolutions=[32, 64, 128],
            t_final=0.5,
            formal_order=2,
            tolerance=0.3,
        )
        assert result.passed
        assert result.study_type == "grid"
        assert result.observed_order_linf >= 1.7

    def test_grid_refinement_summary(self) -> None:
        from ontic.platform.vv.convergence import grid_refinement_study

        result = grid_refinement_study(
            exact_solution=_exact_heat,
            ic_factory=_make_ic,
            solver_factory=_solver_factory,
            resolutions=[32, 64],
            t_final=0.5,
        )
        s = result.summary()
        assert "Grid Refinement Study" in s
        assert "Observed order" in s

    def test_timestep_refinement_ode(self) -> None:
        from ontic.platform.vv.convergence import timestep_refinement_study

        def ode_exact(x: Tensor, t: float) -> Tensor:
            return torch.full_like(x, math.exp(-t))

        def ode_ic(mesh: StructuredMesh) -> SimulationState:
            u0 = torch.ones(mesh.n_cells, dtype=torch.float64)
            return SimulationState(
                t=0.0,
                fields={"u": FieldData(name="u", data=u0, mesh=mesh)},
                mesh=mesh,
            )

        def ode_factory(mesh: StructuredMesh) -> Tuple:
            def rhs(state: SimulationState, t: float) -> Dict[str, Tensor]:
                return {"u": -state.get_field("u").data}
            return ForwardEuler(), rhs, 0.1

        result = timestep_refinement_study(
            exact_solution=ode_exact,
            ic_factory=ode_ic,
            solver_factory=ode_factory,
            N=1,
            dt_values=[0.1, 0.05, 0.025],
            t_final=1.0,
            formal_order=1,
            tolerance=0.3,
        )
        assert result.passed
        assert result.study_type == "timestep"
        assert result.observed_order_linf >= 0.7

    def test_compute_order(self) -> None:
        from ontic.platform.vv.convergence import compute_order

        # Perfect 2nd order: error = C * h^2
        dxs = [0.1, 0.05, 0.025]
        errors = [0.01, 0.0025, 0.000625]
        order = compute_order(dxs, errors)
        assert abs(order - 2.0) < 0.01

    def test_refinement_study_class(self) -> None:
        from ontic.platform.vv.convergence import RefinementStudy

        study = RefinementStudy(
            exact_solution=_exact_heat,
            ic_factory=_make_ic,
            solver_factory=_solver_factory,
        )
        result = study.run_grid_study(
            resolutions=[32, 64],
            t_final=0.5,
        )
        assert result.passed


# ═══════════════════════════════════════════════════════════════════════════════
# Conservation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConservation:
    """Tests for ontic.platform.vv.conservation."""

    def test_mass_integral(self) -> None:
        from ontic.platform.vv.conservation import MassIntegral

        mesh = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        u = torch.ones(10, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        mi = MassIntegral(field_name="u")
        val = mi.compute(state)
        # ∫ 1 dx from 0 to 1 = 1.0
        assert abs(val.item() - 1.0) < 1e-12

    def test_energy_integral(self) -> None:
        from ontic.platform.vv.conservation import EnergyIntegral

        mesh = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        u = 2.0 * torch.ones(10, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )

        def ke_density(s: SimulationState) -> Tensor:
            return 0.5 * s.get_field("u").data ** 2

        ei = EnergyIntegral(ke_density, label="KE", units="J")
        val = ei.compute(state)
        # ∫ 0.5 * 4 dx = 2.0
        assert abs(val.item() - 2.0) < 1e-12

    def test_lp_norm_quantity(self) -> None:
        from ontic.platform.vv.conservation import LpNormQuantity

        mesh = StructuredMesh(shape=(100,), domain=((0.0, 1.0),))
        u = torch.ones(100, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        lp = LpNormQuantity("u", p=2)
        val = lp.compute(state)
        # (∫ 1^2 dx)^{1/2} = 1.0
        assert abs(val.item() - 1.0) < 1e-8

    def test_conservation_monitor_drift(self) -> None:
        from ontic.platform.vv.conservation import (
            ConservationMonitor,
            MassIntegral,
        )

        mesh = StructuredMesh(shape=(64,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)
        monitor = ConservationMonitor(
            quantities=[MassIntegral("u")],
            threshold=0.5,
        )
        monitor.record(state)

        integrator = RK4()
        dt = 0.4 * mesh.dx[0] ** 2 / ALPHA
        for _ in range(50):
            state = integrator.step(state, _heat_rhs, dt)
            monitor.record(state)

        reports = monitor.reports()
        assert len(reports) == 1
        assert reports[0].quantity_name == "mass(u)"
        # For diffusion with Dirichlet BCs, mass decays but drift is < 50%
        assert reports[0].passed

    def test_conservation_monitor_monotone_l2(self) -> None:
        from ontic.platform.vv.conservation import (
            ConservationMonitor,
            LpNormQuantity,
        )

        mesh = StructuredMesh(shape=(64,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)
        monitor = ConservationMonitor(
            quantities=[LpNormQuantity("u", p=2)],
            threshold=1.0,
        )
        monitor.record(state)

        integrator = RK4()
        dt = 0.4 * mesh.dx[0] ** 2 / ALPHA
        for _ in range(50):
            state = integrator.step(state, _heat_rhs, dt)
            monitor.record(state)

        report = monitor.report("L2(u)")
        # L2 norm should decay monotonically for heat equation
        series = report.time_series
        for i in range(len(series) - 1):
            assert series[i] >= series[i + 1] - 1e-12

    def test_conservation_record_from_history(self) -> None:
        from ontic.platform.vv.conservation import (
            ConservationMonitor,
            MassIntegral,
        )

        monitor = ConservationMonitor(
            quantities=[MassIntegral("u")],
            threshold=0.01,
        )
        monitor.record_from_history("mass(u)", [1.0, 1.001, 0.999, 1.0005])
        report = monitor.report("mass(u)")
        assert report.initial_value == 1.0
        assert report.max_absolute_drift == pytest.approx(0.001, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# Stability Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStability:
    """Tests for ontic.platform.vv.stability."""

    def test_cfl_advection_pass(self) -> None:
        from ontic.platform.vv.stability import CFLChecker

        mesh = StructuredMesh(shape=(100,), domain=((0.0, 1.0),))
        u = torch.ones(100, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        cfl = CFLChecker(field_name="u", mode="advection", max_cfl=1.0)
        dt = 0.005  # CFL = 1.0 * 0.005 / 0.01 = 0.5 < 1.0
        verdict = cfl.check(state, dt)
        assert verdict.passed
        assert verdict.metric_value == pytest.approx(0.5, abs=0.01)

    def test_cfl_advection_fail(self) -> None:
        from ontic.platform.vv.stability import CFLChecker

        mesh = StructuredMesh(shape=(100,), domain=((0.0, 1.0),))
        u = 10.0 * torch.ones(100, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        cfl = CFLChecker(field_name="u", mode="advection", max_cfl=1.0)
        dt = 0.005  # CFL = 10 * 0.005 / 0.01 = 5.0 > 1.0
        verdict = cfl.check(state, dt)
        assert not verdict.passed

    def test_cfl_diffusion(self) -> None:
        from ontic.platform.vv.stability import CFLChecker

        mesh = StructuredMesh(shape=(100,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)
        cfl = CFLChecker(mode="diffusion", max_cfl=0.5, coeff=ALPHA)
        dx = mesh.dx[0]
        dt = 0.4 * dx ** 2 / ALPHA  # CFL = 0.01 * 0.4*dx^2/0.01 / dx^2 = 0.4
        verdict = cfl.check(state, dt)
        assert verdict.passed
        assert verdict.metric_value == pytest.approx(0.4, abs=0.01)

    def test_cfl_max_stable_dt(self) -> None:
        from ontic.platform.vv.stability import CFLChecker

        mesh = StructuredMesh(shape=(100,), domain=((0.0, 1.0),))
        u = 2.0 * torch.ones(100, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        cfl = CFLChecker(field_name="u", mode="advection", max_cfl=1.0)
        dt_max = cfl.max_stable_dt(state, safety=0.9)
        # dt_max = 0.9 * 1.0 * 0.01 / 2.0 = 0.0045
        assert dt_max == pytest.approx(0.0045, abs=1e-6)

    def test_blowup_detector_clean(self) -> None:
        from ontic.platform.vv.stability import BlowupDetector

        mesh = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        u = torch.ones(10, dtype=torch.float64)
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        det = BlowupDetector(field_names=["u"])
        verdict = det.check(state, dt=0.01)
        assert verdict.passed

    def test_blowup_detector_nan(self) -> None:
        from ontic.platform.vv.stability import BlowupDetector

        mesh = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        u = torch.ones(10, dtype=torch.float64)
        u[5] = float("nan")
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        det = BlowupDetector(field_names=["u"])
        verdict = det.check(state, dt=0.01)
        assert not verdict.passed
        assert "NaN" in verdict.message

    def test_blowup_detector_inf(self) -> None:
        from ontic.platform.vv.stability import BlowupDetector

        mesh = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        u = torch.ones(10, dtype=torch.float64)
        u[3] = float("inf")
        state = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u, mesh=mesh)},
            mesh=mesh,
        )
        det = BlowupDetector(field_names=["u"])
        verdict = det.check(state, dt=0.01)
        assert not verdict.passed
        assert "Inf" in verdict.message

    def test_blowup_detector_growth(self) -> None:
        from ontic.platform.vv.stability import BlowupDetector

        mesh = StructuredMesh(shape=(10,), domain=((0.0, 1.0),))
        det = BlowupDetector(field_names=["u"], max_growth_rate=2.0)

        # First call: establishes baseline
        u1 = torch.ones(10, dtype=torch.float64)
        s1 = SimulationState(
            t=0.0,
            fields={"u": FieldData(name="u", data=u1, mesh=mesh)},
            mesh=mesh,
        )
        det.check(s1, dt=0.01)

        # Second call: 10x growth
        u2 = 10.0 * torch.ones(10, dtype=torch.float64)
        s2 = SimulationState(
            t=0.01,
            fields={"u": FieldData(name="u", data=u2, mesh=mesh)},
            mesh=mesh,
        )
        verdict = det.check(s2, dt=0.01)
        assert not verdict.passed

    def test_stability_report_aggregate(self) -> None:
        from ontic.platform.vv.stability import (
            StabilityReport,
            StabilityVerdict,
        )

        report = StabilityReport()
        report.add_verdict(
            StabilityVerdict("CFL", True, 0.3, 1.0), step=0
        )
        report.add_verdict(
            StabilityVerdict("CFL", False, 1.5, 1.0, "over limit"), step=5
        )
        assert not report.passed
        assert report.n_violations == 1
        assert report.first_violation_step == 5

    def test_stiffness_estimator(self) -> None:
        from ontic.platform.vv.stability import StiffnessEstimator

        mesh = StructuredMesh(shape=(32,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)

        est = StiffnessEstimator(
            rhs=_heat_rhs,
            field_name="u",
            n_iterations=10,
            stiffness_threshold=100.0,
        )
        dx = mesh.dx[0]
        dt = 0.4 * dx ** 2 / ALPHA
        verdict = est.check(state, dt)
        # At CFL-safe dt, stiffness_number should be moderate
        assert verdict.metric_value > 0

    def test_run_stability_checks(self) -> None:
        from ontic.platform.vv.stability import (
            CFLChecker,
            BlowupDetector,
            run_stability_checks,
        )

        mesh = StructuredMesh(shape=(64,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)
        dx = mesh.dx[0]
        dt = 0.4 * dx ** 2 / ALPHA

        report = run_stability_checks(
            state, dt,
            checks=[
                CFLChecker(mode="diffusion", max_cfl=0.5, coeff=ALPHA),
                BlowupDetector(field_names=["u"]),
            ],
            step=0,
        )
        assert report.passed


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """Tests for ontic.platform.vv.performance."""

    def test_harness_basic(self) -> None:
        from ontic.platform.vv.performance import PerformanceHarness

        mesh = StructuredMesh(shape=(32,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)
        dx = mesh.dx[0]
        dt = 0.4 * dx ** 2 / ALPHA

        harness = PerformanceHarness(problem_name="test")
        report = harness.run(
            integrator=RK4(),
            state=state,
            rhs=_heat_rhs,
            t_span=(0.0, 0.1),
            dt=dt,
        )
        assert report.timing.total_steps > 0
        assert report.timing.total_wall_s > 0
        assert report.timing.mean_step_s > 0
        assert report.n_cells == 32
        assert report.problem_name == "test"

    def test_harness_summary(self) -> None:
        from ontic.platform.vv.performance import PerformanceHarness

        mesh = StructuredMesh(shape=(16,), domain=((0.0, 1.0),))
        state = _make_ic(mesh)
        dx = mesh.dx[0]
        dt = 0.4 * dx ** 2 / ALPHA

        harness = PerformanceHarness(problem_name="summary_test")
        report = harness.run(
            integrator=ForwardEuler(),
            state=state,
            rhs=_heat_rhs,
            t_span=(0.0, 0.01),
            dt=dt,
        )
        s = report.summary()
        assert "summary_test" in s
        assert "steps/s" in s.lower() or "Throughput" in s

    def test_scaling_study_strong(self) -> None:
        from ontic.platform.vv.performance import ScalingStudy

        def factory(N: int) -> Tuple:
            mesh = StructuredMesh(shape=(N,), domain=((0.0, 1.0),))
            state = _make_ic(mesh)
            dx = mesh.dx[0]
            dt = 0.4 * dx ** 2 / ALPHA
            return RK4(), state, _heat_rhs, (0.0, 0.01), dt

        study = ScalingStudy(factory)
        result = study.run_strong([16, 32], n_repeats=1)
        assert result.study_type == "strong"
        assert len(result.sizes) == 2
        assert all(t > 0 for t in result.wall_times)

    def test_memory_snapshot(self) -> None:
        from ontic.platform.vv.performance import _get_memory_snapshot

        snap = _get_memory_snapshot()
        # On Linux, RSS should be > 0
        assert snap.peak_rss_bytes >= 0
        assert snap.current_rss_bytes >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarks:
    """Tests for ontic.platform.vv.benchmarks."""

    def test_golden_output_exact(self) -> None:
        from ontic.platform.vv.benchmarks import GoldenOutput

        golden = GoldenOutput(
            resolution=32,
            t_final=0.5,
            linf_error_bound=1e-3,
            l2_error_bound=1e-3,
            exact_fn=_exact_heat,
        )
        x = torch.linspace(0.05, 0.95, 32, dtype=torch.float64)
        ref = golden.reference(x, 0.5)
        assert ref.shape == (32,)

    def test_golden_output_stored_data(self) -> None:
        from ontic.platform.vv.benchmarks import GoldenOutput

        data = torch.randn(10, dtype=torch.float64)
        golden = GoldenOutput(
            resolution=10,
            t_final=1.0,
            linf_error_bound=0.1,
            l2_error_bound=0.1,
            stored_data=data,
        )
        assert golden.data_hash is not None
        assert golden.verify_hash()

    def test_benchmark_registry_register_and_get(self) -> None:
        from ontic.platform.vv.benchmarks import (
            BenchmarkProblem,
            BenchmarkRegistry,
            GoldenOutput,
        )

        registry = BenchmarkRegistry()

        def dummy_setup(mesh, golden):
            state = _make_ic(mesh)
            integ, rhs, dt = _solver_factory(mesh)
            return integ, state, rhs, (0.0, golden.t_final), dt

        bp = BenchmarkProblem(
            name="test_bench",
            description="Test",
            domain_pack="I",
            golden_outputs={
                32: GoldenOutput(
                    resolution=32, t_final=0.5,
                    linf_error_bound=1e-2, l2_error_bound=1e-2,
                    exact_fn=_exact_heat,
                ),
            },
            setup_fn=dummy_setup,
            tags=["test"],
        )
        registry.register(bp)
        assert "test_bench" in registry.list_benchmarks()
        assert registry.get("test_bench").name == "test_bench"
        assert "test_bench" in registry.list_by_tag("test")
        assert "test_bench" in registry.list_by_domain("I")

    def test_benchmark_registry_run(self) -> None:
        from ontic.platform.vv.benchmarks import (
            BenchmarkProblem,
            BenchmarkRegistry,
            GoldenOutput,
        )

        registry = BenchmarkRegistry()

        def setup(mesh, golden):
            state = _make_ic(mesh)
            integ, rhs, dt = _solver_factory(mesh)
            return integ, state, rhs, (0.0, golden.t_final), dt

        golden_outputs = {
            N: GoldenOutput(
                resolution=N, t_final=0.5,
                linf_error_bound=5.0 / N ** 2,
                l2_error_bound=2.0 / N ** 2,
                exact_fn=_exact_heat,
            )
            for N in [32, 64, 128]
        }
        bp = BenchmarkProblem(
            name="heat_run_test",
            description="Heat benchmark for test",
            domain_pack="I",
            golden_outputs=golden_outputs,
            setup_fn=setup,
            default_resolutions=(32, 64, 128),
            convergence_order=2,
        )
        registry.register(bp)
        result = registry.run("heat_run_test")
        assert result.passed
        assert result.observed_order_linf >= 1.7

    def test_benchmark_duplicate_registration(self) -> None:
        from ontic.platform.vv.benchmarks import (
            BenchmarkProblem,
            BenchmarkRegistry,
            GoldenOutput,
        )

        registry = BenchmarkRegistry()
        bp = BenchmarkProblem(
            name="dup",
            description="Dup",
            domain_pack="I",
            golden_outputs={
                32: GoldenOutput(
                    resolution=32, t_final=0.5,
                    linf_error_bound=1.0, l2_error_bound=1.0,
                    exact_fn=_exact_heat,
                ),
            },
            setup_fn=lambda m, g: (_solver_factory(m)[0], _make_ic(m), *_solver_factory(m)[1:]),
        )
        registry.register(bp)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(bp)

    def test_benchmark_not_found(self) -> None:
        from ontic.platform.vv.benchmarks import BenchmarkRegistry

        registry = BenchmarkRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_global_registry_singleton(self) -> None:
        from ontic.platform.vv.benchmarks import get_benchmark_registry

        r1 = get_benchmark_registry()
        r2 = get_benchmark_registry()
        assert r1 is r2


# ═══════════════════════════════════════════════════════════════════════════════
# V&V __init__ import test
# ═══════════════════════════════════════════════════════════════════════════════


class TestVVImports:
    """Verify that the vv package exports everything correctly."""

    def test_import_all(self) -> None:
        from ontic.platform import vv

        # MMS
        assert hasattr(vv, "ManufacturedSolution")
        assert hasattr(vv, "MMSProblem")
        assert hasattr(vv, "mms_convergence_study")
        # Convergence
        assert hasattr(vv, "RefinementStudy")
        assert hasattr(vv, "grid_refinement_study")
        assert hasattr(vv, "timestep_refinement_study")
        assert hasattr(vv, "compute_order")
        # Conservation
        assert hasattr(vv, "ConservationMonitor")
        assert hasattr(vv, "MassIntegral")
        assert hasattr(vv, "EnergyIntegral")
        # Stability
        assert hasattr(vv, "CFLChecker")
        assert hasattr(vv, "BlowupDetector")
        assert hasattr(vv, "StiffnessEstimator")
        assert hasattr(vv, "StabilityReport")
        # Performance
        assert hasattr(vv, "PerformanceHarness")
        assert hasattr(vv, "ScalingStudy")
        # Benchmarks
        assert hasattr(vv, "BenchmarkProblem")
        assert hasattr(vv, "BenchmarkRegistry")
        assert hasattr(vv, "GoldenOutput")
        assert hasattr(vv, "get_benchmark_registry")

    def test_version(self) -> None:
        from ontic.platform import vv

        assert vv.__version__ == "0.1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Vertical VV Slice
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestVerticalVV:
    """Integration test exercising the full Phase 2 V&V vertical slice."""

    def test_run_vv_vertical_slice(self) -> None:
        from ontic.platform.vertical_vv import run_vv_vertical_slice

        results = run_vv_vertical_slice(verbose=False)
        assert all(results.values()), f"Failed checks: {[k for k, v in results.items() if not v]}"
