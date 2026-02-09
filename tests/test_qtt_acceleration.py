"""
Test suite for Phase 5 — QTT / TN acceleration modules.

Tests:
  • QTT bridging: field_to_qtt, qtt_to_field round-trip.
  • TCI decomposition: tci_from_function, tci_from_field, error-vs-rank.
  • Acceleration policy: mode selection, enablement validation.
  • QTT solver wrapper: step, solve, rank growth report.
  • QTT-accelerated domain solvers: Burgers, AdvDiff, Maxwell, Vlasov.

Runs via:  pytest tests/test_qtt_acceleration.py -v
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    Mesh,
    SimulationState,
    StructuredMesh,
)
from tensornet.platform.protocols import Observable, SolveResult


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mesh_128():
    """1-D mesh with 128 cells (power-of-2 for QTT)."""
    return StructuredMesh(shape=(128,), domain=((0.0, 2.0 * math.pi),))


@pytest.fixture
def mesh_256():
    """1-D mesh with 256 cells (power-of-2 for QTT)."""
    return StructuredMesh(shape=(256,), domain=((0.0, 2.0 * math.pi),))


@pytest.fixture
def smooth_field(mesh_128):
    """Smooth sinusoidal field on 128-pt grid (highly QTT-compressible)."""
    x = torch.linspace(0.0, 2.0 * math.pi, 128, dtype=torch.float64)
    return FieldData(name="u", data=torch.sin(x), mesh=mesh_128)


@pytest.fixture
def smooth_state(mesh_128, smooth_field):
    """SimulationState with a smooth sinusoidal 'u' field."""
    return SimulationState(t=0.0, fields={"u": smooth_field}, mesh=mesh_128)


@pytest.fixture
def smooth_state_256(mesh_256):
    """SimulationState on 256-pt mesh."""
    x = torch.linspace(0.0, 2.0 * math.pi, 256, dtype=torch.float64)
    u = FieldData(name="u", data=torch.sin(x), mesh=mesh_256)
    return SimulationState(t=0.0, fields={"u": u}, mesh=mesh_256)


# ── Tiny test solver implementing Solver protocol ────────────────────────────


class _LinearDecaySolver:
    """
    Minimal solver: u(t+dt) = u(t) * (1 - dt).
    Implements the Solver protocol for QTTAcceleratedSolver wrapping.
    """

    @property
    def name(self) -> str:
        return "LinearDecay"

    def step(self, state: SimulationState, dt: float, **kwargs: Any) -> SimulationState:
        new_fields = {}
        for fname, fd in state.fields.items():
            new_data = fd.data * (1.0 - dt)
            new_fields[fname] = FieldData(
                name=fd.name, data=new_data, mesh=fd.mesh,
                components=fd.components, units=fd.units,
            )
        return SimulationState(
            t=state.t + dt,
            fields=new_fields,
            mesh=state.mesh,
            metadata=state.metadata,
            step_index=state.step_index + 1,
        )

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
        t0, tf = t_span
        n_steps = max(1, int(round((tf - t0) / dt)))
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)
        current = state
        for _ in range(n_steps):
            current = self.step(current, dt)
        return SolveResult(
            final_state=current,
            t_final=current.t,
            steps_taken=n_steps,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Bridging (tensornet.platform.qtt)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQTTBridging:
    """Tests for field_to_qtt, qtt_to_field, QTTOperator, QTTDiscretization."""

    def test_field_to_qtt_produces_cores(self, smooth_field):
        from tensornet.platform.qtt import field_to_qtt

        qtt = field_to_qtt(smooth_field, max_rank=32, tolerance=1e-10)
        assert qtt.name == "u"
        assert qtt.n_qubits == 7  # 2^7 = 128
        assert len(qtt.cores) == 7
        assert qtt.max_rank >= 1
        assert qtt.compression_ratio >= 1.0

    def test_qtt_round_trip_error(self, smooth_field):
        from tensornet.platform.qtt import field_to_qtt, qtt_to_field

        qtt = field_to_qtt(smooth_field, max_rank=64, tolerance=1e-10)
        reconstructed = qtt_to_field(qtt, n_points=128)
        error = (reconstructed.data - smooth_field.data).abs().max().item()
        assert error < 1e-8, f"Round-trip error too large: {error}"

    def test_qtt_roundtrip_error_utility(self, smooth_field):
        from tensornet.platform.qtt import qtt_roundtrip_error

        err = qtt_roundtrip_error(smooth_field, max_rank=64, tolerance=1e-10)
        assert err < 1e-8

    def test_qtt_field_properties(self, smooth_field):
        from tensornet.platform.qtt import field_to_qtt

        qtt = field_to_qtt(smooth_field, max_rank=32)
        assert qtt.grid_size == 128
        assert len(qtt.ranks) == 6  # n_qubits - 1 (interior bond dims)
        assert qtt.storage_elements > 0
        assert qtt.dense_elements == 128
        # Smooth signal should compress well
        assert qtt.compression_ratio > 1.0

    def test_qtt_field_clone(self, smooth_field):
        from tensornet.platform.qtt import field_to_qtt

        qtt = field_to_qtt(smooth_field, max_rank=32)
        cloned = qtt.clone()
        assert cloned.name == qtt.name
        assert cloned.n_qubits == qtt.n_qubits
        assert len(cloned.cores) == len(qtt.cores)
        # Mutating clone should not affect original
        cloned.cores[0] *= 0.0
        assert qtt.cores[0].abs().sum() > 0

    def test_qtt_operator_apply(self, smooth_field):
        from tensornet.platform.qtt import QTTOperator, field_to_qtt

        # Build trivial identity-like MPO: each core is (r_in, 2, 2, r_out)
        n_qubits = 7
        mpo_cores = []
        for k in range(n_qubits):
            r_in = 1 if k == 0 else 1
            r_out = 1 if k == n_qubits - 1 else 1
            core = torch.zeros(r_in, 2, 2, r_out, dtype=torch.float64)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            mpo_cores.append(core)

        op = QTTOperator(name="identity", mpo_cores=mpo_cores)
        assert op.name == "identity"

        # Apply to dense field
        result = op.apply(smooth_field.data)
        assert result.shape == smooth_field.data.shape

    def test_qtt_discretization(self):
        from tensornet.platform.qtt import QTTDiscretization

        disc = QTTDiscretization(max_rank=32, tolerance=1e-8, order=2)
        assert disc.method == "QTT"
        assert disc.order == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TCI Decomposition (tensornet.platform.tci)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTCI:
    """Tests for TCI decomposition engine."""

    def test_tci_from_function_basic(self):
        from tensornet.platform.tci import TCIConfig, tci_from_function

        def smooth_fn(x: Tensor) -> Tensor:
            return torch.sin(2.0 * math.pi * x)

        config = TCIConfig(max_rank=16, tolerance=1e-6, max_sweeps=10)
        result = tci_from_function(smooth_fn, n_qubits=7, config=config)

        assert result.n_qubits == 7
        assert len(result.cores) == 7
        assert result.max_rank_achieved >= 1
        assert result.compression_ratio > 0  # May be < 1 for small grids + high rank
        assert result.n_function_evals > 0

    def test_tci_from_field(self, smooth_field):
        from tensornet.platform.tci import TCIConfig, tci_from_field

        config = TCIConfig(max_rank=32, tolerance=1e-8)
        qtt = tci_from_field(smooth_field, config=config)
        assert qtt.n_qubits == 7
        assert len(qtt.cores) == 7

    def test_tci_error_vs_rank(self, smooth_field):
        from tensornet.platform.tci import tci_error_vs_rank

        points = tci_error_vs_rank(smooth_field, rank_schedule=[2, 4, 8, 16])
        assert len(points) == 4
        # Error should decrease with increasing rank
        for i in range(1, len(points)):
            assert points[i].rank >= points[i - 1].rank
        # Higher rank should give lower error (for smooth signals)
        assert points[-1].error_linf <= points[0].error_linf + 1e-14


# ═══════════════════════════════════════════════════════════════════════════════
# Acceleration Policy (tensornet.platform.acceleration)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAccelerationPolicy:
    """Tests for acceleration policy and metrics."""

    def test_default_policy_warmup_qtt(self):
        from tensornet.platform.acceleration import AccelerationMode, AccelerationPolicy

        policy = AccelerationPolicy(warmup_steps=5)
        # During warmup, policy runs QTT to establish baselines
        mode = policy.should_use_qtt(step_index=2)
        assert mode == AccelerationMode.QTT

    def test_policy_after_warmup_qtt(self):
        from tensornet.platform.acceleration import AccelerationMode, AccelerationPolicy

        policy = AccelerationPolicy(warmup_steps=3)
        mode = policy.should_use_qtt(step_index=5)
        assert mode == AccelerationMode.QTT

    def test_policy_rank_explosion_fallback(self):
        from tensornet.platform.acceleration import (
            AccelerationMetrics,
            AccelerationMode,
            AccelerationPolicy,
        )

        policy = AccelerationPolicy(
            max_allowed_rank=32, warmup_steps=0, error_budget=1e-4
        )
        # Simulate rank explosion
        prev = AccelerationMetrics(
            step_index=0, t=0.0, mode=AccelerationMode.QTT,
            max_rank=10, mean_rank=8.0, compression_ratio=5.0,
        )
        curr = AccelerationMetrics(
            step_index=1, t=0.01, mode=AccelerationMode.QTT,
            max_rank=64, mean_rank=50.0, compression_ratio=1.2,
            error_vs_baseline=0.5,
        )
        mode = policy.should_use_qtt(step_index=1, current_metrics=curr, previous_metrics=prev)
        assert mode == AccelerationMode.FALLBACK

    def test_rank_growth_report(self):
        from tensornet.platform.acceleration import (
            AccelerationMetrics,
            AccelerationMode,
            RankGrowthReport,
        )

        metrics = [
            AccelerationMetrics(
                step_index=i, t=i * 0.01, mode=AccelerationMode.QTT,
                max_rank=i + 2, mean_rank=float(i + 1),
                compression_ratio=10.0 / (i + 1), qtt_time_s=0.001,
            )
            for i in range(10)
        ]
        report = RankGrowthReport.from_metrics(
            metrics, solver_name="test", problem_name="test_problem"
        )
        assert report.n_steps == 10
        assert report.peak_rank == 11  # max(i + 2) for i=9
        assert report.n_qtt_steps == 10
        assert report.n_fallback_steps == 0
        assert len(report.summary()) > 0

    def test_validate_enablement(self):
        from tensornet.platform.acceleration import (
            AccelerationMetrics,
            AccelerationMode,
            AccelerationPolicy,
            RankGrowthReport,
        )

        policy = AccelerationPolicy(max_allowed_rank=16, error_budget=1e-4)
        report = RankGrowthReport(
            solver_name="test",
            problem_name="test",
            n_steps=10,
            n_qtt_steps=10,
            n_fallback_steps=0,
            peak_rank=8,
            max_error=1e-6,
            domain_of_validity="test domain",
            overall_speedup=1.5,
            error_per_step=[1e-6] * 10,
        )
        result = policy.validate_enablement(report)
        assert result["all_passed"]
        assert result["peak_rank_acceptable"]
        assert result["max_error_acceptable"]


# ═══════════════════════════════════════════════════════════════════════════════
# QTT Solver Wrapper (tensornet.platform.qtt_solver)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQTTSolver:
    """Tests for QTTAcceleratedSolver wrapper."""

    def test_qtt_solver_step(self, smooth_state):
        from tensornet.platform.acceleration import AccelerationPolicy
        from tensornet.platform.qtt_solver import QTTAcceleratedSolver

        baseline = _LinearDecaySolver()
        solver = QTTAcceleratedSolver(
            baseline_solver=baseline,
            policy=AccelerationPolicy(warmup_steps=0, max_allowed_rank=64),
            max_rank=32,
        )
        assert "QTT" in solver.name

        new_state = solver.step(smooth_state, dt=0.01)
        assert new_state.t > smooth_state.t
        assert "u" in new_state.fields

    def test_qtt_solver_solve(self, smooth_state):
        from tensornet.platform.acceleration import AccelerationPolicy
        from tensornet.platform.qtt_solver import QTTAcceleratedSolver

        baseline = _LinearDecaySolver()
        solver = QTTAcceleratedSolver(
            baseline_solver=baseline,
            policy=AccelerationPolicy(warmup_steps=2, max_allowed_rank=64),
            max_rank=32,
        )

        result = solver.solve(smooth_state, t_span=(0.0, 0.05), dt=0.01)
        assert result.steps_taken == 5
        assert result.t_final == pytest.approx(0.05, abs=1e-10)

    def test_qtt_solver_rank_growth_report(self, smooth_state):
        from tensornet.platform.acceleration import AccelerationPolicy
        from tensornet.platform.qtt_solver import QTTAcceleratedSolver

        baseline = _LinearDecaySolver()
        solver = QTTAcceleratedSolver(
            baseline_solver=baseline,
            policy=AccelerationPolicy(warmup_steps=0),
            max_rank=32,
        )
        solver.solve(smooth_state, t_span=(0.0, 0.05), dt=0.01)

        report = solver.rank_growth_report(problem_name="decay_test")
        assert report.n_steps >= 5
        assert report.solver_name == solver.name


# ═══════════════════════════════════════════════════════════════════════════════
# QTT-Accelerated Domain Solvers (tensornet.packs.qtt_accelerated)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQTTBurgersSolver:
    """Tests for QTT-accelerated Burgers solver."""

    def test_burgers_step(self, smooth_state):
        from tensornet.packs.qtt_accelerated import QTTBurgersSolver

        solver = QTTBurgersSolver(nu=0.01, max_rank=32)
        assert "Burgers" in solver.name

        new_state = solver.step(smooth_state, dt=0.001)
        assert new_state.t > smooth_state.t
        assert "u" in new_state.fields
        assert new_state.fields["u"].data.shape == (128,)

    def test_burgers_solve(self, smooth_state):
        from tensornet.packs.qtt_accelerated import QTTBurgersSolver

        solver = QTTBurgersSolver(nu=0.02, max_rank=32)
        result = solver.solve(smooth_state, t_span=(0.0, 0.005), dt=0.001)
        assert result.steps_taken == 5
        assert result.final_state.fields["u"].data.shape == (128,)
        # Solution should still be finite
        assert torch.isfinite(result.final_state.fields["u"].data).all()

    def test_burgers_rank_growth_report(self, smooth_state):
        from tensornet.packs.qtt_accelerated import QTTBurgersSolver

        solver = QTTBurgersSolver(nu=0.01, max_rank=32)
        solver.solve(smooth_state, t_span=(0.0, 0.003), dt=0.001)
        report = solver.rank_growth_report()
        assert report.solver_name == solver.name
        assert report.n_steps >= 3

    def test_burgers_error_vs_rank(self, smooth_field):
        from tensornet.packs.qtt_accelerated import QTTBurgersSolver

        solver = QTTBurgersSolver(nu=0.01, max_rank=32)
        points = solver.error_vs_rank(smooth_field, rank_schedule=[2, 4, 8, 16])
        assert len(points) == 4


class TestQTTAdvDiffSolver:
    """Tests for QTT-accelerated advection-diffusion solver."""

    def test_advdiff_step(self, smooth_state):
        from tensornet.packs.qtt_accelerated import QTTAdvDiffSolver

        solver = QTTAdvDiffSolver(c=1.0, alpha=0.01, max_rank=32)
        assert "AdvDiff" in solver.name

        new_state = solver.step(smooth_state, dt=0.001)
        assert new_state.t > smooth_state.t
        assert torch.isfinite(new_state.fields["u"].data).all()

    def test_advdiff_solve_short(self, smooth_state):
        from tensornet.packs.qtt_accelerated import QTTAdvDiffSolver

        solver = QTTAdvDiffSolver(c=0.5, alpha=0.02, max_rank=32)
        result = solver.solve(smooth_state, t_span=(0.0, 0.003), dt=0.001)
        assert result.steps_taken == 3
        assert torch.isfinite(result.final_state.fields["u"].data).all()


class TestQTTMaxwellSolver:
    """Tests for QTT-accelerated Maxwell solver."""

    @pytest.fixture
    def maxwell_state(self, mesh_128):
        """State with E and H fields for Maxwell solver."""
        N = 128
        x = torch.linspace(0.0, 2.0 * math.pi, N, dtype=torch.float64)
        E = FieldData(name="E", data=torch.sin(x), mesh=mesh_128)
        H = FieldData(
            name="H",
            data=torch.cos(x[:N]),
            mesh=mesh_128,
        )
        return SimulationState(
            t=0.0, fields={"E": E, "H": H}, mesh=mesh_128
        )

    def test_maxwell_step(self, maxwell_state):
        from tensornet.packs.qtt_accelerated import QTTMaxwellSolver

        solver = QTTMaxwellSolver(max_rank=32)
        new_state = solver.step(maxwell_state, dt=0.001)
        assert "E" in new_state.fields
        assert "H" in new_state.fields
        assert torch.isfinite(new_state.fields["E"].data).all()

    def test_maxwell_solve_short(self, maxwell_state):
        from tensornet.packs.qtt_accelerated import QTTMaxwellSolver

        solver = QTTMaxwellSolver(max_rank=32)
        result = solver.solve(maxwell_state, t_span=(0.0, 0.003), dt=0.001)
        assert result.steps_taken == 3


class TestQTTVlasovSolver:
    """Tests for QTT-accelerated Vlasov solver."""

    @pytest.fixture
    def vlasov_state(self):
        """State with distribution function f(x, v) for Vlasov solver."""
        Nx, Nv = 64, 128
        mesh = StructuredMesh(
            shape=(Nx * Nv,),
            domain=((0.0, 4.0 * math.pi),),
        )
        x = torch.linspace(0, 4.0 * math.pi, Nx, dtype=torch.float64)
        v = torch.linspace(-6.0, 6.0, Nv, dtype=torch.float64)
        X, V = torch.meshgrid(x, v, indexing="ij")
        # Maxwellian with perturbation
        f = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-V ** 2 / 2.0) * (
            1.0 + 0.01 * torch.cos(0.5 * X)
        )
        fd = FieldData(name="distribution_function", data=f.reshape(-1), mesh=mesh)
        return SimulationState(t=0.0, fields={"distribution_function": fd}, mesh=mesh)

    def test_vlasov_step(self, vlasov_state):
        from tensornet.packs.qtt_accelerated import QTTVlasovSolver

        solver = QTTVlasovSolver(Nx=64, Nv=128, max_rank=16)
        new_state = solver.step(vlasov_state, dt=0.01)
        assert "distribution_function" in new_state.fields
        assert torch.isfinite(new_state.fields["distribution_function"].data).all()

    def test_vlasov_solve_short(self, vlasov_state):
        from tensornet.packs.qtt_accelerated import QTTVlasovSolver

        solver = QTTVlasovSolver(Nx=64, Nv=128, max_rank=16)
        result = solver.solve(vlasov_state, t_span=(0.0, 0.02), dt=0.01, max_steps=2)
        assert result.steps_taken == 2
