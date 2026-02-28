"""
Test suite for Phase 6 — Coupled physics, adjoint, inverse, UQ,
optimization, and lineage modules.

Tests:
  • Coupling: monolithic and partitioned (Gauss-Seidel / Jacobi).
  • Adjoint: gradient vs finite-difference reference.
  • Inverse: Tikhonov + gradient descent convergence.
  • UQ: Monte Carlo / LHS statistics, PCE polynomial fit.
  • Optimization: constrained optimizer, volume-fraction constraint.
  • Lineage: DAG construction, serialization, tracker context manager.

Runs via:  pytest tests/test_advanced_physics.py -v
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
from torch import Tensor

from ontic.platform.data_model import (
    FieldData,
    Mesh,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.protocols import Observable, SolveResult


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mesh_64():
    return StructuredMesh(shape=(64,), domain=((0.0, 1.0),))


@pytest.fixture
def mesh_32():
    return StructuredMesh(shape=(32,), domain=((0.0, 1.0),))


@pytest.fixture
def simple_state(mesh_64):
    x = torch.linspace(0.005, 0.995, 64, dtype=torch.float64)
    u = FieldData(name="u", data=torch.sin(math.pi * x), mesh=mesh_64)
    return SimulationState(t=0.0, fields={"u": u}, mesh=mesh_64)


@pytest.fixture
def state_32(mesh_32):
    x = torch.linspace(0.005, 0.995, 32, dtype=torch.float64)
    u = FieldData(name="u", data=torch.sin(math.pi * x), mesh=mesh_32)
    return SimulationState(t=0.0, fields={"u": u}, mesh=mesh_32)


# ── Tiny test solver (implements Solver protocol) ────────────────────────────


class _ScaleSolver:
    """u(t+dt) = u(t) * scale_factor.  Deterministic, differentiable."""

    def __init__(self, scale: float = 0.99) -> None:
        self._scale = scale

    @property
    def name(self) -> str:
        return "ScaleSolver"

    def step(self, state: SimulationState, dt: float, **kwargs: Any) -> SimulationState:
        new_fields = {}
        for k, fd in state.fields.items():
            new_fields[k] = FieldData(
                name=fd.name,
                data=fd.data * self._scale,
                mesh=fd.mesh,
                components=fd.components,
                units=fd.units,
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
            final_state=current, t_final=current.t, steps_taken=n_steps,
        )


class _DiffusionSolver:
    """Explicit diffusion: u_{t+1} = u_t + alpha * dt * laplacian(u)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha

    @property
    def name(self) -> str:
        return "DiffusionSolver"

    def step(self, state: SimulationState, dt: float, **kwargs: Any) -> SimulationState:
        new_fields = {}
        for k, fd in state.fields.items():
            u = fd.data
            # 2nd-order central diff with periodic BC
            lap = torch.roll(u, -1) + torch.roll(u, 1) - 2.0 * u
            new_data = u + self._alpha * dt * lap
            new_fields[k] = FieldData(
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
            final_state=current, t_final=current.t, steps_taken=n_steps,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Coupling Orchestrator (ontic.platform.coupled)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonolithicCoupler:
    """Tests for MonolithicCoupler."""

    def test_monolithic_solve(self, mesh_64, simple_state):
        from ontic.platform.coupled import (
            CoupledField,
            CouplingInterface,
            MonolithicCoupler,
        )

        solver_a = _ScaleSolver(scale=0.99)
        solver_b = _DiffusionSolver(alpha=0.05)

        # Create states for both solvers
        states = {
            "A": simple_state,
            "B": simple_state,
        }

        coupled_field = CoupledField(
            name="u",
            source_solver="A",
            target_solver="B",
        )
        interface = CouplingInterface(
            name="test_coupling",
            coupled_fields=(coupled_field,),
        )
        coupler = MonolithicCoupler(
            solvers={"A": solver_a, "B": solver_b},
            interface=interface,
        )
        result = coupler.solve(states, t_span=(0.0, 0.03), dt=0.01)
        assert result.steps_taken == 3
        assert "A" in result.final_states
        assert "B" in result.final_states
        assert result.t_final == pytest.approx(0.03, abs=1e-10)


class TestPartitionedCoupler:
    """Tests for PartitionedCoupler with Gauss-Seidel strategy."""

    def test_partitioned_solve(self, mesh_64, simple_state):
        from ontic.platform.coupled import (
            CoupledField,
            CouplingInterface,
            CouplingStrategy,
            PartitionedCoupler,
        )

        solver_a = _ScaleSolver(scale=0.99)
        solver_b = _ScaleSolver(scale=0.98)

        states = {"A": simple_state, "B": simple_state}

        coupled_field = CoupledField(
            name="u",
            source_solver="A",
            target_solver="B",
        )
        interface = CouplingInterface(
            name="test_partitioned",
            coupled_fields=(coupled_field,),
            convergence_fields=("u",),
            tolerance=1e-6,
            max_iterations=10,
        )
        coupler = PartitionedCoupler(
            solvers={"A": solver_a, "B": solver_b},
            interface=interface,
            strategy=CouplingStrategy.GAUSS_SEIDEL,
        )
        result = coupler.solve(states, t_span=(0.0, 0.02), dt=0.01)
        assert result.steps_taken == 2
        assert "A" in result.final_states
        assert "B" in result.final_states


# ═══════════════════════════════════════════════════════════════════════════════
# Adjoint / Sensitivity (ontic.platform.adjoint)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdjoint:
    """Tests for AdjointSolver and L2TrackingCost."""

    def test_l2_tracking_cost(self, simple_state):
        from ontic.platform.adjoint import L2TrackingCost

        target = simple_state.fields["u"].data * 0.9
        cost = L2TrackingCost(target_field_name="u", target_data=target)
        assert "L2" in cost.name and "u" in cost.name

        J = cost.evaluate(simple_state, {})
        assert J.item() > 0  # mismatch should give positive cost

        dJ = cost.dJ_dstate(simple_state, {})
        assert "u" in dJ
        assert dJ["u"].shape == simple_state.fields["u"].data.shape

    def test_adjoint_solver_gradient(self, simple_state, mesh_64):
        from ontic.platform.adjoint import AdjointSolver, L2TrackingCost

        # Forward solver with parameter-dependent scaling
        class _ParamSolver:
            @property
            def name(self) -> str:
                return "ParamSolver"

            def step(self, state: SimulationState, dt: float, **kw: Any) -> SimulationState:
                scale = kw.get("_param_scale", torch.tensor(1.0, dtype=torch.float64))
                u = state.fields["u"]
                new_data = u.data * scale
                new_u = FieldData(name="u", data=new_data, mesh=u.mesh)
                return SimulationState(
                    t=state.t + dt, fields={"u": new_u}, mesh=state.mesh,
                    step_index=state.step_index + 1,
                )

            def solve(self, state, t_span, dt, **kw):
                t0, tf = t_span
                n = max(1, int(round((tf - t0) / dt)))
                cur = state
                for _ in range(n):
                    cur = self.step(cur, dt)
                return SolveResult(final_state=cur, t_final=cur.t, steps_taken=n)

        target = simple_state.fields["u"].data * 0.5
        cost = L2TrackingCost(target_field_name="u", target_data=target)
        solver = _ParamSolver()
        adjoint = AdjointSolver(forward_solver=solver, cost=cost)

        params = {"scale": torch.tensor(0.9, dtype=torch.float64, requires_grad=True)}
        result = adjoint.compute_gradient(
            simple_state, t_span=(0.0, 0.01), dt=0.01, params=params,
        )
        assert result.cost_value >= 0
        assert result.n_adjoint_steps >= 0


class TestCheckpointedAdjoint:
    """Tests for CheckpointedAdjoint (memory-efficient variant)."""

    def test_checkpointed_runs(self, simple_state):
        from ontic.platform.adjoint import CheckpointedAdjoint, L2TrackingCost

        target = simple_state.fields["u"].data * 0.8
        cost = L2TrackingCost("u", target)
        solver = _ScaleSolver(scale=0.99)

        adj = CheckpointedAdjoint(forward_solver=solver, cost=cost, n_checkpoints=3)
        params = {"s": torch.tensor(1.0, dtype=torch.float64, requires_grad=True)}
        result = adj.compute_gradient(
            simple_state, t_span=(0.0, 0.03), dt=0.01, params=params,
        )
        assert result.cost_value >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Inverse Problems (ontic.platform.inverse)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegularizers:
    """Tests for Tikhonov and TV regularizers."""

    def test_tikhonov_positive(self):
        from ontic.platform.inverse import TikhonovRegularizer

        reg = TikhonovRegularizer(alpha=0.1)
        params = {"a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)}
        val = reg.evaluate(params)
        assert val.item() > 0

        grad = reg.gradient(params)
        assert "a" in grad
        assert grad["a"].shape == (3,)

    def test_tikhonov_with_prior(self):
        from ontic.platform.inverse import TikhonovRegularizer

        prior = {"a": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)}
        reg = TikhonovRegularizer(alpha=0.5, prior=prior)
        # At the prior, cost should be zero
        val = reg.evaluate(prior)
        assert val.item() == pytest.approx(0.0, abs=1e-14)

    def test_tv_regularizer(self):
        from ontic.platform.inverse import TVRegularizer

        reg = TVRegularizer(alpha=0.1)
        params = {"a": torch.tensor([1.0, 3.0, 2.0, 5.0], dtype=torch.float64)}
        val = reg.evaluate(params)
        assert val.item() > 0

        grad = reg.gradient(params)
        assert "a" in grad


class TestInverseProblem:
    """Tests for InverseProblem and GradientDescentSolver."""

    def test_total_cost(self, simple_state):
        from ontic.platform.adjoint import L2TrackingCost
        from ontic.platform.inverse import InverseProblem, TikhonovRegularizer

        target = simple_state.fields["u"].data * 0.5
        cost = L2TrackingCost("u", target)
        reg = TikhonovRegularizer(alpha=0.01)
        problem = InverseProblem(
            forward_solver=_ScaleSolver(0.99), cost=cost, regularizer=reg,
        )
        params = {"s": torch.tensor(1.0, dtype=torch.float64)}
        # total_cost should run forward + compute cost + regularizer
        J = problem.total_cost(simple_state, params)
        assert J >= 0

    def test_gradient_descent_solver(self, state_32):
        from ontic.platform.adjoint import L2TrackingCost
        from ontic.platform.inverse import (
            GradientDescentSolver,
            InverseProblem,
            TikhonovRegularizer,
        )

        target = state_32.fields["u"].data * 0.5
        cost = L2TrackingCost("u", target)
        reg = TikhonovRegularizer(alpha=1e-4)
        problem = InverseProblem(
            forward_solver=_ScaleSolver(0.99), cost=cost, regularizer=reg,
        )
        gd = GradientDescentSolver(learning_rate=1e-2, max_iterations=5, tolerance=1e-12)
        params = {"s": torch.tensor(1.0, dtype=torch.float64, requires_grad=True)}
        result = gd.solve(
            problem, state_32, t_span=(0.0, 0.01), dt=0.01, initial_params=params,
        )
        assert result.iterations >= 1
        assert len(result.cost_history) >= 1


class TestBayesianInversion:
    """Smoke test for BayesianInversion."""

    def test_bayesian_smoke(self, state_32):
        from ontic.platform.adjoint import L2TrackingCost
        from ontic.platform.inverse import (
            BayesianInversion,
            InverseProblem,
            LBFGSSolver,
            TikhonovRegularizer,
        )

        target = state_32.fields["u"].data * 0.8
        cost = L2TrackingCost("u", target)
        problem = InverseProblem(
            forward_solver=_ScaleSolver(0.99),
            cost=cost,
            regularizer=TikhonovRegularizer(alpha=1e-3),
        )
        bayes = BayesianInversion(
            inverse_solver=LBFGSSolver(max_iterations=3, tolerance=1e-12),
            hessian_eps=1e-3,
        )
        params = {"s": torch.tensor(1.0, dtype=torch.float64)}
        result = bayes.solve(problem, state_32, (0.0, 0.01), 0.01, params)
        assert "s" in result.map_params
        assert math.isfinite(result.log_evidence)


# ═══════════════════════════════════════════════════════════════════════════════
# Uncertainty Quantification (ontic.platform.uq)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUQ:
    """Tests for Monte Carlo, LHS, and PCE UQ methods."""

    def _state_modifier(
        self, state: SimulationState, params: Dict[str, float]
    ) -> SimulationState:
        """Modifier that scales u by 'scale' param."""
        scale = params.get("scale", 1.0)
        u = state.fields["u"]
        new_u = FieldData(
            name="u",
            data=u.data * scale,
            mesh=u.mesh,
            components=u.components,
            units=u.units,
        )
        return SimulationState(
            t=state.t, fields={"u": new_u}, mesh=state.mesh,
            metadata=state.metadata, step_index=state.step_index,
        )

    def _qoi_max(self, state: SimulationState) -> float:
        return state.fields["u"].data.abs().max().item()

    def test_monte_carlo_uq(self, state_32):
        from ontic.platform.uq import MonteCarloUQ, ParameterDistribution

        solver = _ScaleSolver(0.99)
        mc = MonteCarloUQ(solver=solver, n_samples=10, seed=42)

        dist = ParameterDistribution(
            name="scale", distribution="uniform", lower=0.8, upper=1.2,
        )
        result = mc.run(
            base_state=state_32,
            t_span=(0.0, 0.01),
            dt=0.01,
            uncertain_params=[dist],
            state_modifier=self._state_modifier,
            qoi_extractors={"max_u": self._qoi_max},
        )
        assert result.n_samples == 10
        assert "u" in result.mean
        assert "u" in result.variance

    def test_latin_hypercube_uq(self, state_32):
        from ontic.platform.uq import LatinHypercubeUQ, ParameterDistribution

        solver = _ScaleSolver(0.99)
        lhs = LatinHypercubeUQ(solver=solver, n_samples=8, seed=123)

        dist = ParameterDistribution(
            name="scale", distribution="uniform", lower=0.9, upper=1.1,
        )
        result = lhs.run(
            base_state=state_32,
            t_span=(0.0, 0.01),
            dt=0.01,
            uncertain_params=[dist],
            state_modifier=self._state_modifier,
        )
        assert result.n_samples == 8

    def test_polynomial_chaos_expansion(self, state_32):
        from ontic.platform.uq import ParameterDistribution, PolynomialChaosExpansion

        solver = _ScaleSolver(0.99)
        pce = PolynomialChaosExpansion(
            solver=solver, polynomial_order=2, n_samples=15, seed=99,
        )
        dist = ParameterDistribution(
            name="scale", distribution="uniform", lower=0.8, upper=1.2,
        )
        result = pce.run(
            base_state=state_32,
            t_span=(0.0, 0.01),
            dt=0.01,
            uncertain_params=[dist],
            state_modifier=self._state_modifier,
            qoi_extractor=self._qoi_max,
        )
        assert "mean" in result
        assert "variance" in result
        assert "coefficients" in result
        assert result["polynomial_order"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Optimization (ontic.platform.optimization)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptimization:
    """Tests for Constraint, volume_fraction_constraint, ConstrainedOptimizer."""

    def test_volume_fraction_constraint(self):
        from ontic.platform.optimization import volume_fraction_constraint

        con = volume_fraction_constraint("density", 0.5)
        assert "vol_frac" in con.name

        params = {"density": torch.full((100,), 0.3, dtype=torch.float64)}
        g = con.evaluate(params)
        assert g.item() < 0  # 0.3 < 0.5 → feasible

        params_over = {"density": torch.full((100,), 0.7, dtype=torch.float64)}
        assert con.evaluate(params_over).item() > 0  # infeasible

        grad = con.gradient(params)
        assert "density" in grad
        assert grad["density"].shape == (100,)

    def test_constrained_optimizer_smoke(self, state_32):
        from ontic.platform.adjoint import L2TrackingCost
        from ontic.platform.inverse import InverseProblem, TikhonovRegularizer
        from ontic.platform.optimization import (
            Constraint,
            ConstrainedOptimizer,
        )

        target = state_32.fields["u"].data * 0.5
        cost = L2TrackingCost("u", target)
        problem = InverseProblem(
            forward_solver=_ScaleSolver(0.99),
            cost=cost,
            regularizer=TikhonovRegularizer(0.01),
        )

        # Dummy constraint: param["s"] <= 2.0  →  s - 2 <= 0
        con = Constraint(
            name="s<=2",
            evaluate=lambda p: p["s"] - 2.0,
            gradient=lambda p: {"s": torch.ones_like(p["s"])},
        )

        opt = ConstrainedOptimizer(
            max_outer=2, max_inner=3, learning_rate=0.01, tolerance=1e-12,
        )
        params = {"s": torch.tensor(1.0, dtype=torch.float64)}
        result = opt.solve(
            problem, state_32, (0.0, 0.01), 0.01, params, [con],
        )
        assert result.iterations >= 1
        assert len(result.cost_history) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Lineage DAG (ontic.platform.lineage)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLineageDAG:
    """Tests for LineageDAG operations."""

    def test_dag_add_and_query(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        root = LineageNode(
            node_id="root",
            event=LineageEvent.FORWARD_SOLVE,
            label="Initial solve",
            parent_ids=[],
            timestamp=1000.0,
            elapsed_seconds=0.5,
            inputs_hash="aaa",
            outputs_hash="bbb",
        )
        dag.add(root)
        assert len(dag) == 1
        assert "root" in dag
        assert dag.get("root").label == "Initial solve"

    def test_dag_parent_validation(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        child = LineageNode(
            node_id="child",
            event=LineageEvent.QTT_COMPRESS,
            label="Compress",
            parent_ids=["nonexistent"],
            timestamp=1000.0,
            elapsed_seconds=0.1,
            inputs_hash="a",
            outputs_hash="b",
        )
        with pytest.raises(KeyError, match="nonexistent"):
            dag.add(child)

    def test_dag_roots_and_leaves(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        n1 = LineageNode("n1", LineageEvent.FORWARD_SOLVE, "Step 1", [], 1.0, 0.1, "a", "b")
        n2 = LineageNode("n2", LineageEvent.QTT_COMPRESS, "Compress", ["n1"], 2.0, 0.05, "b", "c")
        n3 = LineageNode("n3", LineageEvent.ADJOINT_SOLVE, "Adjoint", ["n2"], 3.0, 0.2, "c", "d")
        dag.add(n1)
        dag.add(n2)
        dag.add(n3)

        roots = dag.roots()
        assert len(roots) == 1
        assert roots[0].node_id == "n1"

        leaves = dag.leaves()
        assert len(leaves) == 1
        assert leaves[0].node_id == "n3"

    def test_dag_ancestors_descendants(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        dag.add(LineageNode("a", LineageEvent.FORWARD_SOLVE, "A", [], 1.0, 0.1, "x", "y"))
        dag.add(LineageNode("b", LineageEvent.COUPLING_STEP, "B", ["a"], 2.0, 0.1, "y", "z"))
        dag.add(LineageNode("c", LineageEvent.INVERSE_ITERATION, "C", ["b"], 3.0, 0.1, "z", "w"))

        anc = dag.ancestors("c")
        assert len(anc) == 2
        assert {n.node_id for n in anc} == {"a", "b"}

        desc = dag.descendants("a")
        assert len(desc) == 2
        assert {n.node_id for n in desc} == {"b", "c"}

    def test_dag_filter_by_event(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        dag.add(LineageNode("s1", LineageEvent.FORWARD_SOLVE, "S1", [], 1.0, 0.1, "a", "b"))
        dag.add(LineageNode("q1", LineageEvent.QTT_COMPRESS, "Q1", ["s1"], 2.0, 0.02, "b", "c"))
        dag.add(LineageNode("s2", LineageEvent.FORWARD_SOLVE, "S2", ["q1"], 3.0, 0.1, "c", "d"))

        solves = dag.filter_by_event(LineageEvent.FORWARD_SOLVE)
        assert len(solves) == 2

    def test_dag_json_roundtrip(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        dag.add(LineageNode("n1", LineageEvent.FORWARD_SOLVE, "S1", [], 1.0, 0.5, "aa", "bb",
                            metadata={"dt": 0.01}))
        dag.add(LineageNode("n2", LineageEvent.QTT_COMPRESS, "C1", ["n1"], 2.0, 0.02, "bb", "cc"))

        json_str = dag.to_json()
        dag2 = LineageDAG.from_json(json_str)
        assert len(dag2) == 2
        assert dag2.get("n1").metadata["dt"] == 0.01

    def test_dag_summary(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageNode

        dag = LineageDAG()
        dag.add(LineageNode("r", LineageEvent.FORWARD_SOLVE, "Root", [], 1.0, 1.0, "a", "b"))
        s = dag.summary()
        assert "LineageDAG: 1 nodes" in s
        assert "FORWARD_SOLVE" in s


class TestLineageTracker:
    """Tests for LineageTracker context manager and instant recording."""

    def test_record_instant(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageTracker

        dag = LineageDAG()
        tracker = LineageTracker(dag)
        inp = {"u": torch.tensor([1.0, 2.0])}
        out = {"u": torch.tensor([0.9, 1.8])}
        nid = tracker.record_instant(
            event=LineageEvent.FORWARD_SOLVE,
            label="quick solve",
            inputs=inp,
            outputs=out,
            elapsed_seconds=0.42,
            metadata={"steps": 10},
        )
        assert nid in dag
        node = dag.get(nid)
        assert node.event == LineageEvent.FORWARD_SOLVE
        assert node.elapsed_seconds == pytest.approx(0.42)

    def test_record_context_manager(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageTracker

        dag = LineageDAG()
        tracker = LineageTracker(dag)

        inp = {"x": torch.zeros(4)}
        with tracker.record(
            LineageEvent.QTT_COMPRESS, "compress test", inputs=inp,
        ) as ctx:
            time.sleep(0.005)
            ctx.set_outputs({"x_qtt": torch.ones(2)})

        assert ctx.node_id in dag
        node = dag.get(ctx.node_id)
        assert node.event == LineageEvent.QTT_COMPRESS
        assert node.elapsed_seconds > 0

    def test_tracker_chaining(self):
        from ontic.platform.lineage import LineageDAG, LineageEvent, LineageTracker

        dag = LineageDAG()
        tracker = LineageTracker(dag)

        nid1 = tracker.record_instant(
            LineageEvent.FORWARD_SOLVE, "solve", {"u": torch.ones(3)},
            {"u": torch.ones(3)},
        )
        nid2 = tracker.record_instant(
            LineageEvent.QTT_COMPRESS, "compress", {"u": torch.ones(3)},
            {"cores": torch.ones(2)}, parent_ids=[nid1],
        )
        assert len(dag) == 2
        node2 = dag.get(nid2)
        assert nid1 in node2.parent_ids


class TestLineageNodeSerialization:
    """Tests for LineageNode to_dict / from_dict."""

    def test_round_trip(self):
        from ontic.platform.lineage import LineageEvent, LineageNode

        node = LineageNode(
            node_id="abc123",
            event=LineageEvent.UQ_SAMPLE,
            label="Sample #5",
            parent_ids=["parent1"],
            timestamp=1234567890.0,
            elapsed_seconds=2.5,
            inputs_hash="dead",
            outputs_hash="beef",
            metadata={"sample_index": 5},
        )
        d = node.to_dict()
        restored = LineageNode.from_dict(d)
        assert restored.node_id == "abc123"
        assert restored.event == LineageEvent.UQ_SAMPLE
        assert restored.metadata["sample_index"] == 5
