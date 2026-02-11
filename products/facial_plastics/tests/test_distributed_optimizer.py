"""Tests for the distributed (island-model) optimizer."""

from __future__ import annotations

from typing import Any, Dict, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from products.facial_plastics.metrics.distributed_optimizer import (
    DistributedOptimizationResult,
    DistributedOptimizer,
    IslandConfig,
    PoolBackend,
    _evaluate_single,
)
from products.facial_plastics.metrics.optimizer import (
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationResult,
    ParameterBound,
    ParetoFront,
)


# ── Helpers ──────────────────────────────────────────────────────

def _sphere_fn(params: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Simple sphere function: minimize sum-of-squares."""
    return np.array([np.sum(params ** 2)]), {"norm": float(np.linalg.norm(params))}


def _biobj_fn(params: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Two-objective test: f1 = x^2, f2 = (x-2)^2."""
    x = params[0]
    return np.array([x ** 2, (x - 2) ** 2]), {"x_abs": abs(x)}


def _make_objectives_single() -> list[ObjectiveSpec]:
    return [ObjectiveSpec(name="f1", direction="minimize", weight=1.0)]


def _make_objectives_biobj() -> list[ObjectiveSpec]:
    return [
        ObjectiveSpec(name="f1", direction="minimize", weight=1.0),
        ObjectiveSpec(name="f2", direction="minimize", weight=1.0),
    ]


def _make_bounds(n: int = 2, low: float = -5.0, high: float = 5.0) -> list[ParameterBound]:
    return [ParameterBound(name=f"x{i}", low=low, high=high) for i in range(n)]


# ── IslandConfig tests ───────────────────────────────────────────

class TestIslandConfig:
    def test_defaults(self) -> None:
        cfg = IslandConfig()
        assert cfg.n_islands == 4
        assert cfg.migration_interval == 5
        assert cfg.migration_size == 2
        assert cfg.pool_backend == PoolBackend.PROCESS
        assert cfg.max_workers is None
        assert cfg.chunk_size == 1

    def test_custom_values(self) -> None:
        cfg = IslandConfig(
            n_islands=8,
            migration_interval=10,
            migration_size=3,
            pool_backend=PoolBackend.THREAD,
            max_workers=16,
        )
        assert cfg.n_islands == 8
        assert cfg.pool_backend == PoolBackend.THREAD
        assert cfg.max_workers == 16


# ── DistributedOptimizationResult tests ──────────────────────────

class TestDistributedOptimizationResult:
    def test_summary_empty(self) -> None:
        r = DistributedOptimizationResult()
        s = r.summary()
        assert "DistOpt" in s
        assert "0 islands" in s

    def test_summary_populated(self) -> None:
        r = DistributedOptimizationResult(
            n_islands=4, n_migrations=3,
            total_evaluations=1600, wall_clock_seconds=5.5,
        )
        r.combined = OptimizationResult()
        r.combined.pareto_front = ParetoFront(
            individuals=[], objective_names=[], parameter_names=[],
        )
        s = r.summary()
        assert "4 islands" in s
        assert "3 migrations" in s
        assert "1600 evals" in s


# ── _evaluate_single tests ───────────────────────────────────────

class TestEvaluateSingle:
    def test_sphere(self) -> None:
        params = np.array([1.0, 2.0, 3.0])
        obj, cons = _evaluate_single(params, _sphere_fn)
        assert np.isclose(obj[0], 14.0)
        assert "norm" in cons

    def test_biobj(self) -> None:
        params = np.array([1.0])
        obj, cons = _biobj_fn(params)
        assert np.isclose(obj[0], 1.0)
        assert np.isclose(obj[1], 1.0)


# ── Constructor validation tests ─────────────────────────────────

class TestDistributedOptimizerInit:
    def test_valid_construction(self) -> None:
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            island_config=IslandConfig(n_islands=2),
        )
        assert opt._config.n_islands == 2

    def test_invalid_n_islands(self) -> None:
        with pytest.raises(ValueError, match="n_islands"):
            DistributedOptimizer(
                objectives=_make_objectives_single(),
                constraints=[],
                parameter_bounds=_make_bounds(2),
                island_config=IslandConfig(n_islands=0),
            )

    def test_invalid_migration_interval(self) -> None:
        with pytest.raises(ValueError, match="migration_interval"):
            DistributedOptimizer(
                objectives=_make_objectives_single(),
                constraints=[],
                parameter_bounds=_make_bounds(2),
                island_config=IslandConfig(migration_interval=0),
            )

    def test_invalid_migration_size(self) -> None:
        with pytest.raises(ValueError, match="migration_size"):
            DistributedOptimizer(
                objectives=_make_objectives_single(),
                constraints=[],
                parameter_bounds=_make_bounds(2),
                island_config=IslandConfig(migration_size=-1),
            )


# ── Full optimization run tests ──────────────────────────────────

class TestDistributedOptimizerOptimize:
    """Integration tests for the island-model optimizer.

    Use thread backend to avoid pickling issues with local functions
    and keep tests fast.
    """

    def test_single_island_sphere(self) -> None:
        """Single island should behave like baseline NSGA-II."""
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=10,
            n_generations=5,
            seed=42,
            island_config=IslandConfig(
                n_islands=1,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert isinstance(result, DistributedOptimizationResult)
        assert result.n_islands == 1
        assert result.total_evaluations > 0
        assert result.wall_clock_seconds > 0
        assert result.combined.pareto_front is not None
        assert len(result.island_results) == 1

    def test_multi_island_with_migration(self) -> None:
        """Multiple islands with migration should converge."""
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=10,
            n_generations=10,
            seed=123,
            island_config=IslandConfig(
                n_islands=3,
                migration_interval=3,
                migration_size=2,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert result.n_islands == 3
        assert result.n_migrations >= 3  # gens 3,6,9
        assert len(result.island_results) == 3
        assert result.combined.pareto_front is not None

    def test_biobj_pareto_front(self) -> None:
        """Two-objective problem should yield non-trivial Pareto front."""
        opt = DistributedOptimizer(
            objectives=_make_objectives_biobj(),
            constraints=[],
            parameter_bounds=_make_bounds(1, low=-1.0, high=3.0),
            population_size=20,
            n_generations=15,
            seed=99,
            island_config=IslandConfig(
                n_islands=2,
                migration_interval=5,
                migration_size=2,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_biobj_fn)
        pf = result.combined.pareto_front
        assert pf is not None
        # Pareto front should span 0 ≤ x ≤ 2
        assert len(pf.individuals) >= 1

    def test_with_constraints(self) -> None:
        """Constraints should be evaluated and enforced."""
        constraints = [
            ConstraintSpec(name="norm", min_value=0.0, max_value=3.0, is_hard=True),
        ]
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=constraints,
            parameter_bounds=_make_bounds(2),
            population_size=10,
            n_generations=5,
            seed=42,
            island_config=IslandConfig(
                n_islands=2,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        # All Pareto individuals should be feasible
        for ind in result.combined.pareto_front.individuals:
            assert ind.is_feasible

    def test_fully_connected_migration(self) -> None:
        """Fully-connected migration topology should work."""
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=8,
            n_generations=6,
            seed=77,
            island_config=IslandConfig(
                n_islands=3,
                migration_interval=2,
                migration_size=1,
                migration_topology="fully_connected",
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert result.n_migrations >= 3

    def test_zero_migration_size(self) -> None:
        """migration_size=0 disables migration."""
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=8,
            n_generations=6,
            seed=55,
            island_config=IslandConfig(
                n_islands=2,
                migration_interval=2,
                migration_size=0,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert result.n_migrations == 0

    def test_progress_callback(self) -> None:
        """Progress callback should be invoked each generation."""
        calls: list[tuple[int, int, float]] = []

        def cb(gen: int, total: int, hv: float) -> None:
            calls.append((gen, total, hv))

        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=8,
            n_generations=4,
            seed=42,
            island_config=IslandConfig(
                n_islands=1,
                pool_backend=PoolBackend.THREAD,
                max_workers=1,
            ),
        )
        opt.optimize(_sphere_fn, progress_callback=cb)
        assert len(calls) == 4
        assert calls[-1][0] == 4
        assert calls[-1][1] == 4

    def test_best_compromise_populated(self) -> None:
        """Best compromise should be set on the combined result."""
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=10,
            n_generations=5,
            seed=42,
            island_config=IslandConfig(
                n_islands=2,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert result.combined.best_compromise is not None

    def test_convergence_history(self) -> None:
        """Convergence history should have one entry per generation."""
        n_gen = 7
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=8,
            n_generations=n_gen,
            seed=42,
            island_config=IslandConfig(
                n_islands=2,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert len(result.combined.convergence_history) == n_gen

    def test_per_island_results(self) -> None:
        """Each island should have its own OptimizationResult."""
        n_islands = 3
        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=8,
            n_generations=4,
            seed=42,
            island_config=IslandConfig(
                n_islands=n_islands,
                pool_backend=PoolBackend.THREAD,
                max_workers=2,
            ),
        )
        result = opt.optimize(_sphere_fn)
        assert len(result.island_results) == n_islands
        for ir in result.island_results:
            assert isinstance(ir, OptimizationResult)
            assert ir.pareto_front is not None

    def test_evaluation_failure_handled(self) -> None:
        """If evaluate_fn raises, individual should be marked infeasible."""
        call_count = 0

        def failing_fn(params: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
            nonlocal call_count
            call_count += 1
            if call_count % 5 == 0:
                raise RuntimeError("Deliberate failure")
            return np.array([np.sum(params ** 2)]), {}

        opt = DistributedOptimizer(
            objectives=_make_objectives_single(),
            constraints=[],
            parameter_bounds=_make_bounds(2),
            population_size=10,
            n_generations=3,
            seed=42,
            island_config=IslandConfig(
                n_islands=1,
                pool_backend=PoolBackend.THREAD,
                max_workers=1,
            ),
        )
        # Should not raise even with intermittent failures
        result = opt.optimize(failing_fn)
        assert result.total_evaluations > 0
