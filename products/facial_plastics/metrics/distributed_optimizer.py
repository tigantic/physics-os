"""Distributed multi-objective surgical plan optimization.

Wraps the single-node PlanOptimizer with a parallel evaluation backend
so that the expensive per-individual simulation calls fan out across
available CPU cores (or processes).

Backends supported:
  * ``ProcessPoolExecutor`` (stdlib, default)
  * ``ThreadPoolExecutor`` (for I/O-bound or GIL-releasing evaluators)

The interface mirrors ``PlanOptimizer.optimize`` exactly; replace the
optimizer instance and the rest of the pipeline is unchanged.

Design:
  - Island model: N sub-populations evolve independently for
    ``migration_interval`` generations, then exchange elite individuals.
  - Within each island every individual evaluation is dispatched to
    the pool so ``evaluate_fn`` calls run in parallel.
  - No external dependencies (Dask/Ray) — just ``concurrent.futures``.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from products.facial_plastics.metrics.optimizer import (
    ConstraintSpec,
    Individual,
    ObjectiveSpec,
    OptimizationResult,
    ParameterBound,
    ParetoFront,
    PlanOptimizer,
    _crowding_distance,
    _fast_non_dominated_sort,
    _hypervolume_2d,
    _polynomial_mutation,
    _sbx_crossover,
    _tournament_select,
)

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────

class PoolBackend(Enum):
    """Parallel backend selection."""
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class IslandConfig:
    """Configuration for island-model parallel optimizer."""

    n_islands: int = 4
    """Number of sub-populations (islands)."""

    migration_interval: int = 5
    """Exchange elite individuals every N generations."""

    migration_size: int = 2
    """Number of elite individuals migrated between islands per interval."""

    migration_topology: str = "ring"
    """Migration topology: 'ring' or 'fully_connected'."""

    pool_backend: PoolBackend = PoolBackend.PROCESS
    """Parallel backend for evaluate_fn dispatch."""

    max_workers: Optional[int] = None
    """Max worker processes/threads.  ``None`` → os.cpu_count()."""

    chunk_size: int = 1
    """Number of individuals to batch per worker task."""


# ── Results ──────────────────────────────────────────────────────

@dataclass
class DistributedOptimizationResult:
    """Extended result including per-island diagnostics."""

    combined: OptimizationResult = field(default_factory=OptimizationResult)
    """Merged Pareto front across all islands."""

    island_results: List[OptimizationResult] = field(default_factory=list)
    """Per-island OptimizationResult before merge."""

    n_islands: int = 0
    n_migrations: int = 0
    total_evaluations: int = 0
    wall_clock_seconds: float = 0.0

    def summary(self) -> str:
        pf_size = len(self.combined.pareto_front.individuals) if self.combined.pareto_front else 0
        return (
            f"DistOpt: {self.n_islands} islands, "
            f"{self.n_migrations} migrations, "
            f"{self.total_evaluations} evals, "
            f"Pareto size={pf_size}, "
            f"{self.wall_clock_seconds:.1f}s"
        )


# ── Parallel evaluation helpers ──────────────────────────────────

def _evaluate_single(
    params: np.ndarray,
    evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Top-level function (picklable) for ProcessPoolExecutor."""
    return evaluate_fn(params)


# ── Island state ─────────────────────────────────────────────────

class _Island:
    """Internal state for one island sub-population."""

    __slots__ = ("optimizer", "population", "generation", "convergence")

    def __init__(
        self,
        optimizer: PlanOptimizer,
        population: List[Individual],
    ) -> None:
        self.optimizer = optimizer
        self.population = population
        self.generation: int = 0
        self.convergence: List[float] = []


# ── Main class ───────────────────────────────────────────────────

class DistributedOptimizer:
    """Island-model distributed NSGA-II optimizer.

    Usage::

        dist = DistributedOptimizer(
            objectives=[...],
            constraints=[...],
            parameter_bounds=[...],
            island_config=IslandConfig(n_islands=4, max_workers=8),
        )
        result = dist.optimize(evaluate_fn)
        # result.combined is the merged OptimizationResult
    """

    def __init__(
        self,
        objectives: List[ObjectiveSpec],
        constraints: List[ConstraintSpec],
        parameter_bounds: List[ParameterBound],
        *,
        population_size: int = 40,
        n_generations: int = 50,
        crossover_eta: float = 20.0,
        mutation_eta: float = 20.0,
        crossover_prob: float = 0.9,
        seed: int = 42,
        island_config: Optional[IslandConfig] = None,
    ) -> None:
        self._objectives = objectives
        self._constraints = constraints
        self._bounds = parameter_bounds
        self._pop_size = population_size
        self._n_gen = n_generations
        self._cx_eta = crossover_eta
        self._mut_eta = mutation_eta
        self._cx_prob = crossover_prob
        self._seed = seed
        self._config = island_config or IslandConfig()

        if self._config.n_islands < 1:
            raise ValueError("n_islands must be >= 1")
        if self._config.migration_interval < 1:
            raise ValueError("migration_interval must be >= 1")
        if self._config.migration_size < 0:
            raise ValueError("migration_size must be >= 0")

    # ── Public API ───────────────────────────────────────────────

    def optimize(
        self,
        evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
        *,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> DistributedOptimizationResult:
        """Run island-model NSGA-II with parallel evaluation.

        Args:
            evaluate_fn: Same signature as ``PlanOptimizer.optimize``'s
                evaluate_fn.  Must be picklable for process backend.
            progress_callback: Called with ``(generation, total, hypervolume)``
                on the merged front after each generation.

        Returns:
            DistributedOptimizationResult with the merged Pareto front.
        """
        t0 = time.monotonic()
        cfg = self._config
        rng = np.random.default_rng(self._seed)

        # Build per-island optimizers (different seeds for diversity)
        islands: List[_Island] = []
        for i in range(cfg.n_islands):
            island_seed = int(rng.integers(0, 2**31))
            opt = PlanOptimizer(
                objectives=self._objectives,
                constraints=self._constraints,
                parameter_bounds=self._bounds,
                population_size=self._pop_size,
                n_generations=self._n_gen,
                crossover_eta=self._cx_eta,
                mutation_eta=self._mut_eta,
                crossover_prob=self._cx_prob,
                seed=island_seed,
            )
            pop = opt._initialize_population()
            islands.append(_Island(optimizer=opt, population=pop))

        # Choose executor
        max_w = cfg.max_workers or os.cpu_count() or 2
        executor_cls = (
            ProcessPoolExecutor
            if cfg.pool_backend == PoolBackend.PROCESS
            else ThreadPoolExecutor
        )

        total_evals = 0
        n_migrations = 0
        merged_convergence: List[float] = []

        with executor_cls(max_workers=max_w) as pool:
            # Evaluate initial populations (parallel)
            for island in islands:
                total_evals += self._parallel_evaluate(
                    pool, island.population, evaluate_fn,
                    island.optimizer,
                )

            for gen in range(self._n_gen):
                # === Advance each island by one generation ===
                for island in islands:
                    island.population = self._advance_generation(
                        island, gen, pool, evaluate_fn,
                    )
                    island.generation = gen + 1
                    total_evals += len(island.population)

                # === Migration ===
                if (
                    cfg.migration_size > 0
                    and (gen + 1) % cfg.migration_interval == 0
                ):
                    self._migrate(islands, rng)
                    n_migrations += 1

                # === Convergence tracking (merged) ===
                all_pop = [
                    ind for island in islands
                    for ind in island.population
                ]
                feasible = [
                    ind for ind in all_pop
                    if ind.is_feasible and ind.rank == 0
                ]
                hv = 0.0
                if feasible:
                    obj_mat = np.array([ind.objectives for ind in feasible])
                    n_obj = obj_mat.shape[1] if obj_mat.ndim == 2 else 1
                    if n_obj >= 2:
                        ref = np.max(obj_mat, axis=0) + 1.0
                        hv = _hypervolume_2d(obj_mat, ref)
                    else:
                        hv = -float(np.min(obj_mat[:, 0]))
                merged_convergence.append(hv)

                if progress_callback:
                    progress_callback(gen + 1, self._n_gen, hv)

        # === Merge Pareto fronts ===
        result = self._build_result(
            islands, merged_convergence, total_evals, n_migrations, t0,
        )

        logger.info("Distributed optimization complete: %s", result.summary())
        return result

    # ── Internal helpers ─────────────────────────────────────────

    def _parallel_evaluate(
        self,
        pool: ProcessPoolExecutor | ThreadPoolExecutor,
        population: List[Individual],
        evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
        optimizer: PlanOptimizer,
    ) -> int:
        """Dispatch evaluate_fn calls across workers and collect results."""
        futures: List[Tuple[int, Future[Tuple[np.ndarray, Dict[str, float]]]]] = []

        for idx, ind in enumerate(population):
            fut: Future[Tuple[np.ndarray, Dict[str, float]]] = pool.submit(
                _evaluate_single, ind.parameters, evaluate_fn,
            )
            futures.append((idx, fut))

        n_obj = len(optimizer._objectives)
        n_constraints = len(optimizer._constraints)
        obj_signs = optimizer._obj_signs

        for idx, fut in futures:
            ind = population[idx]
            try:
                obj_raw, constraint_values = fut.result()
                ind.objectives = obj_raw * obj_signs
                ind.constraints_violated = 0
                ind.constraint_penalty = 0.0
                for cs in optimizer._constraints:
                    value = constraint_values.get(cs.name, 0.0)
                    feasible, violation = cs.evaluate(value)
                    if not feasible:
                        ind.constraints_violated += 1
                        ind.constraint_penalty += cs.penalty_weight * violation
                    elif violation > 0:
                        ind.constraint_penalty += cs.penalty_weight * violation
                ind.is_feasible = ind.constraints_violated == 0
            except Exception as exc:
                logger.warning("Evaluation failed for %s: %s", ind.uid, exc)
                ind.objectives = np.full(n_obj, -1e10)
                ind.is_feasible = False
                ind.constraints_violated = n_constraints
                ind.constraint_penalty = 1e10

        return len(population)

    def _advance_generation(
        self,
        island: _Island,
        gen: int,
        pool: ProcessPoolExecutor | ThreadPoolExecutor,
        evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
    ) -> List[Individual]:
        """Run one NSGA-II generation for a single island."""
        opt = island.optimizer
        population = island.population
        pop_size = self._pop_size
        rng = opt._rng

        # Non-dominated sort + crowding
        fronts = _fast_non_dominated_sort(population)
        for front in fronts:
            _crowding_distance(population, front)

        # Generate offspring
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = _tournament_select(population, rng)
            p2 = _tournament_select(population, rng)
            c1_params, c2_params = _sbx_crossover(
                p1.parameters, p2.parameters,
                opt._bounds, rng,
                eta=opt._cx_eta, crossover_prob=opt._cx_prob,
            )
            c1_params = _polynomial_mutation(c1_params, opt._bounds, rng, eta=opt._mut_eta)
            c2_params = _polynomial_mutation(c2_params, opt._bounds, rng, eta=opt._mut_eta)
            offspring.append(Individual(parameters=c1_params, generation=gen + 1))
            if len(offspring) < pop_size:
                offspring.append(Individual(parameters=c2_params, generation=gen + 1))

        # Parallel evaluation of offspring
        self._parallel_evaluate(pool, offspring, evaluate_fn, opt)

        # Environmental selection
        combined = population + offspring
        fronts = _fast_non_dominated_sort(combined)
        for front in fronts:
            _crowding_distance(combined, front)

        next_pop: List[Individual] = []
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend([combined[i] for i in front])
            else:
                sorted_front = sorted(
                    front,
                    key=lambda i: combined[i].crowding_distance,
                    reverse=True,
                )
                remaining = pop_size - len(next_pop)
                next_pop.extend([combined[i] for i in sorted_front[:remaining]])
                break

        # Track convergence per island
        feasible = [ind for ind in next_pop if ind.is_feasible and ind.rank == 0]
        if feasible:
            obj_mat = np.array([ind.objectives for ind in feasible])
            n_obj = obj_mat.shape[1] if obj_mat.ndim == 2 else 1
            if n_obj >= 2:
                ref = np.max(obj_mat, axis=0) + 1.0
                island.convergence.append(_hypervolume_2d(obj_mat, ref))
            else:
                # Single-objective: track negative of best (lower is better)
                island.convergence.append(-float(np.min(obj_mat[:, 0])))
        else:
            island.convergence.append(0.0)

        return next_pop

    def _migrate(
        self,
        islands: List[_Island],
        rng: np.random.Generator,
    ) -> None:
        """Exchange elite individuals between islands."""
        cfg = self._config
        n = len(islands)

        # Collect emigrants (best by rank then crowding) from each island
        emigrants: List[List[Individual]] = []
        for island in islands:
            pop = sorted(
                island.population,
                key=lambda ind: (ind.rank, -ind.crowding_distance),
            )
            emigrants.append(pop[:cfg.migration_size])

        # Apply migration according to topology
        if cfg.migration_topology == "ring":
            for i in range(n):
                src = (i - 1) % n
                self._inject_emigrants(islands[i], emigrants[src], rng)
        else:  # fully_connected
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self._inject_emigrants(islands[i], emigrants[j], rng)

    @staticmethod
    def _inject_emigrants(
        island: _Island,
        immigrants: List[Individual],
        rng: np.random.Generator,
    ) -> None:
        """Replace worst individuals on island with immigrants."""
        if not immigrants:
            return
        pop = island.population
        # Sort worst-first (highest rank, lowest crowding)
        worst_indices = sorted(
            range(len(pop)),
            key=lambda i: (-pop[i].rank, pop[i].crowding_distance),
        )
        for k, immigrant in enumerate(immigrants):
            if k < len(worst_indices):
                # Deep copy to avoid cross-island aliasing
                clone = Individual(
                    parameters=immigrant.parameters.copy(),
                    generation=immigrant.generation,
                )
                clone.objectives = immigrant.objectives.copy() if immigrant.objectives is not None else None  # type: ignore[assignment]
                clone.is_feasible = immigrant.is_feasible
                clone.rank = immigrant.rank
                clone.crowding_distance = immigrant.crowding_distance
                clone.constraints_violated = immigrant.constraints_violated
                clone.constraint_penalty = immigrant.constraint_penalty
                pop[worst_indices[k]] = clone

    def _build_result(
        self,
        islands: List[_Island],
        merged_convergence: List[float],
        total_evals: int,
        n_migrations: int,
        t0: float,
    ) -> DistributedOptimizationResult:
        """Merge islands into a single DistributedOptimizationResult."""
        result = DistributedOptimizationResult(
            n_islands=len(islands),
            n_migrations=n_migrations,
            total_evaluations=total_evals,
            wall_clock_seconds=time.monotonic() - t0,
        )

        # Per-island results
        for island in islands:
            ir = OptimizationResult()
            fronts = _fast_non_dominated_sort(island.population)
            pareto_inds = [
                island.population[i] for i in fronts[0]
                if island.population[i].is_feasible
            ]
            ir.pareto_front = ParetoFront(
                individuals=pareto_inds,
                objective_names=[o.name for o in self._objectives],
                parameter_names=[b.name for b in self._bounds],
            )
            weights = np.array([o.weight for o in self._objectives])
            ir.best_compromise = ir.pareto_front.best_compromise(weights)
            ir.n_generations = island.generation
            ir.convergence_history = island.convergence
            result.island_results.append(ir)

        # Merge all populations into one combined front
        all_pop = [ind for island in islands for ind in island.population]
        fronts = _fast_non_dominated_sort(all_pop)
        merged_pareto = [
            all_pop[i] for i in fronts[0] if all_pop[i].is_feasible
        ]

        result.combined = OptimizationResult()
        result.combined.pareto_front = ParetoFront(
            individuals=merged_pareto,
            objective_names=[o.name for o in self._objectives],
            parameter_names=[b.name for b in self._bounds],
        )
        weights = np.array([o.weight for o in self._objectives])
        result.combined.best_compromise = result.combined.pareto_front.best_compromise(
            weights,
        )
        result.combined.n_generations = self._n_gen
        result.combined.n_evaluations = total_evals
        result.combined.convergence_history = merged_convergence
        result.combined.wall_clock_seconds = result.wall_clock_seconds

        return result
