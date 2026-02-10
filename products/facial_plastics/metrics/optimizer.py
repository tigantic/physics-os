"""Multi-objective surgical plan optimization.

Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II)
to optimize surgical plan parameters with competing objectives:

  - Maximize aesthetic score (profile, symmetry, proportions)
  - Maximize functional score (nasal resistance improvement, airflow)
  - Maximize safety index (stress/strain within limits)
  - Minimize tissue disruption (total displacement, removal volume)
  - Minimize healing time (edema, scar formation)

Constraints:
  - Hard safety limits (non-negotiable)
  - Anatomical feasibility (e.g., can't remove more than available)
  - Surgical technique constraints (min/max parameter ranges)

The optimizer operates on the plan parameter space defined by
the plan DSL, evaluating each candidate plan through the full
simulation pipeline.
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Objective and constraint definitions ─────────────────────────

@dataclass
class ObjectiveSpec:
    """Specification of an optimization objective."""
    name: str
    direction: str = "maximize"  # "maximize" or "minimize"
    weight: float = 1.0
    min_acceptable: float = float("-inf")
    max_acceptable: float = float("inf")
    unit: str = ""

    @property
    def is_maximize(self) -> bool:
        return self.direction == "maximize"


@dataclass
class ConstraintSpec:
    """Specification of a hard constraint."""
    name: str
    min_value: float = float("-inf")
    max_value: float = float("inf")
    is_hard: bool = True  # hard = infeasible if violated
    penalty_weight: float = 100.0  # for soft constraints

    def evaluate(self, value: float) -> Tuple[bool, float]:
        """Return (is_feasible, violation_amount)."""
        if value < self.min_value:
            return (not self.is_hard, self.min_value - value)
        if value > self.max_value:
            return (not self.is_hard, value - self.max_value)
        return (True, 0.0)


@dataclass
class ParameterBound:
    """Bounds for an optimizable plan parameter."""
    name: str
    low: float
    high: float
    step: Optional[float] = None  # discrete step size if applicable
    is_integer: bool = False

    def clip(self, value: float) -> float:
        v = np.clip(value, self.low, self.high)
        if self.is_integer:
            v = round(v)
        if self.step is not None:
            v = self.low + round((v - self.low) / self.step) * self.step
        return float(v)


# ── Standard objectives for rhinoplasty ──────────────────────────

def default_rhinoplasty_objectives() -> List[ObjectiveSpec]:
    return [
        ObjectiveSpec(
            name="aesthetic_score",
            direction="maximize",
            weight=1.0,
            min_acceptable=50.0,
            unit="",
        ),
        ObjectiveSpec(
            name="functional_score",
            direction="maximize",
            weight=1.0,
            min_acceptable=60.0,
            unit="",
        ),
        ObjectiveSpec(
            name="safety_index",
            direction="maximize",
            weight=1.5,
            min_acceptable=70.0,
            unit="",
        ),
    ]


def default_rhinoplasty_constraints() -> List[ConstraintSpec]:
    return [
        ConstraintSpec(
            name="safety_index",
            min_value=60.0,
            is_hard=True,
        ),
        ConstraintSpec(
            name="max_skin_tension_pa",
            max_value=100.0e3,
            is_hard=True,
        ),
        ConstraintSpec(
            name="max_principal_strain",
            max_value=0.30,
            is_hard=True,
        ),
        ConstraintSpec(
            name="nasal_resistance",
            min_value=0.5,
            max_value=3.0,
            is_hard=False,
            penalty_weight=50.0,
        ),
    ]


# ── Individual solution ──────────────────────────────────────────

@dataclass
class Individual:
    """A candidate solution in the optimization."""
    parameters: np.ndarray        # (n_params,) decision variables
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    constraints_violated: int = 0
    constraint_penalty: float = 0.0
    is_feasible: bool = True
    rank: int = 0                 # Pareto front rank (0 = non-dominated)
    crowding_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    uid: str = ""

    def __post_init__(self) -> None:
        if not self.uid:
            h = hashlib.md5(self.parameters.tobytes()).hexdigest()[:8]
            self.uid = f"IND-{h}"


@dataclass
class ParetoFront:
    """The Pareto-optimal set of solutions."""
    individuals: List[Individual] = field(default_factory=list)
    objective_names: List[str] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.individuals)

    def objectives_matrix(self) -> np.ndarray:
        """Return (N, M) matrix of objective values."""
        if not self.individuals:
            return np.array([])
        return np.array([ind.objectives for ind in self.individuals])

    def parameters_matrix(self) -> np.ndarray:
        """Return (N, D) matrix of parameter values."""
        if not self.individuals:
            return np.array([])
        return np.array([ind.parameters for ind in self.individuals])

    def best_compromise(self, weights: Optional[np.ndarray] = None) -> Optional[Individual]:
        """Return the compromise solution using weighted sum."""
        if not self.individuals:
            return None

        obj_mat = self.objectives_matrix()
        if obj_mat.size == 0:
            return None

        # Normalize objectives to [0, 1]
        mins = obj_mat.min(axis=0)
        maxs = obj_mat.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-12] = 1.0
        norm = (obj_mat - mins) / ranges

        if weights is None:
            weights = np.ones(obj_mat.shape[1]) / obj_mat.shape[1]

        scores = norm @ weights
        return self.individuals[int(np.argmax(scores))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_solutions": self.size,
            "objective_names": self.objective_names,
            "parameter_names": self.parameter_names,
            "solutions": [
                {
                    "uid": ind.uid,
                    "parameters": ind.parameters.tolist(),
                    "objectives": ind.objectives.tolist(),
                    "is_feasible": ind.is_feasible,
                    "rank": ind.rank,
                }
                for ind in self.individuals
            ],
        }


# ── Optimization result ──────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Complete optimization result."""
    pareto_front: ParetoFront = field(default_factory=ParetoFront)
    best_compromise: Optional[Individual] = None
    n_generations: int = 0
    n_evaluations: int = 0
    convergence_history: List[float] = field(default_factory=list)  # hypervolume
    wall_clock_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"Optimization: {self.n_generations} generations, "
            f"{self.n_evaluations} evaluations, "
            f"Pareto front size={self.pareto_front.size}, "
            f"time={self.wall_clock_seconds:.1f}s"
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "n_generations": self.n_generations,
            "n_evaluations": self.n_evaluations,
            "wall_clock_seconds": self.wall_clock_seconds,
            "pareto_front": self.pareto_front.to_dict(),
            "convergence_history": self.convergence_history,
        }
        if self.best_compromise:
            result["best_compromise"] = {
                "uid": self.best_compromise.uid,
                "parameters": self.best_compromise.parameters.tolist(),
                "objectives": self.best_compromise.objectives.tolist(),
            }
        return result


# ── NSGA-II core ─────────────────────────────────────────────────

def _fast_non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """Fast non-dominated sorting (Deb et al. 2002).

    Returns list of fronts, each a list of indices into population.
    Front 0 = Pareto-optimal set.
    """
    n = len(population)
    domination_count = [0] * n
    dominated_set: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(population[p], population[q]):
                dominated_set[p].append(q)
            elif _dominates(population[q], population[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            population[p].rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _dominates(p: Individual, q: Individual) -> bool:
    """Check if p dominates q (all objectives at least as good, one better).

    Infeasible solutions are always dominated by feasible ones.
    Among infeasible, fewer constraint violations dominate.
    """
    if p.is_feasible and not q.is_feasible:
        return True
    if not p.is_feasible and q.is_feasible:
        return False
    if not p.is_feasible and not q.is_feasible:
        return p.constraint_penalty < q.constraint_penalty

    at_least_one_better = False
    for i in range(len(p.objectives)):
        if p.objectives[i] < q.objectives[i]:
            return False
        if p.objectives[i] > q.objectives[i]:
            at_least_one_better = True
    return at_least_one_better


def _crowding_distance(population: List[Individual], front: List[int]) -> None:
    """Assign crowding distance to individuals in a front."""
    n = len(front)
    if n <= 2:
        for idx in front:
            population[idx].crowding_distance = float("inf")
        return

    for idx in front:
        population[idx].crowding_distance = 0.0

    n_obj = len(population[front[0]].objectives)
    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: population[i].objectives[m])
        population[sorted_front[0]].crowding_distance = float("inf")
        population[sorted_front[-1]].crowding_distance = float("inf")

        f_range = (
            population[sorted_front[-1]].objectives[m]
            - population[sorted_front[0]].objectives[m]
        )
        if f_range < 1e-12:
            continue

        for k in range(1, n - 1):
            population[sorted_front[k]].crowding_distance += (
                population[sorted_front[k + 1]].objectives[m]
                - population[sorted_front[k - 1]].objectives[m]
            ) / f_range


def _tournament_select(
    population: List[Individual],
    rng: np.random.Generator,
) -> Individual:
    """Binary tournament selection."""
    i, j = rng.integers(0, len(population), size=2)
    a, b = population[i], population[j]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding_distance > b.crowding_distance:
        return a
    return b


def _sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    bounds: List[ParameterBound],
    rng: np.random.Generator,
    eta: float = 20.0,
    crossover_prob: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)."""
    c1 = p1.copy()
    c2 = p2.copy()

    if rng.random() > crossover_prob:
        return c1, c2

    for i in range(len(bounds)):
        if rng.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue

        lo, hi = bounds[i].low, bounds[i].high
        if lo >= hi:
            continue

        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

        c1[i] = bounds[i].clip(c1[i])
        c2[i] = bounds[i].clip(c2[i])

    return c1, c2


def _polynomial_mutation(
    x: np.ndarray,
    bounds: List[ParameterBound],
    rng: np.random.Generator,
    eta: float = 20.0,
    mutation_prob: Optional[float] = None,
) -> np.ndarray:
    """Polynomial mutation."""
    result = x.copy()
    if mutation_prob is None:
        mutation_prob = 1.0 / max(len(bounds), 1)

    for i in range(len(bounds)):
        if rng.random() > mutation_prob:
            continue

        lo, hi = bounds[i].low, bounds[i].high
        if lo >= hi:
            continue

        delta = hi - lo
        u = rng.random()
        if u < 0.5:
            deltaq = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            deltaq = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

        result[i] = x[i] + deltaq * delta
        result[i] = bounds[i].clip(result[i])

    return result


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """2D hypervolume indicator (for convergence tracking).

    For >2 objectives, uses the first two.
    """
    if len(points) == 0:
        return 0.0

    pts = points[:, :2] if points.shape[1] > 2 else points
    # Filter dominated by reference
    valid = np.all(pts < ref[:2], axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    # Sort by first objective descending
    order = np.argsort(-pts[:, 0])
    pts = pts[order]

    hv = 0.0
    prev_y = ref[1]
    for i in range(len(pts)):
        hv += (ref[0] - pts[i, 0]) * (prev_y - pts[i, 1])
        prev_y = min(prev_y, pts[i, 1])

    return float(hv)


# ── Main optimizer ───────────────────────────────────────────────

class PlanOptimizer:
    """NSGA-II multi-objective optimizer for surgical plans.

    Optimizes plan parameters to find the Pareto-optimal set
    of solutions balancing aesthetic, functional, and safety
    objectives.
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
    ) -> None:
        if not objectives:
            raise ValueError("At least one objective required")
        if not parameter_bounds:
            raise ValueError("At least one parameter bound required")

        self._objectives = objectives
        self._constraints = constraints
        self._bounds = parameter_bounds
        self._pop_size = population_size
        self._n_gen = n_generations
        self._cx_eta = crossover_eta
        self._mut_eta = mutation_eta
        self._cx_prob = crossover_prob
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Precompute sign flip for maximization → minimization
        self._obj_signs = np.array([
            1.0 if o.is_maximize else -1.0
            for o in objectives
        ])

    def optimize(
        self,
        evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
        *,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> OptimizationResult:
        """Run NSGA-II optimization.

        Args:
            evaluate_fn: Takes parameter array (n_params,), returns
                        (objectives array, constraint_values dict).
            progress_callback: Called with (generation, total, hypervolume).

        Returns:
            OptimizationResult with Pareto front and best compromise.
        """
        import time
        t0 = time.monotonic()

        n_obj = len(self._objectives)
        n_params = len(self._bounds)
        result = OptimizationResult()

        # Initialize population
        population = self._initialize_population()

        # Evaluate initial population
        self._evaluate_population(population, evaluate_fn)

        convergence: List[float] = []

        for gen in range(self._n_gen):
            # Non-dominated sorting
            fronts = _fast_non_dominated_sort(population)

            # Crowding distance
            for front in fronts:
                _crowding_distance(population, front)

            # Generate offspring
            offspring: List[Individual] = []
            while len(offspring) < self._pop_size:
                p1 = _tournament_select(population, self._rng)
                p2 = _tournament_select(population, self._rng)
                c1_params, c2_params = _sbx_crossover(
                    p1.parameters, p2.parameters,
                    self._bounds, self._rng,
                    eta=self._cx_eta,
                    crossover_prob=self._cx_prob,
                )
                c1_params = _polynomial_mutation(
                    c1_params, self._bounds, self._rng, eta=self._mut_eta,
                )
                c2_params = _polynomial_mutation(
                    c2_params, self._bounds, self._rng, eta=self._mut_eta,
                )
                offspring.append(Individual(
                    parameters=c1_params, generation=gen + 1,
                ))
                if len(offspring) < self._pop_size:
                    offspring.append(Individual(
                        parameters=c2_params, generation=gen + 1,
                    ))

            # Evaluate offspring
            self._evaluate_population(offspring, evaluate_fn)

            # Combine parent + offspring
            combined = population + offspring

            # Non-dominated sorting on combined
            fronts = _fast_non_dominated_sort(combined)
            for front in fronts:
                _crowding_distance(combined, front)

            # Select next generation
            population = []
            for front in fronts:
                if len(population) + len(front) <= self._pop_size:
                    population.extend([combined[i] for i in front])
                else:
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(
                        front,
                        key=lambda i: combined[i].crowding_distance,
                        reverse=True,
                    )
                    remaining = self._pop_size - len(population)
                    population.extend([combined[i] for i in sorted_front[:remaining]])
                    break

            # Convergence tracking (hypervolume)
            feasible = [
                ind for ind in population
                if ind.is_feasible and ind.rank == 0
            ]
            if feasible:
                obj_mat = np.array([ind.objectives for ind in feasible])
                ref = np.max(obj_mat, axis=0) + 1.0
                hv = _hypervolume_2d(obj_mat, ref)
                convergence.append(hv)
            else:
                convergence.append(0.0)

            if progress_callback:
                progress_callback(gen + 1, self._n_gen, convergence[-1])

            result.n_evaluations = (gen + 1) * self._pop_size + self._pop_size

        # Extract Pareto front
        fronts = _fast_non_dominated_sort(population)
        pareto_individuals = [population[i] for i in fronts[0] if population[i].is_feasible]

        result.pareto_front = ParetoFront(
            individuals=pareto_individuals,
            objective_names=[o.name for o in self._objectives],
            parameter_names=[b.name for b in self._bounds],
        )

        # Best compromise
        result.best_compromise = result.pareto_front.best_compromise(
            np.array([o.weight for o in self._objectives])
        )

        result.n_generations = self._n_gen
        result.convergence_history = convergence
        result.wall_clock_seconds = time.monotonic() - t0

        logger.info("Optimization complete: %s", result.summary())
        return result

    def _initialize_population(self) -> List[Individual]:
        """Create initial population via LHS."""
        n = self._pop_size
        d = len(self._bounds)
        population: List[Individual] = []

        # Latin Hypercube for initial diversity
        for j in range(d):
            cuts = np.linspace(0, 1, n + 1)
            u = np.array([
                self._rng.uniform(cuts[i], cuts[i + 1]) for i in range(n)
            ])
            self._rng.shuffle(u)
            if not population:
                population = [
                    Individual(parameters=np.zeros(d), generation=0)
                    for _ in range(n)
                ]
            for i in range(n):
                b = self._bounds[j]
                population[i].parameters[j] = b.low + u[i] * (b.high - b.low)
                population[i].parameters[j] = b.clip(population[i].parameters[j])

        return population

    def _evaluate_population(
        self,
        population: List[Individual],
        evaluate_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
    ) -> None:
        """Evaluate all individuals in the population."""
        for ind in population:
            try:
                obj_raw, constraint_values = evaluate_fn(ind.parameters)

                # Apply sign convention (maximize → positive internally)
                ind.objectives = obj_raw * self._obj_signs

                # Check constraints
                ind.constraints_violated = 0
                ind.constraint_penalty = 0.0
                for cs in self._constraints:
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
                ind.objectives = np.full(len(self._objectives), -1e10)
                ind.is_feasible = False
                ind.constraints_violated = len(self._constraints)
                ind.constraint_penalty = 1e10
