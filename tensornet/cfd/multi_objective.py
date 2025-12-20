"""
Multi-Objective Optimization for Aerodynamic Design
====================================================

Extends the single-objective optimization framework to handle
multiple competing objectives common in hypersonic design:

    - Minimize drag coefficient C_D
    - Minimize peak heat flux q_max
    - Maximize lift-to-drag ratio L/D
    - Minimize weight (volume constraints)

Key Concepts:
    - Pareto optimality: No improvement in one objective without
      degrading another
    - Pareto front: Set of all Pareto-optimal solutions
    - Dominance: Solution A dominates B if A is better in at least
      one objective and no worse in all others

Algorithms:
    1. Weighted Sum Method - Simple but uneven Pareto sampling
    2. ε-Constraint Method - Systematic Pareto exploration
    3. NSGA-II - Evolutionary multi-objective optimization
    4. Reference Point Method - User-specified aspiration levels

References:
    [1] Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm:
        NSGA-II", IEEE Trans. Evol. Comput. 6(2), 2002
    [2] Miettinen, "Nonlinear Multiobjective Optimization", Kluwer, 1999
    [3] Obayashi & Sasaki, "Visualization and Data Mining of Pareto 
        Solutions Using Self-Organizing Map", PPSN VIII, 2004
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable, List, Union
from enum import Enum
import random


class MOOAlgorithm(Enum):
    """Multi-objective optimization algorithms."""
    WEIGHTED_SUM = "weighted-sum"
    EPSILON_CONSTRAINT = "epsilon-constraint"
    NSGA_II = "nsga-ii"
    REFERENCE_POINT = "reference-point"


@dataclass
class ObjectiveSpec:
    """Specification for an objective function."""
    name: str
    function: Callable[[torch.Tensor], float]
    gradient: Callable[[torch.Tensor], torch.Tensor]
    minimize: bool = True
    weight: float = 1.0
    reference: Optional[float] = None  # For reference point method
    utopia: Optional[float] = None     # Best achievable value
    nadir: Optional[float] = None      # Worst Pareto value


@dataclass
class ParetoSolution:
    """A single solution on the Pareto front."""
    design: torch.Tensor
    objectives: Dict[str, float]
    rank: int = 0                      # Pareto rank (0 = non-dominated)
    crowding_distance: float = 0.0     # Diversity measure


@dataclass
class MOOResult:
    """Result from multi-objective optimization."""
    pareto_front: List[ParetoSolution]
    hypervolume: float
    n_generations: int
    n_evaluations: int
    utopia_point: Dict[str, float]
    nadir_point: Dict[str, float]


@dataclass
class MOOConfig:
    """Configuration for multi-objective optimization."""
    algorithm: MOOAlgorithm = MOOAlgorithm.NSGA_II
    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    mutation_strength: float = 0.1
    n_weight_samples: int = 21         # For weighted sum
    epsilon_steps: int = 10            # For ε-constraint
    reference_point: Optional[Dict[str, float]] = None


def dominates(obj_a: Dict[str, float], obj_b: Dict[str, float], 
              minimize: Dict[str, bool]) -> bool:
    """
    Check if solution A dominates solution B.
    
    A dominates B if:
        - A is no worse than B in all objectives
        - A is strictly better than B in at least one objective
    
    Args:
        obj_a, obj_b: Objective values for solutions A and B
        minimize: Dict indicating which objectives to minimize
        
    Returns:
        True if A dominates B
    """
    dominated_in_all = True
    better_in_one = False
    
    for name, val_a in obj_a.items():
        val_b = obj_b[name]
        is_min = minimize.get(name, True)
        
        if is_min:
            if val_a > val_b:
                dominated_in_all = False
            if val_a < val_b:
                better_in_one = True
        else:
            if val_a < val_b:
                dominated_in_all = False
            if val_a > val_b:
                better_in_one = True
    
    return dominated_in_all and better_in_one


def fast_non_dominated_sort(
    population: List[ParetoSolution],
    minimize: Dict[str, bool]
) -> List[List[int]]:
    """
    Fast non-dominated sorting (NSGA-II).
    
    Assigns each solution to a Pareto front (rank).
    Rank 0 = non-dominated, Rank 1 = dominated by rank 0, etc.
    
    Args:
        population: List of solutions
        minimize: Minimization flags per objective
        
    Returns:
        List of fronts, each containing indices of solutions
    """
    n = len(population)
    domination_count = [0] * n  # Number of solutions that dominate i
    dominated_set = [[] for _ in range(n)]  # Solutions that i dominates
    
    fronts = [[]]
    
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(population[i].objectives, population[j].objectives, minimize):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(population[j].objectives, population[i].objectives, minimize):
                dominated_set[j].append(i)
                domination_count[i] += 1
    
    # First front: non-dominated solutions
    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    
    # Handle edge case: if no non-dominated solutions found
    if not fronts[0]:
        # All solutions dominate each other? Shouldn't happen but be safe
        fronts[0] = list(range(n))
        return fronts
    
    # Subsequent fronts
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    
    return fronts


def crowding_distance(
    population: List[ParetoSolution],
    front_indices: List[int],
    minimize: Dict[str, bool]
) -> None:
    """
    Compute crowding distance for solutions in a front.
    
    Measures the density of solutions around each point.
    Higher distance = more isolated = better diversity.
    
    Args:
        population: Full population
        front_indices: Indices of solutions in this front
        minimize: Minimization flags (not used directly)
    """
    n = len(front_indices)
    if n == 0:
        return
    
    # Initialize distances
    for i in front_indices:
        population[i].crowding_distance = 0.0
    
    # Get objective names
    obj_names = list(population[front_indices[0]].objectives.keys())
    
    for obj_name in obj_names:
        # Sort by this objective
        sorted_indices = sorted(front_indices, 
                               key=lambda i: population[i].objectives[obj_name])
        
        # Boundary points get infinite distance
        population[sorted_indices[0]].crowding_distance = float('inf')
        population[sorted_indices[-1]].crowding_distance = float('inf')
        
        # Get objective range
        f_min = population[sorted_indices[0]].objectives[obj_name]
        f_max = population[sorted_indices[-1]].objectives[obj_name]
        
        if f_max - f_min < 1e-30:
            continue
        
        # Compute distances for interior points
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            prev_val = population[sorted_indices[i - 1]].objectives[obj_name]
            next_val = population[sorted_indices[i + 1]].objectives[obj_name]
            
            population[idx].crowding_distance += (next_val - prev_val) / (f_max - f_min)


def hypervolume_2d(
    pareto_front: List[ParetoSolution],
    reference: Dict[str, float],
    obj_names: Tuple[str, str]
) -> float:
    """
    Compute hypervolume indicator for 2D Pareto front.
    
    The hypervolume is the area dominated by the Pareto front
    and bounded by the reference point.
    
    Args:
        pareto_front: List of Pareto-optimal solutions
        reference: Reference point (worst acceptable values)
        obj_names: Tuple of (obj1_name, obj2_name)
        
    Returns:
        Hypervolume value
    """
    obj1, obj2 = obj_names
    
    # Sort by first objective
    sorted_front = sorted(pareto_front, key=lambda s: s.objectives[obj1])
    
    hv = 0.0
    prev_obj2 = reference[obj2]
    
    for sol in sorted_front:
        f1 = sol.objectives[obj1]
        f2 = sol.objectives[obj2]
        
        if f1 < reference[obj1] and f2 < prev_obj2:
            hv += (reference[obj1] - f1) * (prev_obj2 - f2)
            prev_obj2 = f2
    
    return hv


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization driver.
    
    Finds Pareto-optimal designs trading off multiple objectives.
    """
    
    def __init__(
        self,
        objectives: List[ObjectiveSpec],
        bounds: Tuple[torch.Tensor, torch.Tensor],
        config: MOOConfig = None
    ):
        """
        Args:
            objectives: List of objective specifications
            bounds: (lower, upper) bounds for design variables
            config: Optimization configuration
        """
        self.objectives = objectives
        self.lower_bounds = bounds[0]
        self.upper_bounds = bounds[1]
        self.n_vars = len(bounds[0])
        self.config = config or MOOConfig()
        
        self.minimize = {obj.name: obj.minimize for obj in objectives}
        self.n_evaluations = 0
    
    def evaluate(self, design: torch.Tensor) -> Dict[str, float]:
        """Evaluate all objectives for a design."""
        self.n_evaluations += 1
        return {obj.name: obj.function(design) for obj in self.objectives}
    
    def random_design(self) -> torch.Tensor:
        """Generate a random design within bounds."""
        return self.lower_bounds + torch.rand(self.n_vars) * (self.upper_bounds - self.lower_bounds)
    
    def initialize_population(self) -> List[ParetoSolution]:
        """Create initial random population."""
        population = []
        for _ in range(self.config.population_size):
            design = self.random_design()
            objectives = self.evaluate(design)
            population.append(ParetoSolution(design=design, objectives=objectives))
        return population
    
    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulated Binary Crossover (SBX).
        
        Creates two offspring from two parents with distribution
        controlled by η_c parameter.
        """
        eta_c = 20.0  # Distribution index
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        for i in range(self.n_vars):
            if random.random() < 0.5:
                continue
            
            if abs(parent1[i] - parent2[i]) < 1e-14:
                continue
            
            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])
            
            lb = self.lower_bounds[i]
            ub = self.upper_bounds[i]
            
            rand = random.random()
            
            # Compute β
            beta = 1.0 + 2.0 * (y1 - lb) / (y2 - y1 + 1e-30)
            alpha = 2.0 - beta ** (-(eta_c + 1))
            
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
            
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            
            # Other child
            beta = 1.0 + 2.0 * (ub - y2) / (y2 - y1 + 1e-30)
            alpha = 2.0 - beta ** (-(eta_c + 1))
            
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
            
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            
            # Bounds
            child1[i] = max(lb, min(ub, c1))
            child2[i] = max(lb, min(ub, c2))
        
        return child1, child2
    
    def mutate(self, design: torch.Tensor) -> torch.Tensor:
        """
        Polynomial mutation.
        """
        eta_m = 20.0  # Distribution index
        mutant = design.clone()
        
        for i in range(self.n_vars):
            if random.random() > self.config.mutation_prob:
                continue
            
            y = design[i]
            lb = self.lower_bounds[i]
            ub = self.upper_bounds[i]
            
            delta1 = (y - lb) / (ub - lb + 1e-30)
            delta2 = (ub - y) / (ub - lb + 1e-30)
            
            rand = random.random()
            
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                deltaq = val ** (1.0 / (eta_m + 1)) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                deltaq = 1.0 - val ** (1.0 / (eta_m + 1))
            
            y_new = y + deltaq * (ub - lb)
            mutant[i] = max(lb, min(ub, y_new))
        
        return mutant
    
    def tournament_selection(
        self,
        population: List[ParetoSolution]
    ) -> ParetoSolution:
        """
        Binary tournament selection based on rank and crowding distance.
        """
        idx1 = random.randint(0, len(population) - 1)
        idx2 = random.randint(0, len(population) - 1)
        
        sol1 = population[idx1]
        sol2 = population[idx2]
        
        # Prefer lower rank
        if sol1.rank < sol2.rank:
            return sol1
        elif sol2.rank < sol1.rank:
            return sol2
        # Same rank: prefer higher crowding distance
        elif sol1.crowding_distance > sol2.crowding_distance:
            return sol1
        else:
            return sol2
    
    def run_nsga2(self) -> MOOResult:
        """
        Run NSGA-II algorithm.
        
        Returns:
            MOOResult with Pareto front and metrics
        """
        config = self.config
        
        # Initialize population
        population = self.initialize_population()
        
        # Compute ranks and crowding distances
        fronts = fast_non_dominated_sort(population, self.minimize)
        for front in fronts:
            crowding_distance(population, front, self.minimize)
        
        for generation in range(config.n_generations):
            # Create offspring
            offspring = []
            
            while len(offspring) < config.population_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                if random.random() < config.crossover_prob:
                    child1, child2 = self.crossover(parent1.design, parent2.design)
                else:
                    child1, child2 = parent1.design.clone(), parent2.design.clone()
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Evaluate
                obj1 = self.evaluate(child1)
                obj2 = self.evaluate(child2)
                
                offspring.append(ParetoSolution(design=child1, objectives=obj1))
                offspring.append(ParetoSolution(design=child2, objectives=obj2))
            
            # Combine parent and offspring
            combined = population + offspring[:config.population_size]
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined, self.minimize)
            
            # Select next generation
            new_population = []
            front_idx = 0
            
            while len(new_population) + len(fronts[front_idx]) <= config.population_size:
                crowding_distance(combined, fronts[front_idx], self.minimize)
                for i in fronts[front_idx]:
                    new_population.append(combined[i])
                front_idx += 1
                if front_idx >= len(fronts):
                    break
            
            # Fill remaining slots with best from next front
            if len(new_population) < config.population_size and front_idx < len(fronts):
                crowding_distance(combined, fronts[front_idx], self.minimize)
                remaining = sorted(fronts[front_idx], 
                                  key=lambda i: combined[i].crowding_distance, 
                                  reverse=True)
                for i in remaining:
                    if len(new_population) >= config.population_size:
                        break
                    new_population.append(combined[i])
            
            population = new_population
        
        # Extract Pareto front
        fronts = fast_non_dominated_sort(population, self.minimize)
        pareto_front = [population[i] for i in fronts[0]]
        
        # Compute utopia and nadir points
        obj_names = [obj.name for obj in self.objectives]
        utopia = {name: min(s.objectives[name] for s in pareto_front) for name in obj_names}
        nadir = {name: max(s.objectives[name] for s in pareto_front) for name in obj_names}
        
        # Compute hypervolume (2D only)
        if len(obj_names) == 2:
            ref_point = {name: nadir[name] * 1.1 for name in obj_names}
            hv = hypervolume_2d(pareto_front, ref_point, tuple(obj_names))
        else:
            hv = 0.0  # TODO: n-D hypervolume
        
        return MOOResult(
            pareto_front=pareto_front,
            hypervolume=hv,
            n_generations=config.n_generations,
            n_evaluations=self.n_evaluations,
            utopia_point=utopia,
            nadir_point=nadir
        )
    
    def run_weighted_sum(self) -> MOOResult:
        """
        Run weighted sum scalarization.
        
        Samples different weight combinations to approximate Pareto front.
        Note: Can miss concave parts of the front.
        """
        from tensornet.cfd.optimization import ShapeOptimizer, OptimizationConfig
        
        config = self.config
        n_weights = config.n_weight_samples
        
        pareto_front = []
        
        # Generate weight combinations (for 2 objectives)
        weights_list = []
        for i in range(n_weights):
            w1 = i / (n_weights - 1)
            w2 = 1.0 - w1
            weights_list.append({self.objectives[0].name: w1, 
                                self.objectives[1].name: w2})
        
        for weights in weights_list:
            # Create weighted objective
            def weighted_obj(x):
                vals = self.evaluate(x)
                return sum(weights[name] * vals[name] for name in weights)
            
            def weighted_grad(x):
                grad = torch.zeros_like(x)
                for obj in self.objectives:
                    grad += weights[obj.name] * obj.gradient(x)
                return grad
            
            # Simple gradient descent
            x = self.random_design()
            lr = 0.01
            
            for _ in range(100):
                g = weighted_grad(x)
                x = x - lr * g
                x = torch.max(torch.min(x, self.upper_bounds), self.lower_bounds)
            
            objectives = self.evaluate(x)
            pareto_front.append(ParetoSolution(design=x, objectives=objectives))
        
        # Remove dominated solutions
        non_dominated = []
        for i, sol in enumerate(pareto_front):
            is_dominated = False
            for j, other in enumerate(pareto_front):
                if i != j and dominates(other.objectives, sol.objectives, self.minimize):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(sol)
        
        obj_names = [obj.name for obj in self.objectives]
        utopia = {name: min(s.objectives[name] for s in non_dominated) for name in obj_names}
        nadir = {name: max(s.objectives[name] for s in non_dominated) for name in obj_names}
        
        return MOOResult(
            pareto_front=non_dominated,
            hypervolume=0.0,
            n_generations=1,
            n_evaluations=self.n_evaluations,
            utopia_point=utopia,
            nadir_point=nadir
        )
    
    def optimize(self) -> MOOResult:
        """Run optimization with configured algorithm."""
        if self.config.algorithm == MOOAlgorithm.NSGA_II:
            return self.run_nsga2()
        elif self.config.algorithm == MOOAlgorithm.WEIGHTED_SUM:
            return self.run_weighted_sum()
        else:
            raise NotImplementedError(f"Algorithm {self.config.algorithm} not implemented")


def create_drag_heating_problem(n_vars: int = 10) -> Tuple[List[ObjectiveSpec], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a test bi-objective problem: minimize drag and heating.
    
    Uses simplified analytical objectives for testing.
    
    Returns:
        (objectives, bounds)
    """
    # Simple test functions
    def drag_objective(x: torch.Tensor) -> float:
        # Sphere-like function
        return (x ** 2).sum().item()
    
    def heating_objective(x: torch.Tensor) -> float:
        # Shifted sphere
        return ((x - 0.5) ** 2).sum().item() + 0.5
    
    def drag_gradient(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x
    
    def heating_gradient(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - 0.5)
    
    objectives = [
        ObjectiveSpec(
            name="drag",
            function=drag_objective,
            gradient=drag_gradient,
            minimize=True
        ),
        ObjectiveSpec(
            name="heating",
            function=heating_objective,
            gradient=heating_gradient,
            minimize=True
        )
    ]
    
    bounds = (
        torch.zeros(n_vars, dtype=torch.float64),
        torch.ones(n_vars, dtype=torch.float64)
    )
    
    return objectives, bounds


def validate_moo():
    """Run validation tests for multi-objective optimization."""
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE OPTIMIZATION VALIDATION")
    print("=" * 70)
    
    # Test 1: Dominance relation
    print("\n[Test 1] Dominance Relation")
    print("-" * 40)
    
    obj_a = {"f1": 1.0, "f2": 2.0}
    obj_b = {"f1": 2.0, "f2": 3.0}
    obj_c = {"f1": 1.5, "f2": 1.5}
    minimize = {"f1": True, "f2": True}
    
    print(f"A = {obj_a}")
    print(f"B = {obj_b}")
    print(f"C = {obj_c}")
    
    assert dominates(obj_a, obj_b, minimize) == True
    print("A dominates B: ✓")
    
    assert dominates(obj_a, obj_c, minimize) == False
    print("A dominates C: False ✓")
    
    assert dominates(obj_c, obj_a, minimize) == False
    print("C dominates A: False ✓")
    
    print("✓ PASS")
    
    # Test 2: Non-dominated sorting
    print("\n[Test 2] Non-Dominated Sorting")
    print("-" * 40)
    
    population = [
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 1.0, "f2": 3.0}),
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 2.0, "f2": 2.0}),
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 3.0, "f2": 1.0}),
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 2.5, "f2": 2.5}),
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 4.0, "f2": 4.0}),
    ]
    
    fronts = fast_non_dominated_sort(population, minimize)
    
    print(f"Front 0 (non-dominated): {fronts[0]}")
    print(f"Front 1: {fronts[1] if len(fronts) > 1 else 'empty'}")
    
    # First three should be non-dominated
    assert set(fronts[0]) == {0, 1, 2}
    print("✓ PASS")
    
    # Test 3: Crowding distance
    print("\n[Test 3] Crowding Distance")
    print("-" * 40)
    
    crowding_distance(population, fronts[0], minimize)
    
    for i in fronts[0]:
        print(f"Solution {i}: cd = {population[i].crowding_distance:.2f}")
    
    # Boundary solutions should have infinite distance
    assert population[0].crowding_distance == float('inf')
    assert population[2].crowding_distance == float('inf')
    print("✓ PASS")
    
    # Test 4: NSGA-II on test problem
    print("\n[Test 4] NSGA-II Optimization")
    print("-" * 40)
    
    objectives, bounds = create_drag_heating_problem(n_vars=5)
    
    config = MOOConfig(
        algorithm=MOOAlgorithm.NSGA_II,
        population_size=20,
        n_generations=10
    )
    
    optimizer = MultiObjectiveOptimizer(objectives, bounds, config)
    result = optimizer.optimize()
    
    print(f"Pareto front size: {len(result.pareto_front)}")
    print(f"Evaluations: {result.n_evaluations}")
    print(f"Utopia point: {result.utopia_point}")
    print(f"Nadir point: {result.nadir_point}")
    
    assert len(result.pareto_front) > 0
    print("✓ PASS")
    
    # Test 5: 2D Hypervolume
    print("\n[Test 5] Hypervolume Indicator")
    print("-" * 40)
    
    # Simple 2D front
    front = [
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 1.0, "f2": 3.0}),
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 2.0, "f2": 2.0}),
        ParetoSolution(design=torch.zeros(2), objectives={"f1": 3.0, "f2": 1.0}),
    ]
    
    reference = {"f1": 4.0, "f2": 4.0}
    hv = hypervolume_2d(front, reference, ("f1", "f2"))
    
    print(f"Hypervolume: {hv:.2f}")
    # Expected: (4-1)*(4-3) + (4-2)*(3-2) + (4-3)*(2-1) = 3 + 2 + 1 = 6
    assert abs(hv - 6.0) < 0.01
    print("✓ PASS")
    
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE OPTIMIZATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    validate_moo()
