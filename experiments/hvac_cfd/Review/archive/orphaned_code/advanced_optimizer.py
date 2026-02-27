"""
TigantiCFD Advanced Optimization
================================

Adjoint-based optimization and multi-objective Pareto optimization.

Capabilities:
- T4.01: Adjoint sensitivity computation
- T4.02: Gradient-based optimization
- T4.03: Multi-objective Pareto optimization
- T4.04: NSGA-II algorithm implementation

Reference:
    Deb, K. (2002). "Multi-Objective Optimization using Evolutionary 
    Algorithms." Wiley.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict
import numpy as np
from enum import Enum


@dataclass
class DesignVariable:
    """Definition of a design variable for optimization."""
    name: str
    lower_bound: float
    upper_bound: float
    initial: Optional[float] = None
    
    def __post_init__(self):
        if self.initial is None:
            self.initial = (self.lower_bound + self.upper_bound) / 2


@dataclass
class Objective:
    """Definition of an optimization objective."""
    name: str
    minimize: bool = True  # True for minimize, False for maximize
    weight: float = 1.0    # For weighted sum methods


@dataclass
class OptimizationResult:
    """Result of single-objective optimization."""
    optimal_x: np.ndarray
    optimal_value: float
    n_iterations: int
    n_evaluations: int
    convergence_history: List[float]
    gradient_norm_history: List[float]
    success: bool
    message: str


@dataclass
class ParetoSolution:
    """A solution on the Pareto front."""
    x: np.ndarray           # Design variables
    objectives: np.ndarray  # Objective values
    rank: int = 0           # Pareto rank (0 = non-dominated)
    crowding_distance: float = 0.0


@dataclass
class ParetoFront:
    """Collection of Pareto-optimal solutions."""
    solutions: List[ParetoSolution]
    
    def get_front(self, rank: int = 0) -> List[ParetoSolution]:
        """Get solutions at specified Pareto rank."""
        return [s for s in self.solutions if s.rank == rank]
    
    def to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (design_vars, objectives) as arrays."""
        X = np.array([s.x for s in self.solutions])
        F = np.array([s.objectives for s in self.solutions])
        return X, F


class AdjointSolver:
    """
    Discrete adjoint solver for sensitivity computation.
    
    Computes gradients of an objective function with respect to
    design variables using the adjoint method, which is efficient
    when the number of objectives << number of design variables.
    
    For CFD:
        ∂J/∂α = ∂J/∂u × ∂u/∂α + ∂J/∂α|_explicit
        
    where the adjoint variable λ satisfies:
        (∂R/∂u)ᵀ λ = (∂J/∂u)ᵀ
    """
    
    def __init__(
        self,
        n_design_vars: int,
        n_state_vars: int,
        compute_residual: Callable[[np.ndarray, np.ndarray], np.ndarray],
        compute_objective: Callable[[np.ndarray, np.ndarray], float]
    ):
        """
        Initialize adjoint solver.
        
        Args:
            n_design_vars: Number of design variables α
            n_state_vars: Number of state variables u
            compute_residual: R(α, u) -> residual vector
            compute_objective: J(α, u) -> scalar objective
        """
        self.n_alpha = n_design_vars
        self.n_u = n_state_vars
        self.R = compute_residual
        self.J = compute_objective
        
        self.eps = 1e-7  # Finite difference step
    
    def compute_jacobian_fd(
        self,
        alpha: np.ndarray,
        u: np.ndarray,
        wrt: str = "u"
    ) -> np.ndarray:
        """
        Compute Jacobian using finite differences.
        
        Args:
            alpha: Design variables
            u: State variables
            wrt: Differentiate with respect to "u" or "alpha"
            
        Returns:
            Jacobian matrix ∂R/∂wrt
        """
        R0 = self.R(alpha, u)
        n_R = len(R0)
        
        if wrt == "u":
            n_var = self.n_u
            jac = np.zeros((n_R, n_var))
            for i in range(n_var):
                u_pert = u.copy()
                u_pert[i] += self.eps
                R_pert = self.R(alpha, u_pert)
                jac[:, i] = (R_pert - R0) / self.eps
        else:
            n_var = self.n_alpha
            jac = np.zeros((n_R, n_var))
            for i in range(n_var):
                alpha_pert = alpha.copy()
                alpha_pert[i] += self.eps
                R_pert = self.R(alpha_pert, u)
                jac[:, i] = (R_pert - R0) / self.eps
        
        return jac
    
    def compute_objective_gradient_fd(
        self,
        alpha: np.ndarray,
        u: np.ndarray,
        wrt: str = "u"
    ) -> np.ndarray:
        """Compute objective gradient using finite differences."""
        J0 = self.J(alpha, u)
        
        if wrt == "u":
            grad = np.zeros(self.n_u)
            for i in range(self.n_u):
                u_pert = u.copy()
                u_pert[i] += self.eps
                grad[i] = (self.J(alpha, u_pert) - J0) / self.eps
        else:
            grad = np.zeros(self.n_alpha)
            for i in range(self.n_alpha):
                alpha_pert = alpha.copy()
                alpha_pert[i] += self.eps
                grad[i] = (self.J(alpha_pert, u) - J0) / self.eps
        
        return grad
    
    def solve_adjoint(
        self,
        alpha: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        Solve the adjoint equation: (∂R/∂u)ᵀ λ = (∂J/∂u)ᵀ
        
        Returns:
            Adjoint variable λ
        """
        # Compute Jacobians
        dRdu = self.compute_jacobian_fd(alpha, u, wrt="u")
        dJdu = self.compute_objective_gradient_fd(alpha, u, wrt="u")
        
        # Solve adjoint system
        # (∂R/∂u)ᵀ λ = (∂J/∂u)ᵀ
        try:
            adjoint = np.linalg.solve(dRdu.T, dJdu)
        except np.linalg.LinAlgError:
            # Use least squares if singular
            adjoint, _, _, _ = np.linalg.lstsq(dRdu.T, dJdu, rcond=None)
        
        return adjoint
    
    def compute_total_gradient(
        self,
        alpha: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        Compute total gradient dJ/dα using adjoint method.
        
        dJ/dα = ∂J/∂α - λᵀ ∂R/∂α
        """
        # Solve adjoint
        adjoint = self.solve_adjoint(alpha, u)
        
        # Partial derivatives
        dJdalpha = self.compute_objective_gradient_fd(alpha, u, wrt="alpha")
        dRdalpha = self.compute_jacobian_fd(alpha, u, wrt="alpha")
        
        # Total gradient
        total_grad = dJdalpha - adjoint @ dRdalpha
        
        return total_grad


class GradientOptimizer:
    """
    Gradient-based optimizer using adjoint sensitivities.
    """
    
    def __init__(
        self,
        variables: List[DesignVariable],
        objective: Callable[[np.ndarray], float],
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        method: str = "BFGS"
    ):
        self.variables = variables
        self.objective = objective
        self.gradient = gradient
        self.method = method
        
        self.n_vars = len(variables)
        self.bounds = [(v.lower_bound, v.upper_bound) for v in variables]
        
    def optimize(
        self,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> OptimizationResult:
        """
        Run gradient-based optimization.
        """
        if x0 is None:
            x0 = np.array([v.initial for v in self.variables])
        
        history = []
        grad_history = []
        n_evals = [0]
        
        def callback_obj(x):
            n_evals[0] += 1
            val = self.objective(x)
            history.append(val)
            return val
        
        x = x0.copy()
        best_x = x.copy()
        best_val = callback_obj(x)
        
        # Simple gradient descent with backtracking line search
        for iteration in range(max_iter):
            # Compute gradient
            if self.gradient is not None:
                grad = self.gradient(x)
            else:
                # Finite difference
                grad = self._fd_gradient(x)
            
            grad_norm = np.linalg.norm(grad)
            grad_history.append(grad_norm)
            
            if grad_norm < tol:
                break
            
            # Backtracking line search
            alpha = 1.0
            c1 = 1e-4
            
            for _ in range(20):
                x_new = x - alpha * grad
                # Project to bounds
                x_new = np.clip(x_new, 
                               [b[0] for b in self.bounds],
                               [b[1] for b in self.bounds])
                
                val_new = callback_obj(x_new)
                
                if val_new < best_val - c1 * alpha * grad_norm**2:
                    x = x_new
                    best_val = val_new
                    best_x = x.copy()
                    break
                
                alpha *= 0.5
        
        return OptimizationResult(
            optimal_x=best_x,
            optimal_value=best_val,
            n_iterations=iteration + 1,
            n_evaluations=n_evals[0],
            convergence_history=history,
            gradient_norm_history=grad_history,
            success=grad_history[-1] < tol if grad_history else False,
            message="Optimization complete"
        )
    
    def _fd_gradient(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Finite difference gradient."""
        grad = np.zeros(self.n_vars)
        f0 = self.objective(x)
        
        for i in range(self.n_vars):
            x_pert = x.copy()
            x_pert[i] += eps
            grad[i] = (self.objective(x_pert) - f0) / eps
        
        return grad


class NSGAII:
    """
    NSGA-II Multi-Objective Optimization Algorithm.
    
    Non-dominated Sorting Genetic Algorithm II for Pareto optimization.
    
    Reference:
        Deb et al. (2002). "A Fast and Elitist Multiobjective Genetic 
        Algorithm: NSGA-II"
    """
    
    def __init__(
        self,
        variables: List[DesignVariable],
        objectives: List[Callable[[np.ndarray], float]],
        pop_size: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1
    ):
        self.variables = variables
        self.objectives = objectives
        self.n_vars = len(variables)
        self.n_obj = len(objectives)
        self.pop_size = pop_size
        self.p_cross = crossover_prob
        self.p_mut = mutation_prob
        
        self.bounds = np.array([(v.lower_bound, v.upper_bound) 
                                for v in variables])
    
    def optimize(
        self,
        n_generations: int = 100,
        seed: Optional[int] = None
    ) -> ParetoFront:
        """
        Run NSGA-II optimization.
        
        Returns:
            ParetoFront containing non-dominated solutions
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population
        population = self._initialize_population()
        
        for gen in range(n_generations):
            # Evaluate objectives
            fitness = self._evaluate(population)
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(fitness)
            
            # Crowding distance
            for front in fronts:
                self._crowding_distance(front, fitness)
            
            # Selection, crossover, mutation
            offspring = self._create_offspring(population, fitness, fronts)
            
            # Combine parent and offspring
            combined = np.vstack([population, offspring])
            combined_fitness = self._evaluate(combined)
            
            # Select next generation
            population = self._select_next_gen(combined, combined_fitness)
        
        # Final evaluation
        fitness = self._evaluate(population)
        fronts = self._fast_non_dominated_sort(fitness)
        
        # Build Pareto front
        solutions = []
        for i, (x, f) in enumerate(zip(population, fitness)):
            rank = next(r for r, front in enumerate(fronts) if i in front)
            solutions.append(ParetoSolution(
                x=x.copy(),
                objectives=f.copy(),
                rank=rank
            ))
        
        return ParetoFront(solutions=solutions)
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        pop = np.random.random((self.pop_size, self.n_vars))
        
        # Scale to bounds
        for i in range(self.n_vars):
            pop[:, i] = (self.bounds[i, 0] + 
                        pop[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]))
        
        return pop
    
    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all objectives for population."""
        fitness = np.zeros((len(population), self.n_obj))
        
        for i, x in enumerate(population):
            for j, obj in enumerate(self.objectives):
                fitness[i, j] = obj(x)
        
        return fitness
    
    def _fast_non_dominated_sort(
        self,
        fitness: np.ndarray
    ) -> List[List[int]]:
        """Fast non-dominated sorting algorithm."""
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        # First front
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Subsequent fronts
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, f1: np.ndarray, f2: np.ndarray) -> bool:
        """Check if f1 dominates f2 (all objectives better or equal, at least one strictly better)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)
    
    def _crowding_distance(
        self,
        front: List[int],
        fitness: np.ndarray
    ) -> None:
        """Compute crowding distance for a front."""
        if len(front) <= 2:
            return
        
        n = len(front)
        distances = np.zeros(n)
        
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_idx = sorted(range(n), key=lambda i: fitness[front[i], m])
            
            # Boundary points get infinite distance
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            # Normalize
            f_max = fitness[front[sorted_idx[-1]], m]
            f_min = fitness[front[sorted_idx[0]], m]
            
            if f_max - f_min > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        (fitness[front[sorted_idx[i + 1]], m] - 
                         fitness[front[sorted_idx[i - 1]], m]) /
                        (f_max - f_min)
                    )
    
    def _create_offspring(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> np.ndarray:
        """Create offspring through selection, crossover, mutation."""
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Tournament selection
            p1 = self._tournament_select(population, fitness, fronts)
            p2 = self._tournament_select(population, fitness, fronts)
            
            # Crossover (SBX)
            if np.random.random() < self.p_cross:
                c1, c2 = self._sbx_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation (polynomial)
            if np.random.random() < self.p_mut:
                c1 = self._polynomial_mutation(c1)
            if np.random.random() < self.p_mut:
                c2 = self._polynomial_mutation(c2)
            
            offspring.extend([c1, c2])
        
        return np.array(offspring[:self.pop_size])
    
    def _tournament_select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> np.ndarray:
        """Binary tournament selection."""
        i1, i2 = np.random.choice(len(population), 2, replace=False)
        
        # Get ranks
        rank1 = next((r for r, f in enumerate(fronts) if i1 in f), len(fronts))
        rank2 = next((r for r, f in enumerate(fronts) if i2 in f), len(fronts))
        
        if rank1 < rank2:
            return population[i1]
        elif rank2 < rank1:
            return population[i2]
        else:
            return population[i1] if np.random.random() < 0.5 else population[i2]
    
    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        eta: float = 15.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover."""
        c1, c2 = p1.copy(), p2.copy()
        
        for i in range(self.n_vars):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    # Bound check
                    c1[i] = np.clip(c1[i], self.bounds[i, 0], self.bounds[i, 1])
                    c2[i] = np.clip(c2[i], self.bounds[i, 0], self.bounds[i, 1])
        
        return c1, c2
    
    def _polynomial_mutation(
        self,
        x: np.ndarray,
        eta: float = 20.0
    ) -> np.ndarray:
        """Polynomial mutation."""
        y = x.copy()
        
        for i in range(self.n_vars):
            if np.random.random() < 1.0 / self.n_vars:
                delta = self.bounds[i, 1] - self.bounds[i, 0]
                u = np.random.random()
                
                if u < 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                y[i] = x[i] + delta_q * delta
                y[i] = np.clip(y[i], self.bounds[i, 0], self.bounds[i, 1])
        
        return y
    
    def _select_next_gen(
        self,
        combined: np.ndarray,
        fitness: np.ndarray
    ) -> np.ndarray:
        """Select next generation from combined parent+offspring."""
        fronts = self._fast_non_dominated_sort(fitness)
        
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop.extend(front)
            else:
                # Need to select subset based on crowding distance
                remaining = self.pop_size - len(new_pop)
                self._crowding_distance(front, fitness)
                
                # Sort by crowding distance (descending)
                sorted_front = sorted(
                    front,
                    key=lambda i: getattr(self, '_crowd_dist', {}).get(i, 0),
                    reverse=True
                )
                new_pop.extend(sorted_front[:remaining])
                break
        
        return combined[new_pop]
