"""
Optimization Toolkit — Gradient-based and topology optimization.

Provides optimization interfaces built on the adjoint sensitivity engine,
including gradient-based parameter optimization and topology optimization
with density-based penalization (SIMP).

Classes:
    ObjectiveFunction       — Wrapper combining cost + constraints.
    ConstrainedOptimizer    — Augmented-Lagrangian constrained optimization.
    TopologyOptimization    — SIMP-based topology optimization.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ontic.platform.adjoint import AdjointSolver, CostFunctional, SensitivityResult
from ontic.platform.data_model import FieldData, SimulationState, StructuredMesh
from ontic.platform.inverse import InverseProblem, InverseProblemResult, TikhonovRegularizer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Constraint:
    """
    An optimization constraint: g(params) <= 0.

    Attributes
    ----------
    name : str
        Constraint name.
    evaluate : Callable
        g(params) → scalar Tensor.  Feasible when <= 0.
    gradient : Callable
        dg/dp(params) → dict of Tensor gradients.
    """

    name: str
    evaluate: Callable[[Dict[str, Tensor]], Tensor]
    gradient: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


def volume_fraction_constraint(
    density_name: str, max_fraction: float
) -> Constraint:
    """
    Volume fraction constraint: mean(ρ) - max_fraction ≤ 0.

    Used in topology optimization to limit material usage.
    """
    def _eval(params: Dict[str, Tensor]) -> Tensor:
        rho = params[density_name]
        return rho.mean() - max_fraction

    def _grad(params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        rho = params[density_name]
        n = rho.numel()
        return {density_name: torch.ones_like(rho) / n}

    return Constraint(
        name=f"vol_frac({density_name})<={max_fraction}",
        evaluate=_eval,
        gradient=_grad,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Augmented-Lagrangian Constrained Optimizer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class OptimizationResult:
    """Result from constrained optimization."""

    optimal_params: Dict[str, Tensor]
    cost_history: List[float]
    constraint_history: Dict[str, List[float]]
    converged: bool
    iterations: int
    elapsed_seconds: float


class ConstrainedOptimizer:
    """
    Augmented-Lagrangian method for constrained optimization.

    Handles inequality constraints g_i(x) <= 0 by introducing
    Lagrange multipliers and a penalty parameter that is increased
    each outer iteration.

    Parameters
    ----------
    max_outer : int
        Maximum outer (penalty update) iterations.
    max_inner : int
        Maximum inner (gradient descent) iterations per outer step.
    learning_rate : float
        Step size for inner gradient steps.
    penalty_init : float
        Initial penalty parameter μ.
    penalty_growth : float
        Factor by which μ is multiplied each outer iteration.
    tolerance : float
        Convergence tolerance on KKT residual.
    """

    def __init__(
        self,
        max_outer: int = 20,
        max_inner: int = 50,
        learning_rate: float = 1e-2,
        penalty_init: float = 1.0,
        penalty_growth: float = 2.0,
        tolerance: float = 1e-5,
    ) -> None:
        self._max_outer = max_outer
        self._max_inner = max_inner
        self._lr = learning_rate
        self._mu = penalty_init
        self._mu_growth = penalty_growth
        self._tol = tolerance

    def solve(
        self,
        problem: InverseProblem,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        initial_params: Dict[str, Tensor],
        constraints: List[Constraint],
    ) -> OptimizationResult:
        """Run augmented-Lagrangian optimization."""
        t0 = time.perf_counter()
        params = {k: v.clone().detach().requires_grad_(True) for k, v in initial_params.items()}

        # Initialize Lagrange multipliers
        lambdas = [torch.tensor(0.0, dtype=torch.float64) for _ in constraints]
        mu = self._mu

        cost_history: List[float] = []
        constraint_history: Dict[str, List[float]] = {c.name: [] for c in constraints}
        converged = False
        total_iter = 0

        for outer in range(self._max_outer):
            # Inner loop: minimize augmented Lagrangian
            for inner in range(self._max_inner):
                # Forward cost + gradient
                cost, grads = problem.total_gradient(initial_state, t_span, dt, params)

                # Add constraint terms
                for i, con in enumerate(constraints):
                    g = con.evaluate(params)
                    g_val = g.item()
                    g_grads = con.gradient(params)

                    # Augmented Lagrangian: λ·g + (μ/2)·max(0, g + λ/μ)²
                    shifted = g_val + lambdas[i].item() / mu
                    if shifted > 0:
                        penalty_grad_scale = mu * shifted
                        for k in grads:
                            if k in g_grads:
                                grads[k] = grads[k] + penalty_grad_scale * g_grads[k]
                        cost += 0.5 * mu * shifted ** 2

                cost_history.append(cost)
                for i, con in enumerate(constraints):
                    constraint_history[con.name].append(con.evaluate(params).item())

                # Gradient step
                grad_norm = sum(g.norm().item() ** 2 for g in grads.values()) ** 0.5
                if grad_norm < self._tol:
                    converged = True
                    break

                with torch.no_grad():
                    for k in params:
                        if k in grads:
                            params[k] -= self._lr * grads[k]
                            params[k].requires_grad_(True)

                total_iter += 1

            if converged:
                break

            # Update multipliers and penalty
            with torch.no_grad():
                for i, con in enumerate(constraints):
                    g = con.evaluate(params).item()
                    lambdas[i] = torch.tensor(
                        max(0, lambdas[i].item() + mu * g),
                        dtype=torch.float64,
                    )
            mu *= self._mu_growth

        elapsed = time.perf_counter() - t0
        return OptimizationResult(
            optimal_params={k: v.detach() for k, v in params.items()},
            cost_history=cost_history,
            constraint_history=constraint_history,
            converged=converged,
            iterations=total_iter,
            elapsed_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Topology Optimization (SIMP)
# ═══════════════════════════════════════════════════════════════════════════════


class TopologyOptimization:
    """
    SIMP (Solid Isotropic Material with Penalization) topology optimization.

    Optimizes a density field ρ ∈ [0, 1] to minimize compliance (or a
    user-defined objective) subject to a volume fraction constraint.

    The density field modulates material properties:
        E(ρ) = ρ^p · E_0

    where p is the SIMP penalization exponent (typically 3).

    Parameters
    ----------
    solver : Any
        Forward physics solver.
    cost : CostFunctional
        Physics-based objective (e.g., compliance).
    penalization : float
        SIMP exponent p (default 3).
    volume_fraction : float
        Maximum allowed material volume fraction.
    filter_radius : float
        Density filter radius for regularization (0 = no filter).
    max_iterations : int
        Maximum OC (optimality criteria) iterations.
    tolerance : float
        Convergence tolerance on density change.
    """

    def __init__(
        self,
        solver: Any,
        cost: CostFunctional,
        penalization: float = 3.0,
        volume_fraction: float = 0.5,
        filter_radius: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
    ) -> None:
        self._solver = solver
        self._cost = cost
        self._p = penalization
        self._vf = volume_fraction
        self._filter_radius = filter_radius
        self._max_iter = max_iterations
        self._tol = tolerance

    def optimize(
        self,
        base_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        density_field_name: str = "density",
    ) -> Dict[str, Any]:
        """
        Run SIMP topology optimization.

        Returns dict with: optimal_density, cost_history, volume_history,
        converged, iterations.
        """
        t0 = time.perf_counter()

        # Initialize uniform density at volume fraction
        mesh = base_state.mesh
        n_cells = mesh.n_cells
        rho = torch.full((n_cells,), self._vf, dtype=torch.float64, requires_grad=True)

        cost_history: List[float] = []
        volume_history: List[float] = []
        converged = False

        for iteration in range(self._max_iter):
            # Penalized density
            rho_penalized = rho ** self._p

            # Inject density into state
            density_field = FieldData(
                name=density_field_name,
                data=rho_penalized.detach(),
                mesh=mesh,
            )
            modified_state = base_state.with_fields(**{density_field_name: density_field})

            # Forward solve
            result = self._solver.solve(modified_state, t_span, dt)
            final = result.final_state

            # Evaluate cost
            J = self._cost.evaluate(final, {density_field_name: rho})
            cost_history.append(J.item())
            volume_history.append(rho.mean().item())

            # Sensitivity
            if rho.grad is not None:
                rho.grad.zero_()
            J.backward()
            sensitivity = rho.grad.clone() if rho.grad is not None else torch.zeros_like(rho)

            # Density filter (smoothing)
            if self._filter_radius > 0 and isinstance(mesh, StructuredMesh) and mesh.ndim == 1:
                sensitivity = self._apply_filter_1d(sensitivity, mesh, self._filter_radius)

            # Optimality Criteria (OC) update
            rho_old = rho.detach().clone()
            rho_new = self._oc_update(rho.detach(), sensitivity, self._vf)

            # Convergence check
            change = (rho_new - rho_old).abs().max().item()
            rho = rho_new.requires_grad_(True)

            if change < self._tol and iteration > 5:
                converged = True
                break

        elapsed = time.perf_counter() - t0
        return {
            "optimal_density": rho.detach(),
            "cost_history": cost_history,
            "volume_history": volume_history,
            "converged": converged,
            "iterations": iteration + 1,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _oc_update(
        rho: Tensor, dc: Tensor, vf: float, move: float = 0.2
    ) -> Tensor:
        """
        Optimality Criteria (OC) density update with volume constraint.

        Bisection on Lagrange multiplier to enforce volume fraction.
        """
        lo = 1e-9
        hi = 1e9
        dc_neg = dc.abs() + 1e-12  # Ensure positive for OC

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            rho_new = rho * (dc_neg / mid).sqrt()
            rho_new = torch.clamp(rho_new, rho - move, rho + move)
            rho_new = torch.clamp(rho_new, 0.001, 1.0)

            if rho_new.mean().item() > vf:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break

        return rho_new

    @staticmethod
    def _apply_filter_1d(
        sensitivity: Tensor, mesh: StructuredMesh, radius: float
    ) -> Tensor:
        """Simple 1-D spatial filter (averaging within radius)."""
        N = sensitivity.shape[0]
        dx = mesh.dx[0]
        n_cells = int(radius / dx)
        if n_cells < 1:
            return sensitivity

        filtered = sensitivity.clone()
        for i in range(N):
            lo = max(0, i - n_cells)
            hi = min(N, i + n_cells + 1)
            filtered[i] = sensitivity[lo:hi].mean()
        return filtered
