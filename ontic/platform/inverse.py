"""
Inverse Problem Toolkit — Regularization and Bayesian wrappers.

Provides framework-level support for inverse problems (parameter estimation,
data assimilation, reconstruction) built on the adjoint interface.

Classes:
    InverseProblem          — Base class for inverse problem definitions.
    TikhonovRegularizer     — L2 (Tikhonov) regularization.
    TVRegularizer           — Total variation regularization.
    GradientDescentSolver   — First-order optimization for inverse problems.
    LBFGSSolver             — L-BFGS quasi-Newton inverse solver.
    BayesianInversion       — Approximate Bayesian posterior via Laplace.
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

from ontic.platform.adjoint import (
    AdjointSolver,
    CostFunctional,
    L2TrackingCost,
    SensitivityResult,
)
from ontic.platform.data_model import SimulationState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Regularizers
# ═══════════════════════════════════════════════════════════════════════════════


class Regularizer(ABC):
    """Base class for regularization terms."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, params: Dict[str, Tensor]) -> Tensor:
        """Compute R(params) → scalar."""
        ...

    @abstractmethod
    def gradient(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute ∂R/∂p for each parameter."""
        ...


class TikhonovRegularizer(Regularizer):
    """
    L2 (Tikhonov) regularization: R = 0.5 * α * || p - p_prior ||².

    Parameters
    ----------
    alpha : float
        Regularization strength.
    prior : dict, optional
        Prior parameter values.  Defaults to zero.
    """

    def __init__(
        self, alpha: float = 1e-3, prior: Optional[Dict[str, Tensor]] = None
    ) -> None:
        self._alpha = alpha
        self._prior = prior or {}

    @property
    def name(self) -> str:
        return f"Tikhonov(α={self._alpha})"

    def evaluate(self, params: Dict[str, Tensor]) -> Tensor:
        total = torch.tensor(0.0, dtype=torch.float64)
        for name, p in params.items():
            p0 = self._prior.get(name, torch.zeros_like(p))
            total = total + 0.5 * self._alpha * ((p - p0) ** 2).sum()
        return total

    def gradient(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        grads: Dict[str, Tensor] = {}
        for name, p in params.items():
            p0 = self._prior.get(name, torch.zeros_like(p))
            grads[name] = self._alpha * (p - p0)
        return grads


class TVRegularizer(Regularizer):
    """
    Total Variation regularization: R = α * Σ |p[i+1] - p[i]|.

    Promotes piecewise-constant parameter distributions.
    Uses a smooth approximation: |x| ≈ sqrt(x² + ε).
    """

    def __init__(self, alpha: float = 1e-3, epsilon: float = 1e-8) -> None:
        self._alpha = alpha
        self._epsilon = epsilon

    @property
    def name(self) -> str:
        return f"TV(α={self._alpha})"

    def evaluate(self, params: Dict[str, Tensor]) -> Tensor:
        total = torch.tensor(0.0, dtype=torch.float64)
        for p in params.values():
            if p.dim() >= 1 and p.shape[0] > 1:
                diff = p[1:] - p[:-1]
                total = total + self._alpha * torch.sqrt(diff ** 2 + self._epsilon).sum()
        return total

    def gradient(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        grads: Dict[str, Tensor] = {}
        for name, p in params.items():
            g = torch.zeros_like(p)
            if p.dim() >= 1 and p.shape[0] > 1:
                diff = p[1:] - p[:-1]
                denom = torch.sqrt(diff ** 2 + self._epsilon)
                signed = self._alpha * diff / denom
                g[1:] += signed
                g[:-1] -= signed
            grads[name] = g
        return grads


# ═══════════════════════════════════════════════════════════════════════════════
# Inverse Problem Definition
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InverseProblemResult:
    """Result from an inverse problem solve."""

    optimal_params: Dict[str, Tensor]
    cost_history: List[float]
    gradient_norm_history: List[float]
    converged: bool
    iterations: int
    elapsed_seconds: float


class InverseProblem:
    """
    A regularized inverse problem: minimize J(u(p), d) + R(p).

    Combines a forward solver, cost functional, regularizer, and
    adjoint gradient computation into a single optimization problem.

    Parameters
    ----------
    forward_solver : Any
        Solver-protocol solver.
    cost : CostFunctional
        Data-misfit term.
    regularizer : Regularizer, optional
        Regularization term.
    """

    def __init__(
        self,
        forward_solver: Any,
        cost: CostFunctional,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        self._adjoint = AdjointSolver(forward_solver, cost)
        self._cost = cost
        self._regularizer = regularizer

    def total_cost(
        self,
        state: SimulationState,
        params: Dict[str, Tensor],
    ) -> float:
        """Evaluate J(state, params) + R(params)."""
        J = self._cost.evaluate(state, params).item()
        if self._regularizer is not None:
            J += self._regularizer.evaluate(params).item()
        return J

    def total_gradient(
        self,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        params: Dict[str, Tensor],
    ) -> Tuple[float, Dict[str, Tensor]]:
        """
        Compute total cost and gradient: dJ/dp + dR/dp.
        """
        sens = self._adjoint.compute_gradient(initial_state, t_span, dt, params)
        grads = dict(sens.gradients)

        total_cost = sens.cost_value
        if self._regularizer is not None:
            total_cost += self._regularizer.evaluate(params).item()
            reg_grads = self._regularizer.gradient(params)
            for k in grads:
                if k in reg_grads:
                    grads[k] = grads[k] + reg_grads[k]

        return total_cost, grads


# ═══════════════════════════════════════════════════════════════════════════════
# Inverse Solvers
# ═══════════════════════════════════════════════════════════════════════════════


class GradientDescentSolver:
    """
    Steepest-descent solver for inverse problems.

    Parameters
    ----------
    learning_rate : float
        Step size.
    max_iterations : int
        Maximum number of gradient steps.
    tolerance : float
        Convergence tolerance on gradient norm.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        self._lr = learning_rate
        self._max_iter = max_iterations
        self._tol = tolerance

    def solve(
        self,
        problem: InverseProblem,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        initial_params: Dict[str, Tensor],
    ) -> InverseProblemResult:
        """Run gradient descent to convergence."""
        t0 = time.perf_counter()
        params = {k: v.clone().detach().requires_grad_(True) for k, v in initial_params.items()}
        cost_history: List[float] = []
        grad_history: List[float] = []
        converged = False

        for iteration in range(self._max_iter):
            cost, grads = problem.total_gradient(initial_state, t_span, dt, params)
            cost_history.append(cost)

            grad_norm = sum(g.norm().item() ** 2 for g in grads.values()) ** 0.5
            grad_history.append(grad_norm)

            if grad_norm < self._tol:
                converged = True
                break

            # Update
            with torch.no_grad():
                for k in params:
                    if k in grads:
                        params[k] -= self._lr * grads[k]
                        params[k].requires_grad_(True)

        elapsed = time.perf_counter() - t0
        return InverseProblemResult(
            optimal_params={k: v.detach() for k, v in params.items()},
            cost_history=cost_history,
            gradient_norm_history=grad_history,
            converged=converged,
            iterations=len(cost_history),
            elapsed_seconds=elapsed,
        )


class LBFGSSolver:
    """
    L-BFGS quasi-Newton solver for inverse problems.

    Uses PyTorch's ``torch.optim.LBFGS`` internally.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        line_search: str = "strong_wolfe",
        tolerance: float = 1e-7,
    ) -> None:
        self._max_iter = max_iterations
        self._line_search = line_search
        self._tol = tolerance

    def solve(
        self,
        problem: InverseProblem,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        initial_params: Dict[str, Tensor],
    ) -> InverseProblemResult:
        """Run L-BFGS optimization."""
        t0 = time.perf_counter()
        params = {k: v.clone().detach().requires_grad_(True) for k, v in initial_params.items()}
        param_list = list(params.values())

        optimizer = torch.optim.LBFGS(
            param_list,
            max_iter=1,
            line_search_fn=self._line_search,
            tolerance_grad=self._tol,
        )

        cost_history: List[float] = []
        grad_history: List[float] = []
        converged = False

        for iteration in range(self._max_iter):
            def closure() -> Tensor:
                optimizer.zero_grad()
                cost, grads = problem.total_gradient(initial_state, t_span, dt, params)
                # Manually set gradients
                for k, p in params.items():
                    if k in grads and p.grad is None:
                        p.grad = grads[k].detach()
                    elif k in grads:
                        p.grad.copy_(grads[k].detach())
                return torch.tensor(cost, requires_grad=True)

            loss = optimizer.step(closure)
            cost_val = loss.item() if isinstance(loss, Tensor) else float(loss)
            cost_history.append(cost_val)

            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in param_list if p.grad is not None
            ) ** 0.5
            grad_history.append(grad_norm)

            if grad_norm < self._tol:
                converged = True
                break

        elapsed = time.perf_counter() - t0
        return InverseProblemResult(
            optimal_params={k: v.detach() for k, v in params.items()},
            cost_history=cost_history,
            gradient_norm_history=grad_history,
            converged=converged,
            iterations=len(cost_history),
            elapsed_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian Inversion (Laplace Approximation)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BayesianResult:
    """Result of Bayesian inversion."""

    map_params: Dict[str, Tensor]
    posterior_mean: Dict[str, Tensor]
    posterior_covariance: Dict[str, Tensor]
    log_evidence: float
    inverse_result: InverseProblemResult


class BayesianInversion:
    """
    Approximate Bayesian inversion via Laplace approximation.

    1. Find the MAP estimate via L-BFGS.
    2. Compute the Hessian at the MAP point via finite differences.
    3. Approximate the posterior as N(MAP, H⁻¹).

    Parameters
    ----------
    inverse_solver : LBFGSSolver or GradientDescentSolver
        Optimizer for the MAP estimate.
    hessian_eps : float
        Finite-difference step for Hessian approximation.
    """

    def __init__(
        self,
        inverse_solver: Optional[Any] = None,
        hessian_eps: float = 1e-4,
    ) -> None:
        self._solver = inverse_solver or LBFGSSolver()
        self._eps = hessian_eps

    def solve(
        self,
        problem: InverseProblem,
        initial_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        initial_params: Dict[str, Tensor],
    ) -> BayesianResult:
        """
        Compute the Laplace approximation to the posterior.
        """
        # Step 1: MAP estimate
        inv_result = self._solver.solve(problem, initial_state, t_span, dt, initial_params)
        map_params = inv_result.optimal_params

        # Step 2: Hessian via finite differences
        posterior_cov: Dict[str, Tensor] = {}
        for name, p_map in map_params.items():
            n = p_map.numel()
            H = torch.zeros(n, n, dtype=torch.float64)

            p_flat = p_map.flatten()
            for i in range(n):
                # Forward difference for Hessian diagonal
                p_plus = p_flat.clone()
                p_plus[i] += self._eps
                p_minus = p_flat.clone()
                p_minus[i] -= self._eps

                params_plus = dict(map_params)
                params_plus[name] = p_plus.reshape(p_map.shape).requires_grad_(True)
                _, g_plus = problem.total_gradient(initial_state, t_span, dt, params_plus)

                params_minus = dict(map_params)
                params_minus[name] = p_minus.reshape(p_map.shape).requires_grad_(True)
                _, g_minus = problem.total_gradient(initial_state, t_span, dt, params_minus)

                H[i, :] = (g_plus[name].flatten() - g_minus[name].flatten()) / (2 * self._eps)

            # Symmetrize and invert
            H = 0.5 * (H + H.T)
            H += 1e-6 * torch.eye(n, dtype=torch.float64)  # Regularize
            cov = torch.linalg.inv(H)
            posterior_cov[name] = cov

        # Log evidence (Laplace approximation)
        d = sum(p.numel() for p in map_params.values())
        log_det_H = sum(
            torch.linalg.slogdet(
                torch.linalg.inv(c) if c.shape[0] > 0 else torch.ones(1, 1)
            )[1].item()
            for c in posterior_cov.values()
        )
        log_evidence = (
            -inv_result.cost_history[-1]
            - 0.5 * log_det_H
            + 0.5 * d * math.log(2 * math.pi)
        )

        return BayesianResult(
            map_params=map_params,
            posterior_mean=map_params,
            posterior_covariance=posterior_cov,
            log_evidence=log_evidence,
            inverse_result=inv_result,
        )
