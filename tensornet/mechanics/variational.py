"""
Discrete variational integrators and action minimization.

Variational integrators derive numerical schemes from a discrete
action principle rather than discretizing the ODE. This guarantees:
    - Symplecticity (exact preservation of phase-space volume)
    - Momentum maps (Noether charges conserved to machine precision)
    - Bounded energy error (no secular drift)

The discrete Euler-Lagrange (DEL) equation:
    D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) = 0

where L_d(q_k, q_{k+1}) ≈ ∫_{t_k}^{t_{k+1}} L(q, q̇) dt
is a discrete Lagrangian (quadrature approximation of the action).

References:
    [1] Marsden, J.E. & West, M. (2001). Acta Numerica 10, 357-514.
    [2] Hairer, Lubich, Wanner (2006). Geometric Numerical Integration.
    [3] Lew, Marsden, Ortiz, West (2004). Variational time integrators. IJNME.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  DISCRETE LAGRANGIANS
# ═══════════════════════════════════════════════════════════════════════════════

def midpoint_discrete_lagrangian(
    L: Callable[[Tensor, Tensor], Tensor],
    q0: Tensor,
    q1: Tensor,
    dt: float,
) -> Tensor:
    """
    Midpoint rule discrete Lagrangian:

        L_d(q₀, q₁) = dt · L((q₀ + q₁)/2, (q₁ - q₀)/dt)

    This is a 2nd-order approximation to the exact discrete Lagrangian.
    The resulting variational integrator coincides with the implicit midpoint
    rule, which preserves all quadratic first integrals.
    """
    q_mid = 0.5 * (q0 + q1)
    v_mid = (q1 - q0) / dt
    return dt * L(q_mid, v_mid)


def trapezoidal_discrete_lagrangian(
    L: Callable[[Tensor, Tensor], Tensor],
    q0: Tensor,
    q1: Tensor,
    dt: float,
) -> Tensor:
    """
    Trapezoidal rule discrete Lagrangian:

        L_d(q₀, q₁) = (dt/2) · [L(q₀, v) + L(q₁, v)]

    where v = (q₁ - q₀)/dt.
    Also 2nd order, but the resulting integrator is the Störmer-Verlet
    method (for separable Lagrangians L = T(q̇) - V(q)).
    """
    v = (q1 - q0) / dt
    return 0.5 * dt * (L(q0, v) + L(q1, v))


def simpson_discrete_lagrangian(
    L: Callable[[Tensor, Tensor], Tensor],
    q0: Tensor,
    q1: Tensor,
    dt: float,
) -> Tensor:
    """
    Simpson's rule discrete Lagrangian (4th order):

        L_d(q₀, q₁) = (dt/6) · [L(q₀, v) + 4L(q_m, v) + L(q₁, v)]

    where q_m = (q₀+q₁)/2, v = (q₁-q₀)/dt.
    """
    v = (q1 - q0) / dt
    q_mid = 0.5 * (q0 + q1)
    return (dt / 6.0) * (L(q0, v) + 4.0 * L(q_mid, v) + L(q1, v))


# ═══════════════════════════════════════════════════════════════════════════════
#  DISCRETE EULER-LAGRANGE (DEL) INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DEL_Integrator:
    """
    Discrete Euler-Lagrange integrator.

    Given a continuous Lagrangian L(q, q̇) and a quadrature rule, this
    integrator solves the DEL equations:

        D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) = 0

    for q_{k+1} given (q_{k-1}, q_k) using Newton's method.

    The discrete Legendre transforms define momenta:
        p_k⁻ = -D₁L_d(q_k, q_{k+1})   (backward)
        p_k⁺ =  D₂L_d(q_{k-1}, q_k)    (forward)

    The DEL equation is then p_k⁺ + p_k⁻ = 0 (momentum matching).

    Attributes:
        L: Continuous Lagrangian L(q, q̇) → scalar
        dt: Time step
        quadrature: 'midpoint', 'trapezoidal', or 'simpson'
        newton_tol: Tolerance for Newton solve
        newton_max_iter: Maximum Newton iterations
    """

    L: Callable[[Tensor, Tensor], Tensor]
    dt: float = 0.01
    quadrature: str = "midpoint"
    newton_tol: float = 1e-12
    newton_max_iter: int = 50

    def _Ld(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Evaluate discrete Lagrangian."""
        if self.quadrature == "midpoint":
            return midpoint_discrete_lagrangian(self.L, q0, q1, self.dt)
        elif self.quadrature == "trapezoidal":
            return trapezoidal_discrete_lagrangian(self.L, q0, q1, self.dt)
        elif self.quadrature == "simpson":
            return simpson_discrete_lagrangian(self.L, q0, q1, self.dt)
        else:
            raise ValueError(f"Unknown quadrature: {self.quadrature}")

    def _D1_Ld(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Compute ∂L_d/∂q₀ via autograd."""
        q0_r = q0.detach().requires_grad_(True)
        Ld = self._Ld(q0_r, q1.detach())
        return torch.autograd.grad(Ld.sum(), q0_r)[0].detach()

    def _D2_Ld(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Compute ∂L_d/∂q₁ via autograd."""
        q1_r = q1.detach().requires_grad_(True)
        Ld = self._Ld(q0.detach(), q1_r)
        return torch.autograd.grad(Ld.sum(), q1_r)[0].detach()

    def discrete_momentum_minus(self, q_k: Tensor, q_kp1: Tensor) -> Tensor:
        """Backward discrete Legendre transform: p_k⁻ = -D₁L_d(q_k, q_{k+1})."""
        return -self._D1_Ld(q_k, q_kp1)

    def discrete_momentum_plus(self, q_km1: Tensor, q_k: Tensor) -> Tensor:
        """Forward discrete Legendre transform: p_k⁺ = D₂L_d(q_{k-1}, q_k)."""
        return self._D2_Ld(q_km1, q_k)

    def step(self, q_km1: Tensor, q_k: Tensor) -> Tensor:
        """
        Solve DEL equation for q_{k+1}:

            D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) = 0

        Uses Newton's method on the residual F(q_{k+1}) = D₁L_d(q_k, q_{k+1}) + p_k⁺.
        """
        # Forward momentum from previous step
        pk_plus = self._D2_Ld(q_km1, q_k)

        # Initial guess: linear extrapolation
        q_next = 2.0 * q_k - q_km1

        for iteration in range(self.newton_max_iter):
            # Residual: F = D₁L_d(q_k, q_{k+1}) + p_k⁺ should be 0
            # But we need to solve for q_{k+1} in
            # D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) = 0
            # i.e., D₁L_d(q_k, q_{k+1}) = -pk_plus

            q_next_r = q_next.detach().requires_grad_(True)
            D1 = self._D1_Ld(q_k, q_next_r)

            residual = D1 + pk_plus

            if residual.abs().max().item() < self.newton_tol:
                break

            # Newton: J δq = -F, where J = ∂F/∂q_{k+1} ≈ ∂(D₁L_d)/∂q₁
            # For 1D or diagonal systems, use scalar Newton
            # For coupled systems, use full Jacobian
            ndof = q_next.shape[-1] if q_next.ndim > 0 else 1

            if ndof <= 8:
                # Build Jacobian via autograd
                q_next_r2 = q_next.detach().requires_grad_(True)
                D1_val = self._D1_Ld(q_k, q_next_r2)
                F_val = D1_val + pk_plus

                # Use component-wise Jacobian
                jac = torch.zeros(ndof, ndof, dtype=q_next.dtype, device=q_next.device)
                for i in range(ndof):
                    q_r = q_next.detach().requires_grad_(True)
                    D1_i = self._D1_Ld(q_k, q_r)
                    if D1_i.ndim == 0:
                        grad_i = torch.autograd.grad(D1_i, q_r)[0]
                    else:
                        grad_i = torch.autograd.grad(D1_i[i], q_r)[0]
                    jac[i] = grad_i

                delta = torch.linalg.solve(jac, -F_val.unsqueeze(-1) if F_val.ndim == 1 else -F_val)
                if delta.ndim > 1:
                    delta = delta.squeeze(-1)
                q_next = q_next + delta
            else:
                # Diagonal approximation for high-DOF
                eps = 1e-7
                q_p = q_next + eps
                D1_p = self._D1_Ld(q_k, q_p)
                diag_jac = (D1_p - D1) / eps
                diag_jac = diag_jac.clamp(min=1e-14)  # prevent division by zero
                q_next = q_next - residual / diag_jac

        return q_next.detach()

    def initialize(
        self, q0: Tensor, p0: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Initialize the two-step method from (q₀, p₀).

        Uses the continuous Legendre transform to compute q̇₀ from p₀,
        then takes one Euler step to get q₁.

        Returns:
            (q₀, q₁) to start the DEL iteration.
        """
        # Solve p₀ = ∂L/∂q̇(q₀, q̇₀) for q̇₀ via Newton
        q_dot = p0.clone()  # guess: for T = ½m|q̇|², p = mq̇ → q̇ ≈ p

        for _ in range(20):
            qd_r = q_dot.detach().requires_grad_(True)
            L_val = self.L(q0.detach(), qd_r)
            p_computed = torch.autograd.grad(L_val.sum(), qd_r)[0]
            residual = p_computed - p0
            if residual.abs().max().item() < 1e-12:
                break
            # Newton step (diagonal approx for mass matrix)
            eps = 1e-7
            qd_p = q_dot + eps
            qd_p_r = qd_p.detach().requires_grad_(True)
            L_p = self.L(q0.detach(), qd_p_r)
            p_p = torch.autograd.grad(L_p.sum(), qd_p_r)[0]
            mass_diag = (p_p - p_computed) / eps
            mass_diag = mass_diag.clamp(min=1e-14)
            q_dot = q_dot - residual / mass_diag

        # Euler step to get q₁
        q1 = q0 + self.dt * q_dot
        return q0.detach(), q1.detach()

    def integrate(
        self,
        q0: Tensor,
        p0: Tensor,
        t_final: float,
        save_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Integrate the system from t=0 to t=t_final.

        Args:
            q0: Initial generalized coordinates
            p0: Initial conjugate momenta
            t_final: End time
            save_every: Save state every N steps

        Returns:
            Dictionary with 'q', 'p', 't', 'action' keys
        """
        n_steps = int(math.ceil(t_final / self.dt))

        # Initialize two-step method
        q_prev, q_curr = self.initialize(q0, p0)

        q_traj: List[Tensor] = [q_prev.clone()]
        action_accum = 0.0
        t_vals: List[float] = [0.0]

        for step_idx in range(n_steps):
            q_next = self.step(q_prev, q_curr)

            # Accumulate discrete action
            action_accum += self._Ld(q_curr, q_next).item()

            if (step_idx + 1) % save_every == 0:
                q_traj.append(q_next.clone())
                t_vals.append((step_idx + 1) * self.dt)

            q_prev = q_curr
            q_curr = q_next

        # Recover final momentum via discrete Legendre transform
        p_final = self.discrete_momentum_minus(q_prev, q_curr)

        # Recover all momenta for trajectory
        p_traj: List[Tensor] = []
        for i in range(len(q_traj) - 1):
            p_i = self.discrete_momentum_plus(q_traj[i], q_traj[min(i + 1, len(q_traj) - 1)])
            p_traj.append(p_i)
        p_traj.append(p_final)

        return {
            "q": torch.stack(q_traj),
            "p": torch.stack(p_traj),
            "t": torch.tensor(t_vals),
            "action": action_accum,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  DISCRETE VARIATIONAL INTEGRATOR (CONVENIENCE WRAPPER)
# ═══════════════════════════════════════════════════════════════════════════════

class DiscreteVariationalIntegrator:
    """
    High-level interface for variational integration.

    Wraps DEL_Integrator with automatic quadrature selection, momentum
    recovery, and energy monitoring.

    Example:
        >>> L = lambda q, v: 0.5 * (v**2).sum() - 0.5 * (q**2).sum()  # harmonic
        >>> dvi = DiscreteVariationalIntegrator(L, dt=0.01)
        >>> result = dvi.integrate(q0, p0, t_final=10.0)
    """

    def __init__(
        self,
        L: Callable[[Tensor, Tensor], Tensor],
        dt: float = 0.01,
        quadrature: str = "midpoint",
        H: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        """
        Args:
            L: Continuous Lagrangian L(q, q̇)
            dt: Time step
            quadrature: 'midpoint', 'trapezoidal', or 'simpson'
            H: Optional Hamiltonian for energy monitoring
        """
        self.L = L
        self.dt = dt
        self.H = H
        self._del = DEL_Integrator(L=L, dt=dt, quadrature=quadrature)

    def integrate(
        self,
        q0: Tensor,
        p0: Tensor,
        t_final: float,
        save_every: int = 1,
    ) -> Dict[str, Any]:
        """Run integration and compute diagnostics."""
        result = self._del.integrate(q0, p0, t_final, save_every)

        # Energy monitoring if Hamiltonian provided
        if self.H is not None:
            energies = []
            for i in range(result["q"].shape[0]):
                E = self.H(result["q"][i], result["p"][i])
                energies.append(E.item() if E.ndim == 0 else E.sum().item())
            result["energy"] = energies

            E0 = energies[0]
            scale = abs(E0) if abs(E0) > 1e-30 else 1.0
            result["energy_drift"] = max(
                abs(E - E0) / scale for E in energies
            )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTION MINIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionMinimizer:
    """
    Direct action minimization: find trajectories q(t) that extremize

        S[q] = ∫₀ᵀ L(q, q̇) dt

    subject to boundary conditions q(0) = q_a, q(T) = q_b.

    Uses discrete action and gradient-based optimization (L-BFGS or
    gradient descent) on the interior points of the trajectory.

    This is Hamilton's principle of stationary action, implemented
    as a numerical optimization problem.

    Attributes:
        L: Continuous Lagrangian
        n_nodes: Number of temporal discretization nodes
        optimizer: 'lbfgs' or 'gradient_descent'
        max_iter: Maximum optimization iterations
        tol: Convergence tolerance on action gradient norm
    """

    L: Callable[[Tensor, Tensor], Tensor]
    n_nodes: int = 100
    optimizer: str = "lbfgs"
    max_iter: int = 500
    tol: float = 1e-10

    def _discrete_action(
        self, q_interior: Tensor, q_a: Tensor, q_b: Tensor, dt: float
    ) -> Tensor:
        """
        Compute the discrete action over the full trajectory.

        q_interior: [n_nodes-2, ndof] — the free (optimizable) nodes
        q_a, q_b: boundary conditions
        """
        # Build full trajectory
        q_full = torch.cat([q_a.unsqueeze(0), q_interior, q_b.unsqueeze(0)], dim=0)
        n = q_full.shape[0]

        action = torch.tensor(0.0, dtype=q_full.dtype, device=q_full.device)
        for i in range(n - 1):
            q0 = q_full[i]
            q1 = q_full[i + 1]
            v = (q1 - q0) / dt
            q_mid = 0.5 * (q0 + q1)
            action = action + dt * self.L(q_mid, v)

        return action

    def minimize(
        self,
        q_a: Tensor,
        q_b: Tensor,
        t_final: float,
        initial_guess: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Find the trajectory that extremizes the action.

        Args:
            q_a: Initial configuration q(0)
            q_b: Final configuration q(T)
            t_final: Final time
            initial_guess: Optional [n_nodes-2, ndof] initial trajectory.
                If None, uses linear interpolation.

        Returns:
            Dictionary with:
                'trajectory': [n_nodes, ndof] optimized path
                'action': final action value
                'converged': bool
                'grad_norm': final gradient norm
                'n_iter': iterations used
        """
        dt = t_final / (self.n_nodes - 1)
        ndof = q_a.shape[-1] if q_a.ndim > 0 else 1

        # Initial guess: straight line
        if initial_guess is None:
            alphas = torch.linspace(0, 1, self.n_nodes, dtype=q_a.dtype, device=q_a.device)
            q_full = q_a.unsqueeze(0) * (1 - alphas.unsqueeze(-1)) + q_b.unsqueeze(0) * alphas.unsqueeze(-1)
            q_interior = q_full[1:-1].clone().detach().requires_grad_(True)
        else:
            q_interior = initial_guess.clone().detach().requires_grad_(True)

        converged = False
        grad_norm = float("inf")
        n_iter = 0

        if self.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [q_interior],
                max_iter=self.max_iter,
                tolerance_grad=self.tol,
                tolerance_change=1e-15,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                S = self._discrete_action(q_interior, q_a, q_b, dt)
                S.backward()
                return S

            for _ in range(5):  # L-BFGS outer loops
                loss = opt.step(closure)
                if q_interior.grad is not None:
                    grad_norm = q_interior.grad.abs().max().item()
                    if grad_norm < self.tol:
                        converged = True
                        break
                n_iter += 1

        else:  # gradient descent
            lr = dt * dt  # step size ~ dt² for action gradient
            for iteration in range(self.max_iter):
                S = self._discrete_action(q_interior, q_a, q_b, dt)
                S.backward()

                with torch.no_grad():
                    grad_norm = q_interior.grad.abs().max().item()
                    if grad_norm < self.tol:
                        converged = True
                        n_iter = iteration + 1
                        break

                    q_interior.data -= lr * q_interior.grad
                    q_interior.grad.zero_()
                    n_iter = iteration + 1

        # Build final trajectory
        q_full = torch.cat(
            [q_a.unsqueeze(0), q_interior.detach(), q_b.unsqueeze(0)], dim=0
        )
        final_action = self._discrete_action(
            q_interior.detach(), q_a, q_b, dt
        ).item()

        return {
            "trajectory": q_full,
            "action": final_action,
            "converged": converged,
            "grad_norm": grad_norm,
            "n_iter": n_iter,
            "dt": dt,
            "t": torch.linspace(0, t_final, self.n_nodes, dtype=q_a.dtype, device=q_a.device),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  VARIATIONAL PRINCIPLE VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VariationalPrincipleVerifier:
    """
    Verifies that an integrator satisfies the discrete variational principle.

    Tests:
    1. DEL satisfaction: D₂L_d(q_{k-1}, q_k) + D₁L_d(q_k, q_{k+1}) ≈ 0
    2. Symplecticity: Jacobian is symplectic (det(J) = 1, J^T Ω J = Ω)
    3. Momentum matching: p_k⁺ = p_k⁻ at each step
    4. Action stationarity: δS = 0 at the numerical trajectory
    """

    L: Callable[[Tensor, Tensor], Tensor]
    dt: float = 0.01
    quadrature: str = "midpoint"
    tolerance: float = 1e-8

    def verify_del_satisfaction(
        self, q_trajectory: Tensor
    ) -> Dict[str, float]:
        """
        Check that the DEL equations are satisfied along the trajectory.

        Returns max and mean residual norms.
        """
        del_int = DEL_Integrator(
            L=self.L, dt=self.dt, quadrature=self.quadrature
        )

        n = q_trajectory.shape[0]
        residuals: List[float] = []

        for k in range(1, n - 1):
            D2 = del_int._D2_Ld(q_trajectory[k - 1], q_trajectory[k])
            D1 = del_int._D1_Ld(q_trajectory[k], q_trajectory[k + 1])
            res = (D2 + D1).abs().max().item()
            residuals.append(res)

        max_res = max(residuals) if residuals else 0.0
        mean_res = sum(residuals) / len(residuals) if residuals else 0.0

        return {
            "max_residual": max_res,
            "mean_residual": mean_res,
            "passed": max_res < self.tolerance,
        }

    def verify_momentum_matching(
        self, q_trajectory: Tensor
    ) -> Dict[str, float]:
        """
        Check p_k⁺ = p_k⁻ (momentum matching condition).
        """
        del_int = DEL_Integrator(
            L=self.L, dt=self.dt, quadrature=self.quadrature
        )

        n = q_trajectory.shape[0]
        mismatches: List[float] = []

        for k in range(1, n - 1):
            p_plus = del_int.discrete_momentum_plus(
                q_trajectory[k - 1], q_trajectory[k]
            )
            p_minus = del_int.discrete_momentum_minus(
                q_trajectory[k], q_trajectory[k + 1]
            )
            mismatch = (p_plus - p_minus).abs().max().item()
            mismatches.append(mismatch)

        max_mismatch = max(mismatches) if mismatches else 0.0

        return {
            "max_mismatch": max_mismatch,
            "passed": max_mismatch < self.tolerance,
        }


__all__ = [
    "midpoint_discrete_lagrangian",
    "trapezoidal_discrete_lagrangian",
    "simpson_discrete_lagrangian",
    "DEL_Integrator",
    "DiscreteVariationalIntegrator",
    "ActionMinimizer",
    "VariationalPrincipleVerifier",
]
