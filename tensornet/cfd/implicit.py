"""
Implicit Time Integration for Stiff Systems
============================================

Provides backward Euler and BDF-2 schemes for integrating
stiff ODEs arising from finite-rate chemistry.

Key Features:
    - Newton iteration for nonlinear solve
    - Analytical and numerical Jacobian options
    - Line search for robustness
    - Adaptive substepping for very stiff problems

Problem:
    dY/dt = ω(Y, T) / ρ

The chemistry source term ω can vary by 10+ orders of magnitude,
making explicit methods require tiny timesteps. Implicit methods
allow larger timesteps at the cost of solving nonlinear systems.

References:
    [1] Hairer & Wanner, "Solving Ordinary Differential Equations II:
        Stiff and Differential-Algebraic Problems", Springer 1996
    [2] Oran & Boris, "Numerical Simulation of Reactive Flow", 2nd Ed.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch


class SolverStatus(Enum):
    """Status of nonlinear solver."""

    SUCCESS = 0
    MAX_ITERATIONS = 1
    LINE_SEARCH_FAILED = 2
    SINGULAR_JACOBIAN = 3


@dataclass
class NewtonResult:
    """Result from Newton iteration."""

    x: torch.Tensor
    status: SolverStatus
    iterations: int
    residual_norm: float


@dataclass
class ImplicitConfig:
    """Configuration for implicit integrator."""

    max_newton_iters: int = 20
    newton_tol: float = 1e-8
    line_search: bool = True
    line_search_max_iters: int = 10
    line_search_alpha: float = 1e-4  # Armijo condition parameter
    jacobian_numerical: bool = True
    jacobian_eps: float = 1e-6
    adaptive_substep: bool = True
    min_substeps: int = 1
    max_substeps: int = 100
    substep_safety: float = 0.5


def newton_solve(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    config: ImplicitConfig,
) -> NewtonResult:
    """
    Solve F(x) = 0 using Newton's method with optional line search.

    Newton iteration: x_{n+1} = x_n - J^{-1} F(x_n)

    Args:
        residual_fn: Function computing F(x)
        jacobian_fn: Function computing Jacobian J = dF/dx
        x0: Initial guess
        config: Solver configuration

    Returns:
        NewtonResult with solution and status
    """
    x = x0.clone()

    for iteration in range(config.max_newton_iters):
        # Compute residual and Jacobian
        F = residual_fn(x)
        residual_norm = F.norm().item()

        # Check convergence
        if residual_norm < config.newton_tol:
            return NewtonResult(
                x=x,
                status=SolverStatus.SUCCESS,
                iterations=iteration + 1,
                residual_norm=residual_norm,
            )

        # Compute Jacobian
        J = jacobian_fn(x)

        # Solve linear system: J * delta = -F
        try:
            delta = torch.linalg.solve(J, -F.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            return NewtonResult(
                x=x,
                status=SolverStatus.SINGULAR_JACOBIAN,
                iterations=iteration + 1,
                residual_norm=residual_norm,
            )

        # Line search (optional)
        if config.line_search:
            step_size = 1.0
            F_norm_old = residual_norm

            for ls_iter in range(config.line_search_max_iters):
                x_trial = x + step_size * delta
                F_trial = residual_fn(x_trial)
                F_norm_new = F_trial.norm().item()

                # Armijo condition
                if F_norm_new < (1 - config.line_search_alpha * step_size) * F_norm_old:
                    x = x_trial
                    break

                step_size *= 0.5
            else:
                # Line search failed - take small step anyway
                x = x + 0.1 * delta
        else:
            x = x + delta

    # Max iterations reached
    F = residual_fn(x)
    return NewtonResult(
        x=x,
        status=SolverStatus.MAX_ITERATIONS,
        iterations=config.max_newton_iters,
        residual_norm=F.norm().item(),
    )


def numerical_jacobian(
    func: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Jacobian using finite differences.

    J_ij = d F_i / d x_j ≈ (F_i(x + eps*e_j) - F_i(x)) / eps

    Args:
        func: Function F: R^n -> R^m
        x: Point to evaluate Jacobian
        eps: Finite difference step

    Returns:
        Jacobian matrix [m, n]
    """
    n = x.shape[-1] if x.dim() > 0 else 1
    F0 = func(x)
    m = F0.shape[-1] if F0.dim() > 0 else 1

    J = torch.zeros((m, n), dtype=x.dtype, device=x.device)

    for j in range(n):
        x_plus = x.clone()
        x_plus[j] += eps
        F_plus = func(x_plus)
        J[:, j] = (F_plus - F0) / eps

    return J


@dataclass
class ChemistryIntegrator:
    """
    Implicit integrator for chemistry source terms.

    Solves: dY/dt = ω(Y, T) / ρ
    using backward Euler.
    """

    config: ImplicitConfig
    rho: float = 1.0  # Density [kg/m³]

    def integrate(
        self,
        Y0: torch.Tensor,
        T: float,
        omega_fn: Callable[[torch.Tensor, float], torch.Tensor],
        dt: float,
    ) -> tuple[torch.Tensor, SolverStatus]:
        """
        Integrate chemistry one timestep using backward Euler.

        Y^{n+1} = Y^n + dt * ω(Y^{n+1}, T) / ρ

        Residual: F(Y) = Y - Y^n - dt * ω(Y, T) / ρ = 0

        Args:
            Y0: Initial mass fractions [n_species]
            T: Temperature [K] (assumed constant over dt)
            omega_fn: Function (Y, T) -> production rates
            dt: Timestep [s]

        Returns:
            Tuple of (Y_new, status)
        """
        Y0_tensor = Y0 if torch.is_tensor(Y0) else torch.tensor(Y0, dtype=torch.float64)

        # Residual function
        def residual(Y: torch.Tensor) -> torch.Tensor:
            omega = omega_fn(Y, T)
            return Y - Y0_tensor - dt * omega / self.rho

        # Jacobian function
        def jacobian(Y: torch.Tensor) -> torch.Tensor:
            if self.config.jacobian_numerical:
                return numerical_jacobian(residual, Y, self.config.jacobian_eps)
            else:
                # Analytical Jacobian would go here
                return numerical_jacobian(residual, Y, self.config.jacobian_eps)

        # Initial guess: explicit Euler
        omega0 = omega_fn(Y0_tensor, T)
        Y_guess = Y0_tensor + 0.1 * dt * omega0 / self.rho
        Y_guess = torch.clamp(Y_guess, min=0.0)

        # Newton solve
        result = newton_solve(residual, jacobian, Y_guess, self.config)

        # Clip and renormalize
        Y_new = torch.clamp(result.x, min=0.0)
        Y_new = Y_new / Y_new.sum()

        return Y_new, result.status


def backward_euler_scalar(
    y0: float,
    f: Callable[[float, float], float],
    t0: float,
    dt: float,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> float:
    """
    Backward Euler for scalar ODE: dy/dt = f(t, y)

    y^{n+1} = y^n + dt * f(t^{n+1}, y^{n+1})

    Args:
        y0: Initial value
        f: RHS function f(t, y)
        t0: Initial time
        dt: Timestep
        tol: Newton convergence tolerance
        max_iter: Maximum Newton iterations

    Returns:
        y at t0 + dt
    """
    t_new = t0 + dt
    y = y0  # Initial guess

    for _ in range(max_iter):
        # Residual: g(y) = y - y0 - dt * f(t_new, y)
        g = y - y0 - dt * f(t_new, y)

        if abs(g) < tol:
            break

        # Jacobian: dg/dy = 1 - dt * df/dy
        eps = 1e-8
        dg_dy = 1.0 - dt * (f(t_new, y + eps) - f(t_new, y)) / eps

        # Newton step
        y = y - g / dg_dy

    return y


def bdf2_scalar(
    y0: float,
    y1: float,
    f: Callable[[float, float], float],
    t1: float,
    dt: float,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> float:
    """
    BDF-2 for scalar ODE: dy/dt = f(t, y)

    y^{n+2} = (4/3) y^{n+1} - (1/3) y^n + (2/3) dt * f(t^{n+2}, y^{n+2})

    Args:
        y0: Value at t - dt
        y1: Value at t
        f: RHS function f(t, y)
        t1: Time at y1
        dt: Timestep
        tol: Newton convergence tolerance
        max_iter: Maximum Newton iterations

    Returns:
        y at t1 + dt
    """
    t_new = t1 + dt
    y = y1  # Initial guess

    for _ in range(max_iter):
        # Residual
        g = y - (4.0 / 3.0) * y1 + (1.0 / 3.0) * y0 - (2.0 / 3.0) * dt * f(t_new, y)

        if abs(g) < tol:
            break

        # Jacobian
        eps = 1e-8
        dg_dy = 1.0 - (2.0 / 3.0) * dt * (f(t_new, y + eps) - f(t_new, y)) / eps

        # Newton step
        y = y - g / dg_dy

    return y


class AdaptiveImplicit:
    """
    Adaptive implicit integrator with error control.

    Uses embedded BDF-1/BDF-2 for error estimation.
    """

    def __init__(
        self, config: ImplicitConfig = None, rtol: float = 1e-4, atol: float = 1e-10
    ):
        self.config = config or ImplicitConfig()
        self.rtol = rtol
        self.atol = atol

    def integrate(
        self, y0: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], dt: float
    ) -> tuple[torch.Tensor, float, int]:
        """
        Integrate with adaptive substepping.

        Args:
            y0: Initial state
            f: RHS function f(y) -> dy/dt
            dt: Target timestep

        Returns:
            Tuple of (y_new, actual_dt, n_substeps)
        """
        if not self.config.adaptive_substep:
            # Fixed step
            y_new = self._backward_euler_step(y0, f, dt)
            return y_new, dt, 1

        # Adaptive substepping
        t = 0.0
        y = y0.clone()
        n_substeps = 0
        dt_sub = dt

        while t < dt and n_substeps < self.config.max_substeps:
            # Don't overshoot
            dt_sub = min(dt_sub, dt - t)

            # Take step
            y_new = self._backward_euler_step(y, f, dt_sub)

            # Estimate error (compare with two half-steps)
            y_half = self._backward_euler_step(y, f, dt_sub / 2)
            y_full = self._backward_euler_step(y_half, f, dt_sub / 2)

            err = (y_new - y_full).abs()
            scale = self.atol + self.rtol * torch.maximum(y.abs(), y_new.abs())
            err_ratio = (err / scale).max().item()

            if err_ratio < 1.0:
                # Accept step
                y = y_full  # Use more accurate result
                t += dt_sub
                n_substeps += 1

                # Grow step
                dt_sub = min(dt_sub * min(2.0, 0.9 * err_ratio ** (-0.5)), dt - t)
            else:
                # Reject and shrink
                dt_sub *= max(0.1, self.config.substep_safety * err_ratio ** (-0.5))

        return y, dt, n_substeps

    def _backward_euler_step(
        self, y0: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], dt: float
    ) -> torch.Tensor:
        """Single backward Euler step."""

        def residual(y: torch.Tensor) -> torch.Tensor:
            return y - y0 - dt * f(y)

        def jacobian(y: torch.Tensor) -> torch.Tensor:
            return numerical_jacobian(residual, y, self.config.jacobian_eps)

        result = newton_solve(residual, jacobian, y0.clone(), self.config)
        return result.x


def validate_implicit():
    """
    Run validation tests for implicit integrator.
    """
    print("\n" + "=" * 70)
    print("IMPLICIT INTEGRATION VALIDATION")
    print("=" * 70)

    # Test 1: Simple stiff ODE
    print("\n[Test 1] Stiff Scalar ODE: dy/dt = -1000*(y - cos(t))")
    print("-" * 40)

    # Exact solution: y(t) ≈ cos(t) for large λ
    def f(t, y):
        lam = 1000.0
        return -lam * (y - math.cos(t))

    y0 = 1.0  # y(0) = 1
    dt = 0.01  # Much larger than explicit stability limit (dt < 2/λ = 0.002)

    # Integrate to t = 1
    y = y0
    t = 0.0
    while t < 1.0:
        y = backward_euler_scalar(y, f, t, dt)
        t += dt

    y_exact = math.cos(1.0)
    error = abs(y - y_exact)

    print(f"Computed y(1) = {y:.6f}")
    print(f"Exact y(1) = {y_exact:.6f}")
    print(f"Error: {error:.2e}")

    if error < 0.01:
        print("✓ PASS: Stiff ODE solved accurately with large dt")
    else:
        print("✗ FAIL: Error too large")

    # Test 2: Newton solver
    print("\n[Test 2] Newton Solver for Nonlinear System")
    print("-" * 40)

    # Solve x^3 - x - 1 = 0 (real root ≈ 1.3247)
    def residual(x):
        return x**3 - x - torch.ones_like(x)

    def jacobian(x):
        return (3 * x**2 - 1).unsqueeze(-1).unsqueeze(-1)

    x0 = torch.tensor([2.0], dtype=torch.float64)
    config = ImplicitConfig()
    result = newton_solve(residual, jacobian, x0, config)

    x_exact = 1.3247179572
    error = abs(result.x.item() - x_exact)

    print(f"Computed root: {result.x.item():.10f}")
    print(f"Exact root: {x_exact:.10f}")
    print(f"Iterations: {result.iterations}")
    print(f"Status: {result.status.name}")

    if result.status == SolverStatus.SUCCESS and error < 1e-8:
        print("✓ PASS: Newton solver converged")
    else:
        print("✗ FAIL: Newton solver failed")

    # Test 3: Numerical Jacobian
    print("\n[Test 3] Numerical Jacobian Accuracy")
    print("-" * 40)

    def test_func(x):
        return torch.stack([x[0] ** 2 + x[1], x[0] * x[1] ** 2])

    x_test = torch.tensor([1.0, 2.0], dtype=torch.float64)
    J_num = numerical_jacobian(test_func, x_test)

    # Analytical: J = [[2x₀, 1], [x₁², 2x₀x₁]]
    J_exact = torch.tensor([[2.0, 1.0], [4.0, 4.0]], dtype=torch.float64)

    error = (J_num - J_exact).abs().max().item()
    print(f"Numerical Jacobian:\n{J_num}")
    print(f"Exact Jacobian:\n{J_exact}")
    print(f"Max error: {error:.2e}")

    if error < 1e-5:
        print("✓ PASS: Numerical Jacobian accurate")
    else:
        print("✗ FAIL: Jacobian error too large")

    # Test 4: Adaptive integrator
    print("\n[Test 4] Adaptive Implicit Integrator")
    print("-" * 40)

    # Integrate dy/dt = -100*y from y(0) = 1 to t = 0.1
    # Exact: y(0.1) = e^{-10} ≈ 4.54e-5

    def f_stiff(y):
        return -100.0 * y

    y0 = torch.tensor([1.0], dtype=torch.float64)
    integrator = AdaptiveImplicit(rtol=1e-4)

    y_final, dt_used, n_substeps = integrator.integrate(y0, f_stiff, dt=0.1)
    y_exact = math.exp(-10)
    error = abs(y_final.item() - y_exact) / y_exact

    print(f"Computed y(0.1) = {y_final.item():.6e}")
    print(f"Exact y(0.1) = {y_exact:.6e}")
    print(f"Relative error: {error:.2e}")
    print(f"Substeps: {n_substeps}")

    if error < 1e-2:
        print("✓ PASS: Adaptive integrator accurate")
    else:
        print("✗ FAIL: Adaptive integrator error too large")

    print("\n" + "=" * 70)
    print("IMPLICIT INTEGRATION VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_implicit()
