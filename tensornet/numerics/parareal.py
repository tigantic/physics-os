"""
Parareal — Parallel-in-Time Integration
=========================================

The Parareal algorithm enables parallel time integration by combining
a coarse (cheap) propagator :math:`G` with a fine (accurate) propagator
:math:`F`:

.. math::
    U_{n+1}^{k+1} = G(U_n^{k+1}) + F(U_n^k) - G(U_n^k)

Convergence is guaranteed after at most :math:`N` iterations
(where :math:`N` is the number of time slices), but typically
converges much faster.

References:
    [1] Lions, Maday & Turinici, "A parareal in time discretization
        of PDE's", C. R. Acad. Sci. 332, 661 (2001).
    [2] Gander & Vandewalle, "Analysis of the parareal time-parallel
        time-integration method", SIAM J. Sci. Comput. 29, 556 (2007).

Domain I.3.1 — Numerics / Time integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


PropagatorFn = Callable[[NDArray, float, float], NDArray]
"""Signature: propagator(u, t_start, t_end) -> u_end."""


@dataclass
class PararealResult:
    """
    Result of Parareal solve.

    Attributes:
        solution: Solution at each time-slice boundary ``(N+1, *state_shape)``.
        iterations: Number of Parareal iterations to convergence.
        residuals: List of max-norm residual per iteration.
    """
    solution: NDArray
    iterations: int
    residuals: list


class PararealSolver:
    """
    Parareal parallel-in-time integrator.

    Example::

        def fine(u, t0, t1):
            # RK4 with 1000 sub-steps
            ...
        def coarse(u, t0, t1):
            # Forward Euler with 10 sub-steps
            ...
        solver = PararealSolver(fine, coarse)
        result = solver.solve(u0, t_span=(0, 10), N=20)
    """

    def __init__(
        self,
        fine: PropagatorFn,
        coarse: PropagatorFn,
        max_iter: int = 50,
        tol: float = 1e-10,
    ) -> None:
        """
        Parameters:
            fine: Accurate (expensive) propagator.
            coarse: Cheap (approximate) propagator.
            max_iter: Maximum Parareal iterations.
            tol: Convergence tolerance on max-norm update.
        """
        self.fine = fine
        self.coarse = coarse
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        u0: NDArray,
        t_span: Tuple[float, float],
        N: int,
    ) -> PararealResult:
        """
        Run the Parareal algorithm.

        Parameters:
            u0: Initial condition.
            t_span: (t_start, t_end).
            N: Number of time slices.

        Returns:
            PararealResult.
        """
        t0, tf = t_span
        dt = (tf - t0) / N
        times = np.linspace(t0, tf, N + 1)

        shape = u0.shape
        U = np.zeros((N + 1,) + shape)
        U[0] = u0.copy()

        # Initialise with coarse propagator
        for n in range(N):
            U[n + 1] = self.coarse(U[n], times[n], times[n + 1])

        residuals = []

        for k in range(self.max_iter):
            U_old = U.copy()

            # Fine propagation (parallelisable in practice)
            F_vals = np.zeros_like(U)
            G_old = np.zeros_like(U)
            for n in range(N):
                F_vals[n + 1] = self.fine(U_old[n], times[n], times[n + 1])
                G_old[n + 1] = self.coarse(U_old[n], times[n], times[n + 1])

            # Sequential correction
            for n in range(N):
                G_new = self.coarse(U[n], times[n], times[n + 1])
                U[n + 1] = G_new + F_vals[n + 1] - G_old[n + 1]

            # Convergence check
            max_update = np.max(np.abs(U - U_old))
            residuals.append(float(max_update))

            if max_update < self.tol:
                return PararealResult(
                    solution=U,
                    iterations=k + 1,
                    residuals=residuals,
                )

        return PararealResult(
            solution=U,
            iterations=self.max_iter,
            residuals=residuals,
        )


# ---------------------------------------------------------------------------
#  Convenience propagator factories
# ---------------------------------------------------------------------------

def forward_euler_propagator(
    rhs: Callable[[NDArray, float], NDArray],
    n_sub: int = 10,
) -> PropagatorFn:
    """Create a forward-Euler propagator with *n_sub* sub-steps."""
    def _propagate(u: NDArray, t0: float, t1: float) -> NDArray:
        dt_sub = (t1 - t0) / n_sub
        t = t0
        y = u.copy()
        for _ in range(n_sub):
            y = y + dt_sub * rhs(y, t)
            t += dt_sub
        return y
    return _propagate


def rk4_propagator(
    rhs: Callable[[NDArray, float], NDArray],
    n_sub: int = 100,
) -> PropagatorFn:
    """Create an RK4 propagator with *n_sub* sub-steps."""
    def _propagate(u: NDArray, t0: float, t1: float) -> NDArray:
        dt_sub = (t1 - t0) / n_sub
        t = t0
        y = u.copy()
        for _ in range(n_sub):
            k1 = rhs(y, t)
            k2 = rhs(y + 0.5 * dt_sub * k1, t + 0.5 * dt_sub)
            k3 = rhs(y + 0.5 * dt_sub * k2, t + 0.5 * dt_sub)
            k4 = rhs(y + dt_sub * k3, t + dt_sub)
            y = y + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += dt_sub
        return y
    return _propagate
