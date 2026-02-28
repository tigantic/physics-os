"""
Nonlinear Dynamics & Chaos Trace Adapter (I.5)
================================================

Standalone Lorenz attractor + Lyapunov exponent solver with trace logging.
Conservation: Lyapunov exponent bounds, attractor containment.

No existing solver — embeds a complete RK4 integration engine for
generic autonomous ODE systems.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class ChaosConservation:
    max_lyapunov: float
    trajectory_bound: float
    energy_proxy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "max_lyapunov": self.max_lyapunov,
            "trajectory_bound": self.trajectory_bound,
            "energy_proxy": self.energy_proxy,
        }


def lorenz_rhs(state: NDArray, sigma: float, rho: float, beta: float) -> NDArray:
    """Lorenz system RHS."""
    x, y, z = state[0], state[1], state[2]
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


class NonlinearDynamicsTraceAdapter:
    """
    Lorenz system integrator with Lyapunov exponent estimation.

    Parameters
    ----------
    sigma, rho, beta : float
        Lorenz parameters (default: classic chaotic regime).
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _rhs(self, y: NDArray) -> NDArray:
        return lorenz_rhs(y, self.sigma, self.rho, self.beta)

    def _rk4_step(self, y: NDArray, dt: float) -> NDArray:
        k1 = self._rhs(y)
        k2 = self._rhs(y + 0.5 * dt * k1)
        k3 = self._rhs(y + 0.5 * dt * k2)
        k4 = self._rhs(y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(
        self,
        y0: NDArray,
        t_final: float,
        dt: float = 0.01,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Integrate Lorenz system with RK4 and estimate max Lyapunov exponent.

        Parameters
        ----------
        y0 : (3,) initial state
        t_final : float
        dt : float

        Returns
        -------
        trajectory (n_steps+1, 3), t, n_steps, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        y = y0.copy().astype(np.float64)
        n_steps = int(t_final / dt)

        trajectory = np.zeros((n_steps + 1, 3))
        trajectory[0] = y

        # Perturbation vector for Lyapunov estimation
        delta = np.array([1e-10, 0.0, 0.0])
        y_pert = y + delta
        lyap_sum = 0.0
        t = 0.0

        cons = ChaosConservation(
            max_lyapunov=0.0,
            trajectory_bound=float(np.linalg.norm(y)),
            energy_proxy=float(0.5 * np.sum(y**2)),
        )
        _record(session, 0, t, y, cons)

        for step in range(1, n_steps + 1):
            y = self._rk4_step(y, dt)
            y_pert = self._rk4_step(y_pert, dt)

            diff = y_pert - y
            dist = np.linalg.norm(diff)
            if dist > 0:
                lyap_sum += np.log(dist / 1e-10)
                y_pert = y + diff * (1e-10 / dist)

            trajectory[step] = y
            t += dt

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                lyap = lyap_sum / (step * dt) if step > 0 else 0.0
                cons = ChaosConservation(
                    max_lyapunov=float(lyap),
                    trajectory_bound=float(np.max(np.abs(trajectory[: step + 1]))),
                    energy_proxy=float(0.5 * np.sum(y**2)),
                )
                _record(session, step, t, y, cons)

        return trajectory, t, n_steps, session


def _record(
    session: TraceSession,
    step: int,
    t: float,
    state: NDArray,
    cons: ChaosConservation,
) -> None:
    session.log_custom(

        name="nonlinear_dynamics_step",

        input_hashes=[_hash_array(state)],

        output_hashes=[_hash_array(state)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
