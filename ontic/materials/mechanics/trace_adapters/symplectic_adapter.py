"""
Symplectic Integrator Trace Adapter (I.2)
==========================================

Wraps ontic.mechanics.symplectic.SymplecticIntegratorSuite for STARK tracing.
Conservation: Hamiltonian (energy), symplectic 2-form area.

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
class SymplecticConservation:
    hamiltonian: float
    phase_space_area: float

    def to_dict(self) -> dict[str, float]:
        return {
            "hamiltonian": self.hamiltonian,
            "phase_space_area": self.phase_space_area,
        }


class SymplecticTraceAdapter:
    """
    Standalone symplectic integrator (Störmer-Verlet) with trace logging.

    Operates on a Hamiltonian H(q, p) = T(p) + V(q) with separable kinetic
    and potential energies.  Uses leapfrog (2nd order symplectic).

    Parameters
    ----------
    dim : int
        Phase-space dimension per coordinate.
    kinetic : callable(p) -> float
        Kinetic energy function.
    potential : callable(q) -> float
        Potential energy function.
    grad_potential : callable(q) -> NDArray
        Gradient of V w.r.t. q.
    mass : float
        Particle mass (for T = p²/2m).
    """

    def __init__(
        self,
        dim: int = 1,
        potential: callable = None,
        grad_potential: callable = None,
        mass: float = 1.0,
    ) -> None:
        self.dim = dim
        self.mass = mass
        if potential is None:
            self.potential = lambda q: 0.5 * np.sum(q**2)
            self.grad_potential = lambda q: q.copy()
        else:
            self.potential = potential
            self.grad_potential = grad_potential

    def _hamiltonian(self, q: NDArray, p: NDArray) -> float:
        return float(np.sum(p**2) / (2 * self.mass) + self.potential(q))

    def solve(
        self,
        q0: NDArray,
        p0: NDArray,
        t_final: float,
        dt: float = 0.01,
    ) -> tuple[NDArray, NDArray, float, int, TraceSession]:
        """
        Störmer-Verlet integration.

        Returns
        -------
        q_final, p_final, t, n_steps, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        q = q0.copy().astype(np.float64)
        p = p0.copy().astype(np.float64)
        t = 0.0
        n_steps = int(t_final / dt)

        H0 = self._hamiltonian(q, p)
        _record(session, 0, t, q, p, H0, 0.0)

        for step in range(1, n_steps + 1):
            p -= 0.5 * dt * self.grad_potential(q)
            q += dt * p / self.mass
            p -= 0.5 * dt * self.grad_potential(q)
            t += dt

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                H = self._hamiltonian(q, p)
                _record(session, step, t, q, p, H, abs(H - H0))

        return q, p, t, n_steps, session


def _record(
    session: TraceSession,
    step: int,
    t: float,
    q: NDArray,
    p: NDArray,
    hamiltonian: float,
    energy_drift: float,
) -> None:
    session.log_custom(
        name="symplectic_step",
        input_hashes=[_hash_array(q), _hash_array(p)],
        output_hashes=[_hash_array(q)],
        metrics={"step": step, "time": t, "hamiltonian": hamiltonian, "energy_drift": energy_drift},
    )
