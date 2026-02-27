"""
Newtonian Dynamics Trace Adapter (I.1)
=======================================

Wraps tensornet.guidance.trajectory.TrajectorySolver for STARK trace logging.
Conservation: linear momentum, angular momentum, total energy.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class NewtonianConservation:
    """Conservation quantities per step."""
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    linear_momentum_mag: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "total_energy": self.total_energy,
            "linear_momentum_mag": self.linear_momentum_mag,
        }


class NewtonianDynamicsTraceAdapter:
    """
    Standalone N-body gravitational dynamics with trace logging.

    Embeds a simple N-body leapfrog integrator rather than wrapping
    TrajectorySolver (which is a 6-DOF vehicle sim with Torch deps).
    This gives a clean, dependency-free Newtonian dynamics solver.

    Parameters
    ----------
    n_bodies : int
        Number of particles.
    dim : int
        Spatial dimensions (2 or 3).
    G : float
        Gravitational constant (default 1.0 for normalised units).
    softening : float
        Gravitational softening length.
    """

    def __init__(
        self,
        n_bodies: int = 3,
        dim: int = 3,
        G: float = 1.0,
        softening: float = 1e-4,
    ) -> None:
        self.n_bodies = n_bodies
        self.dim = dim
        self.G = G
        self.softening = softening

    def _accelerations(self, pos: NDArray, masses: NDArray) -> NDArray:
        """Compute gravitational accelerations for all bodies."""
        n = self.n_bodies
        acc = np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rij = pos[j] - pos[i]
                r2 = np.dot(rij, rij) + self.softening**2
                acc[i] += self.G * masses[j] * rij / r2**1.5
        return acc

    def _compute_conservation(
        self,
        pos: NDArray,
        vel: NDArray,
        masses: NDArray,
    ) -> NewtonianConservation:
        """Compute conservation quantities."""
        ke = 0.5 * np.sum(masses[:, None] * vel**2)
        pe = 0.0
        n = self.n_bodies
        for i in range(n):
            for j in range(i + 1, n):
                rij = pos[j] - pos[i]
                r = np.sqrt(np.dot(rij, rij) + self.softening**2)
                pe -= self.G * masses[i] * masses[j] / r
        total_mom = np.sum(masses[:, None] * vel, axis=0)
        return NewtonianConservation(
            kinetic_energy=float(ke),
            potential_energy=float(pe),
            total_energy=float(ke + pe),
            linear_momentum_mag=float(np.linalg.norm(total_mom)),
        )

    def solve(
        self,
        pos0: NDArray,
        vel0: NDArray,
        masses: NDArray,
        t_final: float,
        dt: float = 0.001,
    ) -> tuple[NDArray, NDArray, float, int, TraceSession]:
        """
        Leapfrog integration of N-body system.

        Parameters
        ----------
        pos0 : (n_bodies, dim) initial positions
        vel0 : (n_bodies, dim) initial velocities
        masses : (n_bodies,) particle masses
        t_final : float
        dt : float

        Returns
        -------
        pos_final, vel_final, t, n_steps, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        pos = pos0.copy().astype(np.float64)
        vel = vel0.copy().astype(np.float64)
        masses = np.asarray(masses, dtype=np.float64)
        t = 0.0
        n_steps = int(t_final / dt)

        acc = self._accelerations(pos, masses)
        cons = self._compute_conservation(pos, vel, masses)
        _record(session, 0, t, pos, vel, cons)

        for step in range(1, n_steps + 1):
            vel += 0.5 * dt * acc
            pos += dt * vel
            acc = self._accelerations(pos, masses)
            vel += 0.5 * dt * acc
            t += dt

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._compute_conservation(pos, vel, masses)
                _record(session, step, t, pos, vel, cons)

        return pos, vel, t, n_steps, session


def _record(
    session: TraceSession,
    step: int,
    t: float,
    pos: NDArray,
    vel: NDArray,
    cons: NewtonianConservation,
) -> None:
    """Append a trace entry to the session."""
    from tensornet.core.trace import TraceSession

    session.log_custom(
        name="newtonian_dynamics_step",
        input_hashes=[_hash_array(pos), _hash_array(vel)],
        output_hashes=[_hash_array(pos)],
        metrics={"step": step, "time": t, **cons.to_dict()},
    )
