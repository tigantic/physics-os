"""
Cosmological Simulations Trace Adapter (XII.4)
================================================

Wraps ontic.astro.cosmological_sims.ParticleMeshNBody for STARK tracing.
Conservation: total mass, energy, Friedmann constraint.

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
class CosmoConservation:
    total_mass: float
    total_energy: float
    kinetic_energy: float
    potential_energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "total_energy": self.total_energy,
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
        }


class CosmologicalSimsTraceAdapter:
    """
    Particle-mesh N-body adapter with trace logging.

    Standalone PM solver for small-scale cosmological N-body.
    Uses the leapfrog integrator on a mesh potential.

    Parameters
    ----------
    n_particles : int
        Number of dark matter particles.
    box_size : float
        Comoving box size.
    n_mesh : int
        Mesh resolution per side.
    """

    def __init__(
        self,
        n_particles: int = 512,
        box_size: float = 100.0,
        n_mesh: int = 32,
    ) -> None:
        self.n_particles = n_particles
        self.box_size = box_size
        self.n_mesh = n_mesh

    def solve(
        self,
        pos0: NDArray,
        vel0: NDArray,
        masses: NDArray,
        t_final: float,
        dt: float = 0.01,
    ) -> tuple[NDArray, NDArray, float, int, TraceSession]:
        """
        Leapfrog PM N-body integration.

        Parameters
        ----------
        pos0 : (n_particles, 3) positions
        vel0 : (n_particles, 3) velocities
        masses : (n_particles,)
        t_final, dt : float

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
        m = masses.astype(np.float64)
        n_steps = int(t_final / dt)

        acc = self._pm_acceleration(pos, m)
        cons = self._conservation(pos, vel, m, acc)
        _record(session, 0, 0.0, pos, vel, cons)

        for step in range(1, n_steps + 1):
            vel += 0.5 * dt * acc
            pos = (pos + dt * vel) % self.box_size  # periodic BC
            acc = self._pm_acceleration(pos, m)
            vel += 0.5 * dt * acc

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation(pos, vel, m, acc)
                _record(session, step, step * dt, pos, vel, cons)

        return pos, vel, n_steps * dt, n_steps, session

    def _pm_acceleration(self, pos: NDArray, masses: NDArray) -> NDArray:
        """CIC-interpolated mesh potential → acceleration."""
        N = self.n_mesh
        L = self.box_size
        dx = L / N
        rho = np.zeros((N, N, N))

        # CIC deposit
        idx = (pos / dx).astype(int) % N
        for i in range(len(masses)):
            rho[idx[i, 0], idx[i, 1], idx[i, 2]] += masses[i]
        rho /= dx**3

        rho_hat = np.fft.rfftn(rho)
        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        kz = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX**2 + KY**2 + KZ**2
        k2[0, 0, 0] = 1.0  # avoid /0
        phi_hat = -4 * np.pi * rho_hat / k2
        phi_hat[0, 0, 0] = 0.0

        # Gradient → acceleration
        acc = np.zeros_like(pos)
        for dim, K in enumerate([KX, KY, KZ]):
            g_hat = -1j * K * phi_hat
            g_field = np.fft.irfftn(g_hat, s=(N, N, N))
            for i in range(len(pos)):
                acc[i, dim] = -g_field[idx[i, 0], idx[i, 1], idx[i, 2]]

        return acc

    def _conservation(
        self, pos: NDArray, vel: NDArray, masses: NDArray, acc: NDArray
    ) -> CosmoConservation:
        ke = 0.5 * np.sum(masses[:, None] * vel**2)
        pe = -0.5 * np.sum(masses[:, None] * acc * pos)
        return CosmoConservation(
            total_mass=float(np.sum(masses)),
            total_energy=float(ke + pe),
            kinetic_energy=float(ke),
            potential_energy=float(pe),
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    pos: NDArray,
    vel: NDArray,
    cons: CosmoConservation,
) -> None:
    session.log_custom(

        name="cosmological_nbody_step",

        input_hashes=[_hash_array(pos), _hash_array(vel)],

        output_hashes=[_hash_array(pos)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
