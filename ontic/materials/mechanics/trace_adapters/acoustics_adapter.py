"""
Acoustics & Vibration Trace Adapter (I.6)
==========================================

Standalone 1D acoustic wave equation solver with trace logging.
Conservation: acoustic energy, reciprocity.

No existing solver — embeds a complete FDTD 1D acoustic solver.

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
class AcousticsConservation:
    total_energy: float
    max_pressure: float
    rms_pressure: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_energy": self.total_energy,
            "max_pressure": self.max_pressure,
            "rms_pressure": self.rms_pressure,
        }


class AcousticsTraceAdapter:
    """
    1D FDTD acoustic wave solver with trace logging.

    Solves the 1D wave equation: ∂²p/∂t² = c² ∂²p/∂x²
    using a staggered-grid leapfrog scheme (pressure + velocity).

    Parameters
    ----------
    nx : int
        Number of grid points.
    Lx : float
        Domain length (m).
    c : float
        Speed of sound (m/s).
    rho : float
        Density (kg/m³).
    """

    def __init__(
        self,
        nx: int = 200,
        Lx: float = 1.0,
        c: float = 343.0,
        rho: float = 1.225,
    ) -> None:
        self.nx = nx
        self.Lx = Lx
        self.c = c
        self.rho = rho
        self.dx = Lx / nx

    def solve(
        self,
        p0: NDArray,
        t_final: float,
        dt: float | None = None,
        source_pos: int | None = None,
        source_freq: float = 1000.0,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        FDTD integration of 1D acoustics.

        Parameters
        ----------
        p0 : (nx,) initial pressure field.
        t_final : float
        dt : float or None (auto CFL).
        source_pos : int or None
            Grid index for point source.
        source_freq : float
            Source frequency (Hz).

        Returns
        -------
        pressure_final, t, n_steps, session
        """
        if dt is None:
            dt = 0.5 * self.dx / self.c

        session = TraceSession()


        session.log_custom(


            name="input_state",


            input_hashes=[],


            output_hashes=[],


            params={},


            metrics={},


        )
        n_steps = int(t_final / dt)
        p = p0.copy().astype(np.float64)
        vx = np.zeros(self.nx + 1)

        cons = self._conservation(p, vx)
        _record(session, 0, 0.0, p, vx, cons)

        Z = self.rho * self.c
        courant = self.c * dt / self.dx

        for step in range(1, n_steps + 1):
            # Update velocity (staggered)
            vx[1:-1] -= (dt / (self.rho * self.dx)) * (p[1:] - p[:-1])
            # Absorbing BCs
            vx[0] = p[0] / Z
            vx[-1] = -p[-1] / Z

            # Update pressure
            p -= (self.rho * self.c**2 * dt / self.dx) * (vx[1:] - vx[:-1])

            # Point source
            if source_pos is not None:
                t_now = step * dt
                p[source_pos] += dt * np.sin(2 * np.pi * source_freq * t_now)

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation(p, vx)
                _record(session, step, step * dt, p, vx, cons)

        return p, n_steps * dt, n_steps, session

    def _conservation(self, p: NDArray, vx: NDArray) -> AcousticsConservation:
        pe = 0.5 * np.sum(p**2) / (self.rho * self.c**2) * self.dx
        ke = 0.5 * self.rho * np.sum(vx**2) * self.dx
        return AcousticsConservation(
            total_energy=float(pe + ke),
            max_pressure=float(np.max(np.abs(p))),
            rms_pressure=float(np.sqrt(np.mean(p**2))),
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    p: NDArray,
    vx: NDArray,
    cons: AcousticsConservation,
) -> None:
    session.log_custom(

        name="acoustics_step",

        input_hashes=[_hash_array(p), _hash_array(vx)],

        output_hashes=[_hash_array(p)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
