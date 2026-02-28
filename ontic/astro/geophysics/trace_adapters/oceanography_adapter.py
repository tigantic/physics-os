"""
Oceanography Trace Adapter (XIII.5)
=====================================

Wraps ontic.geophysics.oceanography.ShallowWaterEquations for STARK tracing.
Conservation: total energy (KE + PE), mass (∫η).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession

G_EARTH = 9.80665


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class OceanConservation:
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    mass_anomaly: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "total_energy": self.total_energy,
            "mass_anomaly": self.mass_anomaly,
        }


class OceanographyTraceAdapter:
    """
    Rotating shallow water equations adapter with trace logging.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    Lx, Ly : float
        Domain size (m).
    H : float
        Mean depth (m).
    f0 : float
        Coriolis parameter (s⁻¹).
    """

    def __init__(
        self,
        nx: int = 50,
        ny: int = 50,
        Lx: float = 1e6,
        Ly: float = 1e6,
        H: float = 4000.0,
        f0: float = 1e-4,
    ) -> None:
        from ontic.astro.geophysics.oceanography import ShallowWaterEquations

        self.solver = ShallowWaterEquations(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, H=H, f0=f0,
        )
        self.H = H
        self.dx = Lx / nx
        self.dy = Ly / ny

    def solve(
        self,
        eta0: NDArray,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[NDArray, NDArray, NDArray, float, int, TraceSession]:
        """
        Time-step shallow water equations.

        Parameters
        ----------
        eta0 : Initial surface elevation perturbation.
        t_final : End time (s).
        dt : Time step (s).  Defaults to CFL estimate.

        Returns
        -------
        u, v, eta, t, n_steps, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        if dt is None:
            dt = self.solver.cfl_timestep(cfl=0.4)

        n_steps = int(t_final / dt)
        self.solver.eta = eta0.copy()

        cons = self._conservation()
        _record(session, 0, 0.0, self.solver.u, self.solver.v, self.solver.eta, cons)

        for step in range(1, n_steps + 1):
            self.solver.step(dt)
            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation()
                _record(
                    session, step, step * dt,
                    self.solver.u, self.solver.v, self.solver.eta, cons,
                )

        return (
            self.solver.u,
            self.solver.v,
            self.solver.eta,
            n_steps * dt,
            n_steps,
            session,
        )

    def _conservation(self) -> OceanConservation:
        u = self.solver.u
        v = self.solver.v
        eta = self.solver.eta
        ke = float(0.5 * self.H * np.sum(u**2 + v**2) * self.dx * self.dy)
        pe = float(0.5 * G_EARTH * np.sum(eta**2) * self.dx * self.dy)
        mass = float(np.sum(eta) * self.dx * self.dy)
        return OceanConservation(
            kinetic_energy=ke,
            potential_energy=pe,
            total_energy=ke + pe,
            mass_anomaly=mass,
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    u: NDArray,
    v: NDArray,
    eta: NDArray,
    cons: OceanConservation,
) -> None:
    session.log_custom(

        name="ocean_step",

        input_hashes=[_hash_array(u), _hash_array(v), _hash_array(eta)],

        output_hashes=[_hash_array(u), _hash_array(v), _hash_array(eta)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
