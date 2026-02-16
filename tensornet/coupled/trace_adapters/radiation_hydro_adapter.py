"""
Radiation Hydrodynamics Trace Adapter (XVIII.6)
=================================================

Wraps tensornet.radiation.RadiationEuler1D for STARK tracing.
Conservation: total energy (matter + radiation), mass.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class RadiationHydroConservation:
    total_mass: float
    matter_energy: float
    radiation_energy: float
    total_energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "matter_energy": self.matter_energy,
            "radiation_energy": self.radiation_energy,
            "total_energy": self.total_energy,
        }


class RadiationHydroTraceAdapter:
    """
    Radiation-Euler 1D adapter with trace logging.

    Parameters
    ----------
    nx : int
        Grid points.
    Lx : float
        Domain length (m).
    gamma : float
        Adiabatic index.
    """

    def __init__(
        self,
        nx: int = 400,
        Lx: float = 1.0,
        gamma: float = 5.0 / 3.0,
    ) -> None:
        from tensornet.radiation import RadiationEuler1D

        self.solver = RadiationEuler1D(nx=nx, Lx=Lx, gamma=gamma)
        self.dx = Lx / nx

    def solve(
        self,
        t_final: float,
        dt: float,
    ) -> tuple[NDArray, NDArray, NDArray, float, int, TraceSession]:
        """
        Time-step radiation-hydrodynamics.

        Returns
        -------
        rho, p, Er, t, n_steps, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        n_steps = int(t_final / dt)

        cons = self._conservation()
        _record(session, 0, 0.0, self.solver.rho, self.solver.Er, cons)

        for step in range(1, n_steps + 1):
            self.solver.step(dt)
            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation()
                _record(
                    session, step, step * dt,
                    self.solver.rho, self.solver.Er, cons,
                )

        return (
            self.solver.rho,
            self.solver.p,
            self.solver.Er,
            n_steps * dt,
            n_steps,
            session,
        )

    def _conservation(self) -> RadiationHydroConservation:
        rho = self.solver.rho
        total_mass = float(np.sum(rho) * self.dx)
        matter_e = float(np.sum(self.solver.total_energy()) * self.dx)
        rad_e = float(np.sum(self.solver.Er) * self.dx)
        return RadiationHydroConservation(
            total_mass=total_mass,
            matter_energy=matter_e,
            radiation_energy=rad_e,
            total_energy=matter_e + rad_e,
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    rho: NDArray,
    Er: NDArray,
    cons: RadiationHydroConservation,
) -> None:
    session.log_custom(

        name="radiation_hydro_step",

        input_hashes=[_hash_array(rho), _hash_array(Er)],

        output_hashes=[_hash_array(rho), _hash_array(Er)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
