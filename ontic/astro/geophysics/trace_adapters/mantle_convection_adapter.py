"""
Mantle Convection Trace Adapter (XIII.2)
=========================================

Wraps ontic.geophysics.mantle_convection.MantleConvection2D for tracing.
Conservation: mass, thermal energy, Nusselt number.

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
class MantleConservation:
    thermal_energy: float
    nusselt_number: float
    max_velocity: float
    rms_temperature: float

    def to_dict(self) -> dict[str, float]:
        return {
            "thermal_energy": self.thermal_energy,
            "nusselt_number": self.nusselt_number,
            "max_velocity": self.max_velocity,
            "rms_temperature": self.rms_temperature,
        }


class MantleConvectionTraceAdapter:
    """
    2D Rayleigh-Bénard mantle convection adapter with trace logging.

    Parameters
    ----------
    nx, nz : int
        Grid dimensions.
    Ra : float
        Rayleigh number.
    """

    def __init__(
        self,
        nx: int = 64,
        nz: int = 64,
        Ra: float = 1e4,
    ) -> None:
        from ontic.astro.geophysics.mantle_convection import MantleConvection2D

        self.solver = MantleConvection2D(nx=nx, nz=nz, Ra=Ra)
        self.nx = nx
        self.nz = nz

    def solve(
        self,
        t_final: float,
        dt: float = 1e-5,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Time-step mantle convection.

        Returns
        -------
        T_field, t, n_steps, session
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

        T = getattr(self.solver, "T", np.zeros((self.nz, self.nx)))
        cons = self._conservation(T, np.zeros_like(T))
        _record(session, 0, 0.0, T, cons)

        for step in range(1, n_steps + 1):
            result = self.solver.step(dt=dt)
            if isinstance(result, tuple):
                T = result[0] if len(result) > 0 else T
                psi = result[1] if len(result) > 1 else np.zeros_like(T)
            else:
                T = getattr(self.solver, "T", T)
                psi = np.zeros_like(T)

            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation(T, psi)
                _record(session, step, step * dt, T, cons)

        return T, n_steps * dt, n_steps, session

    def _conservation(self, T: NDArray, psi: NDArray) -> MantleConservation:
        te = float(np.sum(T))
        dT_dz = np.gradient(T, axis=0) if T.ndim == 2 else np.array([0.0])
        Nu = float(np.abs(np.mean(dT_dz[0]))) if dT_dz.size > 0 else 1.0
        return MantleConservation(
            thermal_energy=te,
            nusselt_number=Nu,
            max_velocity=float(np.max(np.abs(psi))),
            rms_temperature=float(np.sqrt(np.mean(T**2))),
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    T: NDArray,
    cons: MantleConservation,
) -> None:
    session.log_custom(

        name="mantle_convection_step",

        input_hashes=[_hash_array(T)],

        output_hashes=[_hash_array(T)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
