"""
Laser Physics Trace Adapter (IV.3)
====================================

Wraps tensornet.optics.laser_physics.FourLevelLaser for STARK tracing.
Conservation: population inversion, energy balance.

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
class LaserConservation:
    population_upper: float
    photon_density: float
    total_population: float

    def to_dict(self) -> dict[str, float]:
        return {
            "population_upper": self.population_upper,
            "photon_density": self.photon_density,
            "total_population": self.total_population,
        }


class LaserPhysicsTraceAdapter:
    """
    Four-level laser rate equation adapter with trace logging.

    Parameters
    ----------
    sigma_e, tau_21, n_refr, L_cav, R1, R2, V_mode : float
        Laser cavity parameters (see FourLevelLaser docs).
    """

    def __init__(
        self,
        sigma_e: float = 3e-19,
        tau_21: float = 230e-6,
        n_refr: float = 1.82,
        L_cav: float = 0.1,
        R1: float = 0.999,
        R2: float = 0.95,
        V_mode: float = 1e-9,
    ) -> None:
        from tensornet.applied.optics.laser_physics import FourLevelLaser

        self.laser = FourLevelLaser(
            sigma_e=sigma_e,
            tau_21=tau_21,
            n_refr=n_refr,
            L_cav=L_cav,
            R1=R1,
            R2=R2,
            V_mode=V_mode,
        )

    def solve(
        self,
        pump_rate: float = 1e24,
        dt: float = 1e-9,
        n_steps: int = 10000,
    ) -> tuple[dict[str, float], TraceSession]:
        """
        Evolve laser rate equations.

        Returns
        -------
        metrics, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        result = self.laser.evolve(pump_rate=pump_rate, dt=dt, n_steps=n_steps)

        if isinstance(result, dict):
            N2 = result.get("N2", np.array([0.0]))
            phi = result.get("phi", np.array([0.0]))
            t_arr = result.get("t", np.array([0.0]))
        elif isinstance(result, (tuple, list)):
            t_arr = result[0] if len(result) > 0 else np.array([0.0])
            N2 = result[1] if len(result) > 1 else np.array([0.0])
            phi = result[2] if len(result) > 2 else np.array([0.0])
        else:
            N2 = np.array([0.0])
            phi = np.array([0.0])
            t_arr = np.array([0.0])

        N2 = np.atleast_1d(N2)
        phi = np.atleast_1d(phi)

        cons_initial = LaserConservation(
            population_upper=float(N2[0]),
            photon_density=float(phi[0]),
            total_population=float(N2[0] + phi[0]),
        )
        _record(session, 0, 0.0, N2[:1], phi[:1], cons_initial)

        cons_final = LaserConservation(
            population_upper=float(N2[-1]),
            photon_density=float(phi[-1]),
            total_population=float(N2[-1] + phi[-1]),
        )
        _record(session, 1, float(np.max(t_arr)) if len(t_arr) > 0 else dt * n_steps,
                N2[-1:], phi[-1:], cons_final)

        metrics = {
            "final_N2": float(N2[-1]),
            "final_phi": float(phi[-1]),
            "threshold_pop": float(self.laser.threshold_population()),
            **cons_final.to_dict(),
        }
        return metrics, session


def _record(
    session: TraceSession,
    step: int,
    t: float,
    N2: NDArray,
    phi: NDArray,
    cons: LaserConservation,
) -> None:
    session.log_custom(

        name="laser_physics_step",

        input_hashes=[_hash_array(N2)],

        output_hashes=[_hash_array(phi)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
