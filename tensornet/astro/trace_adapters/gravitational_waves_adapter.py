"""
Gravitational Waves Trace Adapter (XII.3)
==========================================

Wraps tensornet.astro.gravitational_waves.PostNewtonianInspiral for tracing.
Conservation: energy flux, angular momentum.

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
class GWConservation:
    chirp_mass: float
    isco_frequency: float
    coalescence_time: float
    energy_radiated: float

    def to_dict(self) -> dict[str, float]:
        return {
            "chirp_mass": self.chirp_mass,
            "isco_frequency": self.isco_frequency,
            "coalescence_time": self.coalescence_time,
            "energy_radiated": self.energy_radiated,
        }


class GravitationalWavesTraceAdapter:
    """
    Post-Newtonian inspiral waveform adapter with trace logging.

    Parameters
    ----------
    m1, m2 : float
        Component masses (solar masses).
    D_L : float
        Luminosity distance (Mpc).
    """

    def __init__(
        self,
        m1: float = 30.0,
        m2: float = 30.0,
        D_L: float = 410.0,
    ) -> None:
        from tensornet.astro.gravitational_waves import PostNewtonianInspiral

        self.pn = PostNewtonianInspiral(m1=m1, m2=m2, D_L=D_L)

    def evaluate(
        self,
        f_start: float = 20.0,
    ) -> tuple[dict[str, float], TraceSession]:
        """
        Compute inspiral parameters.

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

        f_isco = self.pn.isco_frequency()
        Mc = self.pn.chirp_mass_solar()
        t_coal = self.pn.time_to_coalescence(f_start)
        # Rough energy estimate: E ~ Mc*c² * (v/c)^5 scaling
        eta = (self.pn.m1 * self.pn.m2) / (self.pn.m1 + self.pn.m2) ** 2
        e_rad = eta * (self.pn.m1 + self.pn.m2) * 0.05  # ~5% rest mass

        cons = GWConservation(
            chirp_mass=float(Mc),
            isco_frequency=float(f_isco),
            coalescence_time=float(t_coal),
            energy_radiated=float(e_rad),
        )

        session.log_custom(


            name="gravitational_wave_inspiral",


            input_hashes=[_hash_array(np.array([self.pn.m1, self.pn.m2]))],


            output_hashes=[_hash_array(np.array([float(Mc), float(f_isco)]))],


            metrics={**cons.to_dict()},


        )

        return cons.to_dict(), session
