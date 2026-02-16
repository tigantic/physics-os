"""
Compact Objects Trace Adapter (XII.2)
======================================

Wraps tensornet.astro.compact_objects.TOVSolver for STARK tracing.
Conservation: TOV mass-energy, geodesic constants.

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
class CompactObjectConservation:
    total_mass_solar: float
    radius_km: float
    central_density: float
    compactness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass_solar": self.total_mass_solar,
            "radius_km": self.radius_km,
            "central_density": self.central_density,
            "compactness": self.compactness,
        }


class CompactObjectsTraceAdapter:
    """
    TOV integration adapter with trace logging.

    Parameters
    ----------
    K : float
        Polytropic constant.
    Gamma : float
        Adiabatic index.
    """

    def __init__(self, K: float = 5.38e9, Gamma: float = 2.34) -> None:
        from tensornet.astro.compact_objects import TOVSolver

        self.solver = TOVSolver(K=K, Gamma=Gamma)

    def solve(
        self,
        rho_c: float = 1e15,
    ) -> tuple[dict[str, float], CompactObjectConservation, TraceSession]:
        """
        Integrate TOV equations for given central density.

        Returns
        -------
        result_dict, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        result = self.solver.integrate(rho_c=rho_c)

        M = result.get("M_solar", result.get("M", 0.0))
        R = result.get("R_km", result.get("R", 0.0))
        G_cgs = 6.674e-8
        c_cgs = 3e10
        M_kg = float(M) * 1.989e33  # solar masses to grams
        R_cm = float(R) * 1e5
        compactness = G_cgs * M_kg / (R_cm * c_cgs**2) if R_cm > 0 else 0.0

        cons = CompactObjectConservation(
            total_mass_solar=float(M),
            radius_km=float(R),
            central_density=float(rho_c),
            compactness=float(compactness),
        )

        session.log_custom(


            name="tov_integrate",


            input_hashes=[_hash_array(np.array([rho_c]))],


            output_hashes=[_hash_array(np.array([float(M), float(R)]))],


            metrics={**cons.to_dict()},


        )

        return result, cons, session
