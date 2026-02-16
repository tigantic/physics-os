"""
Stellar Structure Trace Adapter (XII.1)
========================================

Wraps tensornet.astro.stellar_structure.StellarStructure for STARK tracing.
Conservation: hydrostatic equilibrium, luminosity profile.

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
class StellarConservation:
    total_mass: float
    central_pressure: float
    surface_luminosity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "central_pressure": self.central_pressure,
            "surface_luminosity": self.surface_luminosity,
        }


class StellarStructureTraceAdapter:
    """
    Stellar structure integration adapter with trace logging.

    Parameters
    ----------
    M_star : float
        Stellar mass (kg). Default: solar mass.
    """

    def __init__(self, M_star: float = 1.989e30) -> None:
        from tensornet.astro.stellar_structure import StellarStructure

        self.solver = StellarStructure(M_star=M_star)

    def solve(
        self,
        n_shells: int = 500,
        rho_c: float = 1.5e2,
        T_c: float = 1.5e7,
    ) -> tuple[dict[str, NDArray], StellarConservation, TraceSession]:
        """
        Integrate stellar structure equations.

        Returns
        -------
        profiles, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        profiles = self.solver.integrate(
            n_shells=n_shells, rho_c=rho_c, T_c=T_c
        )

        r = profiles.get("r", np.array([0.0]))
        m = profiles.get("m", np.array([0.0]))
        P = profiles.get("P", np.array([0.0]))
        L = profiles.get("L", np.array([0.0]))

        cons = StellarConservation(
            total_mass=float(m[-1]) if len(m) > 0 else 0.0,
            central_pressure=float(P[0]) if len(P) > 0 else 0.0,
            surface_luminosity=float(L[-1]) if len(L) > 0 else 0.0,
        )

        session.log_custom(


            name="stellar_structure_integrate",


            input_hashes=[_hash_array(np.array([rho_c, T_c]))],


            output_hashes=[_hash_array(r), _hash_array(m)],


            metrics={"n_shells": n_shells, **cons.to_dict()},


        )

        return profiles, cons, session
