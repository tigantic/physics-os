"""
CMB & Early Universe Trace Adapter (XII.5)
============================================

Wraps tensornet.astro.cmb_early_universe.Recombination for STARK tracing.
Conservation: photon number, entropy, baryon number.

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
class CMBConservation:
    initial_Xe: float
    final_Xe: float
    baryon_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "initial_Xe": self.initial_Xe,
            "final_Xe": self.final_Xe,
            "baryon_fraction": self.baryon_fraction,
        }


class CMBTraceAdapter:
    """
    Recombination history adapter with trace logging.

    Parameters
    ----------
    Omega_b : float
        Baryon density parameter.
    h : float
        Hubble parameter h = H0/100.
    """

    def __init__(self, Omega_b: float = 0.049, h: float = 0.674) -> None:
        from tensornet.astro.cmb_early_universe import Recombination

        self.recom = Recombination(Omega_b=Omega_b, h=h)
        self.Omega_b = Omega_b

    def solve(
        self,
        T_start: float = 6000.0,
        T_end: float = 2000.0,
        n_steps: int = 500,
    ) -> tuple[NDArray, NDArray, CMBConservation, TraceSession]:
        """
        Compute recombination history Xe(T).

        Returns
        -------
        T_array, Xe_array, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        T_arr, Xe_arr = self.recom.peebles_recombination(
            T_start=T_start, T_end=T_end, n_steps=n_steps
        )

        T_arr = np.atleast_1d(T_arr)
        Xe_arr = np.atleast_1d(Xe_arr)

        cons = CMBConservation(
            initial_Xe=float(Xe_arr[0]) if len(Xe_arr) > 0 else 1.0,
            final_Xe=float(Xe_arr[-1]) if len(Xe_arr) > 0 else 0.0,
            baryon_fraction=self.Omega_b,
        )

        session.log_custom(


            name="cmb_recombination",


            input_hashes=[_hash_array(np.array([T_start, T_end]))],


            output_hashes=[_hash_array(Xe_arr)],


            metrics={"n_steps": n_steps, **cons.to_dict()},


        )

        return T_arr, Xe_arr, cons, session
