"""
Ceramics Trace Adapter (XIV.7)
================================

Wraps tensornet.materials.ceramics.SinteringModel for STARK tracing.
Conservation: thermodynamic consistency of neck growth kinetics.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.array(v).tobytes()).hexdigest()


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class CeramicsConservation:
    neck_ratio: float
    neck_monotone: bool
    densification_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "neck_ratio": self.neck_ratio,
            "neck_monotone": float(self.neck_monotone),
            "densification_rate": self.densification_rate,
        }


class CeramicsTraceAdapter:
    """
    Sintering model adapter with trace logging.

    Parameters
    ----------
    mechanism : str
        Sintering mechanism: 'surface', 'volume', 'gb', or 'evap'.
    a : float
        Particle radius (m).
    """

    MECHANISM_MAP = {
        "surface": "SURFACE_DIFFUSION",
        "volume": "VOLUME_DIFFUSION",
        "gb": "GB_DIFFUSION",
        "evap": "EVAP_CONDENSATION",
    }

    def __init__(
        self,
        mechanism: str = "volume",
        a: float = 1e-6,
    ) -> None:
        from tensornet.materials.ceramics import SinteringModel

        mech_attr = self.MECHANISM_MAP.get(mechanism, "VOLUME_DIFFUSION")
        mech = getattr(SinteringModel, mech_attr)
        self.solver = SinteringModel(mechanism=mech, a=a)

    def evaluate(
        self,
        times: NDArray,
        T: float = 1573.0,
    ) -> tuple[NDArray, list[CeramicsConservation], TraceSession]:
        """
        Evaluate neck growth x/a over time at temperature T.

        Parameters
        ----------
        times : 1-D array of times (s).
        T : Temperature (K).

        Returns
        -------
        neck_ratios, conservation_list, session
        """
        session = TraceSession()

        ratios = np.array([self.solver.neck_ratio(float(t), T) for t in times])

        prev_ratio = 0.0
        cons_list: list[CeramicsConservation] = []
        for i, t_val in enumerate(times):
            r = float(ratios[i])
            dens = self.solver.densification_rate(rho=0.65, T=T)
            cons = CeramicsConservation(
                neck_ratio=r,
                neck_monotone=r >= prev_ratio - 1e-12,
                densification_rate=dens,
            )
            cons_list.append(cons)
            session.log_custom(

                name="sintering_evaluate",

                input_hashes=[_hash_scalar(float(t_val)), _hash_scalar(T)],

                output_hashes=[_hash_scalar(r)],

                metrics={"time": float(t_val), "T": T, **cons.to_dict()},

            )
            prev_ratio = r

        return ratios, cons_list, session

    def master_sintering(
        self,
        t_arr: NDArray,
        T_arr: NDArray,
    ) -> tuple[NDArray, TraceSession]:
        """
        Compute master sintering curve Θ(t, T).

        Returns
        -------
        theta, session
        """
        session = TraceSession()
        theta = self.solver.master_sintering_curve_theta(t_arr, T_arr)

        session.log_custom(


            name="msc_evaluate",


            input_hashes=[_hash_array(t_arr), _hash_array(T_arr)],


            output_hashes=[_hash_array(theta)],


            metrics={"n_points": len(t_arr), "T_max": float(np.max(T_arr))},


        )

        return theta, session
