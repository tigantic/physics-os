"""
Photochemistry Trace Adapter (XV.4)
======================================

Wraps tensornet.chemistry.photochemistry.FranckCondonFactors and InternalConversion
for STARK tracing. Conservation: FC sum rule (Σ FC factors = 1).

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
class PhotochemistryConservation:
    fc_sum: float
    fc_sum_error: float
    huang_rhys: float

    def to_dict(self) -> dict[str, float]:
        return {
            "fc_sum": self.fc_sum,
            "fc_sum_error": self.fc_sum_error,
            "huang_rhys": self.huang_rhys,
        }


class PhotochemistryTraceAdapter:
    """
    Franck-Condon / internal conversion adapter with trace logging.

    Parameters
    ----------
    S : float
        Huang-Rhys factor.
    """

    def __init__(self, S: float = 1.0) -> None:
        from tensornet.life_sci.chemistry.photochemistry import FranckCondonFactors

        self.solver = FranckCondonFactors(S=S)
        self.S = S

    def evaluate(
        self,
        v_max: int = 20,
    ) -> tuple[NDArray, PhotochemistryConservation, TraceSession]:
        """
        Compute Franck-Condon progression up to v_max.

        Returns
        -------
        fc_spectrum, conservation, session
        """
        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        spectrum = self.solver.spectrum(v_max=v_max)
        fc_sum = float(np.sum(spectrum))

        cons = PhotochemistryConservation(
            fc_sum=fc_sum,
            fc_sum_error=abs(fc_sum - 1.0),
            huang_rhys=self.S,
        )

        session.log_custom(


            name="fc_evaluate",


            input_hashes=[_hash_scalar(self.S)],


            output_hashes=[_hash_array(spectrum)],


            metrics={"v_max": v_max, **cons.to_dict()},


        )

        return spectrum, cons, session

    def ic_rate(
        self,
        E_gap: float = 2.0,
        V_el: float = 0.01,
        omega_M: float = 1400.0,
        T: float = 300.0,
    ) -> tuple[float, TraceSession]:
        """
        Compute internal conversion rate.

        Parameters
        ----------
        E_gap : Energy gap (eV).
        V_el : Electronic coupling (eV).
        omega_M : Accepting-mode frequency (cm⁻¹).
        T : Temperature (K).

        Returns
        -------
        rate, session
        """
        from tensornet.life_sci.chemistry.photochemistry import InternalConversion

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )
        ic = InternalConversion(E_gap=E_gap, V_el=V_el, omega_M=omega_M, S=self.S)
        rate = ic.rate(T=T)

        session.log_custom(


            name="ic_rate_evaluate",


            input_hashes=[_hash_scalar(E_gap), _hash_scalar(T)],


            output_hashes=[_hash_scalar(rate)],


            metrics={"E_gap": E_gap, "V_el": V_el, "T": T, "rate": rate},


        )

        return rate, session
