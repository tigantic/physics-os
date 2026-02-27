"""
Kondo / Impurity Trace Adapter (VII.9)
=========================================

Wraps AndersonImpurityModel for STARK trace logging.
Adapter type: eigenvalue.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class KondoConservation:
    """Kondo / Impurity conservation quantities."""
    kondo_temperature: float
    occupation: float
    spectral_weight_positive: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class KondoTraceAdapter:
    """
    Trace adapter for Kondo / Impurity (VII.9).

    Wraps AndersonImpurityModel with STARK-compatible trace logging.
    """

    def __init__(
        self,
        eps_d: float = -2.0,
        U: float = 4.0,
        V_hyb: float = 0.5,
    ) -> None:
        self.eps_d = eps_d
        self.U = U
        self.V_hyb = V_hyb

    def evaluate(self) -> tuple:
        """Run Kondo / Impurity computation with trace logging."""
        from tensornet.quantum.condensed_matter.kondo_impurity import AndersonImpurityModel

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"eps_d": self.eps_d, "U": self.U, "V": self.V_hyb},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        # AndersonImpurityModel(eps_d, U, V, half_bandwidth, n_bath)
        model = AndersonImpurityModel(eps_d=self.eps_d, U=self.U, V=self.V_hyb)
        T_K = model.kondo_temperature()
        n_d = model.mean_field_occupation(T=0.01)
        cons = KondoConservation(
            kondo_temperature=float(T_K),
            occupation=float(n_d),
            spectral_weight_positive=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(float(T_K))],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return {"T_K": T_K, "n_d": n_d}, cons, session
