"""
Beyond Standard Model Trace Adapter (X.6)
============================================

Wraps NeutrinoOscillations for STARK trace logging.
Adapter type: algebraic.

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
class BeyondSMConservation:
    """Beyond Standard Model conservation quantities."""
    oscillation_probability_sum: float
    relic_density: float
    unitarity_satisfied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class BeyondSmTraceAdapter:
    """
    Trace adapter for Beyond Standard Model (X.6).

    Wraps NeutrinoOscillations with STARK-compatible trace logging.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self) -> tuple:
        """Run Beyond Standard Model computation with trace logging."""
        from tensornet.applied.particle.beyond_sm import NeutrinoOscillations, DarkMatterRelic

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        osc = NeutrinoOscillations()
        # oscillation_probability(alpha, beta, L_km, E_GeV)
        P_ee = osc.oscillation_probability(0, 0, L_km=1000.0, E_GeV=1.0)
        P_emu = osc.oscillation_probability(0, 1, L_km=1000.0, E_GeV=1.0)
        P_etau = osc.oscillation_probability(0, 2, L_km=1000.0, E_GeV=1.0)
        P_sum = float(P_ee + P_emu + P_etau)
        dm = DarkMatterRelic()
        omega = dm.relic_density()
        cons = BeyondSMConservation(
            oscillation_probability_sum=P_sum,
            relic_density=float(omega),
            unitarity_satisfied=bool(abs(P_sum - 1.0) < 0.05),
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(P_sum)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return {"P_sum": P_sum, "relic_density": omega}, cons, session
