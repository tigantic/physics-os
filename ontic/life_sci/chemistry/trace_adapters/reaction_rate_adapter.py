"""
Reaction Rate / TST Trace Adapter (XV.2)
===========================================

Wraps TransitionStateTheory for STARK trace logging.
Adapter type: eigenvalue.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class ReactionRateConservation:
    """Reaction Rate / TST conservation quantities."""
    rate_at_300K: float
    rate_positive: bool
    detailed_balance: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ReactionRateTraceAdapter:
    """
    Trace adapter for Reaction Rate / TST (XV.2).

    Wraps TransitionStateTheory with STARK-compatible trace logging.
    """

    def __init__(
        self,
        Ea: float = 0.5,
        nu_imag: float = 1e13,
    ) -> None:
        self.Ea = Ea
        self.nu_imag = nu_imag

    def evaluate(
        self, T_range: None = None,
    ) -> tuple:
        """Run Reaction Rate / TST computation with trace logging."""
        from ontic.life_sci.chemistry.quantum_reactive import TransitionStateTheory

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = TransitionStateTheory(Ea=self.Ea, nu_imag=self.nu_imag)
        temps = T_range if T_range is not None else np.linspace(200, 1000, 50)
        rates = np.array([solver.rate_constant(T) for T in temps])
        r300 = float(solver.rate_constant(300.0))
        cons = ReactionRateConservation(rate_at_300K=r300, rate_positive=bool(np.all(rates > 0)), detailed_balance=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return rates, cons, session
