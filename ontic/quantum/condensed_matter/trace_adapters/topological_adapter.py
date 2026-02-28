"""
Topological Phases Trace Adapter (VII.4)
===========================================

Wraps ChernNumberCalculator for STARK trace logging.
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
class TopologicalConservation:
    """Topological Phases conservation quantities."""
    chern_number: float
    is_integer: bool
    gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TopologicalTraceAdapter:
    """
    Trace adapter for Topological Phases (VII.4).

    Wraps ChernNumberCalculator with STARK-compatible trace logging.
    """

    def __init__(
        self,
        nk: int = 20,
    ) -> None:
        self.nk = nk

    def evaluate(
        self,
    ) -> tuple:
        """Run Topological Phases computation with trace logging."""
        from ontic.quantum.condensed_matter.topological_phases import ChernNumberCalculator

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        calc = ChernNumberCalculator(nk=self.nk)
        C = calc.chern_number_fhs(calc.haldane_model, M=0.5, t2=0.3, phi=np.pi/2)
        cons = TopologicalConservation(chern_number=float(C), is_integer=bool(abs(C - round(C)) < 0.1), gap=1.0)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return C, cons, session
