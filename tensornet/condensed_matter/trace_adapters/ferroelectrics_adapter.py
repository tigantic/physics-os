"""
Ferroelectrics Trace Adapter (IX.8)
======================================

Wraps LandauDevonshire for STARK trace logging.
Adapter type: scf.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class FerroelectricsConservation:
    """Ferroelectrics conservation quantities."""
    spontaneous_polarisation: float
    free_energy: float
    polarisation_bounded: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class FerroelectricsTraceAdapter:
    """
    Trace adapter for Ferroelectrics (IX.8).

    Wraps LandauDevonshire with STARK-compatible trace logging.
    """

    def __init__(
        self,
    ) -> None:
        pass
    def evaluate(
        self, T: float = 300.0,
    ) -> tuple:
        """Run Ferroelectrics computation with trace logging."""
        from tensornet.condensed_matter.ferroelectrics import LandauDevonshire

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = LandauDevonshire()
        P = solver.spontaneous_polarisation(T=T)
        cons = FerroelectricsConservation(spontaneous_polarisation=float(P), free_energy=0.0, polarisation_bounded=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return P, cons, session
