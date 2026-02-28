"""
MBL & Disorder Trace Adapter (VII.5)
=======================================

Wraps RandomFieldXXZ for STARK trace logging.
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
class MBLConservation:
    """MBL & Disorder conservation quantities."""
    mean_gap_ratio: float
    is_mbl: bool
    n_eigenvalues: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class MBLTraceAdapter:
    """
    Trace adapter for MBL & Disorder (VII.5).

    Wraps RandomFieldXXZ with STARK-compatible trace logging.
    """

    def __init__(
        self,
        L: int = 8,
        W: float = 5.0,
    ) -> None:
        self.L = L
        self.W = W

    def evaluate(
        self,
    ) -> tuple:
        """Run MBL & Disorder computation with trace logging."""
        from ontic.quantum.condensed_matter.mbl_disorder import RandomFieldXXZ, LevelStatistics

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        model = RandomFieldXXZ(L=self.L, W=self.W, seed=42)
        H = model.build_hamiltonian()
        evals, evecs = model.diagonalize()
        stats = LevelStatistics(evals)
        r = stats.mean_gap_ratio()
        cons = MBLConservation(mean_gap_ratio=float(r), is_mbl=bool(r < 0.45), n_eigenvalues=len(evals))
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return evals, cons, session
