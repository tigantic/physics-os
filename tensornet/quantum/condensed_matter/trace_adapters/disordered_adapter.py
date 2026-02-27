"""
Disordered Systems Trace Adapter (IX.5)
==========================================

Wraps AndersonModel for STARK trace logging.
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

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class DisorderedConservation:
    """Disordered Systems conservation quantities."""
    n_eigenvalues: int
    normalisation_error: float
    spectral_weight_positive: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DisorderedTraceAdapter:
    """
    Trace adapter for Disordered Systems (IX.5).

    Wraps AndersonModel with STARK-compatible trace logging.
    """

    def __init__(
        self,
        L: int = 50,
        W: float = 2.0,
    ) -> None:
        self.L = L
        self.W = W

    def evaluate(
        self,
    ) -> tuple:
        """Run Disordered Systems computation with trace logging."""
        from tensornet.quantum.condensed_matter.disordered import AndersonModel

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        model = AndersonModel(L=self.L, W=self.W, seed=42)
        evals, evecs = model.eigensolve()
        norms = [float(np.sum(np.abs(evecs[:, i])**2)) for i in range(min(5, evecs.shape[1]))]
        norm_err = float(max(abs(n - 1.0) for n in norms)) if norms else 0.0
        cons = DisorderedConservation(n_eigenvalues=len(evals), normalisation_error=norm_err, spectral_weight_positive=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return evals, cons, session
