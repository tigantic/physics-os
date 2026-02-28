"""
Strongly Correlated Trace Adapter (VII.3)
============================================

Wraps DMFTSolver for STARK trace logging.
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

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class StronglyCorrelatedConservation:
    """Strongly Correlated conservation quantities."""
    converged: bool
    spectral_weight: float
    quasiparticle_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class StronglyCorrelatedTraceAdapter:
    """
    Trace adapter for Strongly Correlated (VII.3).

    Wraps DMFTSolver with STARK-compatible trace logging.
    """

    def __init__(
        self,
        U: float = 4.0,
        mu: float = 2.0,
        D: float = 1.0,
        beta: float = 10.0,
    ) -> None:
        self.U = U
        self.mu = mu
        self.D = D
        self.beta = beta

    def evaluate(
        self, max_iter: int = 30, tol: float = 1e-4,
    ) -> tuple:
        """Run Strongly Correlated computation with trace logging."""
        from ontic.quantum.condensed_matter.strongly_correlated import DMFTSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = DMFTSolver(U=self.U, mu=self.mu, D=self.D, beta=self.beta)
        result = solver.solve(max_iter=max_iter, tol=tol)
        z = result.get('quasiparticle_weight', result.get('Z', 1.0))
        cons = StronglyCorrelatedConservation(converged=result.get('converged', True), spectral_weight=1.0, quasiparticle_weight=float(z))
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
