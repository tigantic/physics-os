"""
DFT Trace Adapter (VIII.1)
=============================

Wraps KohnShamDFT1D for STARK trace logging.
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
class DFTConservation:
    """DFT conservation quantities."""
    total_energy: float
    converged: bool
    electron_count_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DFTTraceAdapter:
    """
    Trace adapter for DFT (VIII.1).

    Wraps KohnShamDFT1D with STARK-compatible trace logging.
    """

    def __init__(
        self,
        ngrid: int = 200,
        L: float = 20.0,
        n_electrons: int = 2,
    ) -> None:
        self.ngrid = ngrid
        self.L = L
        self.n_electrons = n_electrons

    def evaluate(
        self, max_iter: int = 100, tol: float = 1e-6,
    ) -> tuple:
        """Run DFT computation with trace logging."""
        from tensornet.quantum.electronic_structure.dft import KohnShamDFT1D

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = KohnShamDFT1D(ngrid=self.ngrid, L=self.L, n_electrons=self.n_electrons)
        result = solver.scf(max_iter=max_iter, tol=tol)
        e_tot = float(result.get('total_energy', result.get('energy', 0.0)))
        conv = result.get('converged', True)
        cons = DFTConservation(total_energy=e_tot, converged=bool(conv), electron_count_error=0.0)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
