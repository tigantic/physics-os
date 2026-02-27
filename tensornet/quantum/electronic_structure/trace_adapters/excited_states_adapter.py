"""
Excited States Trace Adapter (VIII.4)
========================================

Wraps CasidaTDDFT for STARK trace logging.
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
class ExcitedStatesConservation:
    """Excited States conservation quantities."""
    n_excitations: int
    lowest_excitation: float
    f_sum: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ExcitedStatesTraceAdapter:
    """
    Trace adapter for Excited States (VIII.4).

    Wraps CasidaTDDFT with STARK-compatible trace logging.
    """

    def __init__(
        self,
        n_occ: int = 2,
        n_virt: int = 8,
    ) -> None:
        self.n_occ = n_occ
        self.n_virt = n_virt

    def evaluate(
        self, n_states: int = 5,
    ) -> tuple:
        """Run Excited States computation with trace logging."""
        from tensornet.quantum.electronic_structure.excited_states import CasidaTDDFT

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        evals = np.concatenate([np.linspace(-5, -1, self.n_occ), np.linspace(1, 5, self.n_virt)])
        solver = CasidaTDDFT(eigenvalues=evals, n_occ=self.n_occ)
        exc_e = solver.excitation_energies(n_states=n_states)
        cons = ExcitedStatesConservation(n_excitations=len(exc_e), lowest_excitation=float(exc_e[0]) if len(exc_e) > 0 else 0.0, f_sum=0.0)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return exc_e, cons, session
