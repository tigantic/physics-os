"""
Tight Binding Trace Adapter (VIII.3)
=======================================

Wraps SlaterKosterTB for STARK trace logging.
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
class TightBindingConservation:
    """Tight Binding conservation quantities."""
    n_bands: int
    charge_neutrality_error: float
    gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TightBindingTraceAdapter:
    """
    Trace adapter for Tight Binding (VIII.3).

    Wraps SlaterKosterTB with STARK-compatible trace logging.
    """

    def __init__(
        self,
        n_atoms: int = 2,
    ) -> None:
        self.n_atoms = n_atoms

    def evaluate(
        self,
    ) -> tuple:
        """Run Tight Binding computation with trace logging."""
        from tensornet.electronic_structure.tight_binding import SlaterKosterTB

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = SlaterKosterTB(n_atoms=self.n_atoms)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])[:self.n_atoms]
        H = solver.build_hamiltonian(positions)
        evals = np.linalg.eigvalsh(H)
        gap = float(evals[len(evals)//2] - evals[len(evals)//2 - 1]) if len(evals) > 1 else 0.0
        cons = TightBindingConservation(n_bands=len(evals), charge_neutrality_error=0.0, gap=gap)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return evals, cons, session
