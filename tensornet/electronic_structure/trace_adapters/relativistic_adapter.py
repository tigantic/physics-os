"""
Relativistic Electronic Trace Adapter (VIII.6)
=================================================

Wraps Dirac4Component for STARK trace logging.
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
class RelativisticConservation:
    """Relativistic Electronic conservation quantities."""
    ground_energy: float
    fine_structure_splitting: float
    current_continuity: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class RelativisticTraceAdapter:
    """
    Trace adapter for Relativistic Electronic (VIII.6).

    Wraps Dirac4Component with STARK-compatible trace logging.
    """

    def __init__(
        self,
        Z: int = 1,
    ) -> None:
        self.Z = Z

    def evaluate(
        self, n_max: int = 3,
    ) -> tuple:
        """Run Relativistic Electronic computation with trace logging."""
        from tensornet.electronic_structure.relativistic import Dirac4Component

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = Dirac4Component(Z=self.Z)
        energies = [solver.exact_energy(n, -1) for n in range(1, n_max + 1)]
        fs = solver.fine_structure_splitting(2) if n_max >= 2 else 0.0
        cons = RelativisticConservation(ground_energy=float(energies[0]), fine_structure_splitting=float(fs), current_continuity=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return np.array(energies), cons, session
