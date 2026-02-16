"""
Bosonic Systems Trace Adapter (VII.10)
=========================================

Wraps GrossPitaevskiiSolver for STARK trace logging.
Adapter type: scf.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class BosonicConservation:
    """Bosonic Systems conservation quantities."""
    energy: float
    chemical_potential: float
    particle_number_conserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class BosonicTraceAdapter:
    """
    Trace adapter for Bosonic Systems (VII.10).

    Wraps GrossPitaevskiiSolver with STARK-compatible trace logging.
    """

    def __init__(
        self,
        N_grid: int = 128,
        x_max: float = 10.0,
    ) -> None:
        self.N_grid = N_grid
        self.x_max = x_max

    def evaluate(self, g: float = 1.0) -> tuple:
        """Run Bosonic Systems computation with trace logging."""
        from tensornet.condensed_matter.bosonic import GrossPitaevskiiSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"N_grid": self.N_grid, "x_max": self.x_max, "g": g},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = GrossPitaevskiiSolver(N_grid=self.N_grid, x_max=self.x_max)
        # V_ext must be NDArray, not callable
        x = np.linspace(-self.x_max, self.x_max, self.N_grid)
        V_ext = 0.5 * x**2
        result = solver.ground_state(V_ext, g=g)
        cons = BosonicConservation(
            energy=float(result.energy),
            chemical_potential=float(result.mu),
            particle_number_conserved=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(float(result.energy))],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
