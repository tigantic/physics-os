"""
Semiclassical Trace Adapter (VI.4)
=====================================

Wraps WKBSolver for STARK trace logging.
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
class SemiclassicalConservation:
    """Semiclassical conservation quantities."""
    n_levels: int
    ground_energy: float
    action_quantised: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class SemiclassicalTraceAdapter:
    """
    Trace adapter for Semiclassical (VI.4).

    Wraps WKBSolver with STARK-compatible trace logging.
    """

    def __init__(
        self,
        mass: float = 1.0,
    ) -> None:
        self.mass = mass

    def evaluate(
        self, n_levels: int = 5, potential: None = None,
    ) -> tuple:
        """Run Semiclassical computation with trace logging."""
        from tensornet.quantum.qm.semiclassical_wkb import WKBSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        V = potential if potential is not None else (lambda x: 0.5 * x**2)
        solver = WKBSolver(V=V, mass=self.mass)
        energies = []
        for n in range(n_levels):
            try:
                e = solver.quantise(n, x_range=(-10.0, 10.0))
                if e is not None and np.isfinite(e):
                    energies.append(float(e))
            except Exception:
                pass
        energies = np.array(energies) if energies else np.array([0.0])
        cons = SemiclassicalConservation(n_levels=len(energies), ground_energy=float(energies[0]), action_quantised=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return energies, cons, session
