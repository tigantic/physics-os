"""
TISE Trace Adapter (VI.1)
============================

Wraps DVRSolver for STARK trace logging.
Adapter type: eigenvalue.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class TISEConservation:
    """TISE conservation quantities."""
    n_eigenvalues: int
    ground_energy: float
    norm_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TISETraceAdapter:
    """
    Trace adapter for TISE (VI.1).

    Wraps DVRSolver with STARK-compatible trace logging.
    """

    def __init__(
        self,
        x_min: float = -10.0,
        x_max: float = 10.0,
        n_grid: int = 200,
        mass: float = 1.0,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.n_grid = n_grid
        self.mass = mass

    def solve(
        self,
        n_states: int = 5,
        potential: Optional[Callable] = None,
    ) -> tuple:
        """Run TISE computation with trace logging."""
        from tensornet.quantum_mechanics.stationary import DVRSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_grid": self.n_grid, "n_states": n_states},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = DVRSolver(
            x_min=self.x_min, x_max=self.x_max,
            n_grid=self.n_grid, mass=self.mass,
        )
        V = potential if potential is not None else (lambda x: 0.5 * x**2)
        result = solver.solve(V, n_states=n_states)
        evals = result.energies[:n_states]
        wf = result.wavefunctions  # shape: (n_states, n_grid)
        dx = (self.x_max - self.x_min) / self.n_grid

        norms = []
        n_wf = wf.shape[0] if wf.ndim == 2 else 1
        for i in range(min(n_states, n_wf)):
            psi_i = wf[i] if wf.ndim == 2 else wf
            norm_sq = float(np.sum(np.abs(psi_i) ** 2) * dx)
            norms.append(abs(norm_sq - 1.0))
        norm_err = float(max(norms)) if norms else 0.0

        cons = TISEConservation(
            n_eigenvalues=len(evals),
            ground_energy=float(evals[0]) if len(evals) > 0 else 0.0,
            norm_error=norm_err,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="solve_complete",
            input_hashes=[],
            output_hashes=[_hash_array(evals)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return evals, cons, session
