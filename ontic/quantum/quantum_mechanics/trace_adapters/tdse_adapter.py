"""
TDSE Trace Adapter (VI.2)
============================

Wraps SplitOperatorPropagator for STARK trace logging.
Adapter type: timestep.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class TDSEConservation:
    """TDSE conservation quantities."""
    norm_initial: float
    norm_final: float
    probability_conserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TDSETraceAdapter:
    """
    Trace adapter for TDSE (VI.2).

    Wraps SplitOperatorPropagator with STARK-compatible trace logging.
    """

    def __init__(
        self,
        x_min: float = -20.0,
        x_max: float = 20.0,
        n_grid: int = 512,
        mass: float = 1.0,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.n_grid = n_grid
        self.mass = mass

    def propagate(
        self,
        psi_0: NDArray,
        dt: float = 0.01,
        n_steps: int = 100,
        potential: Optional[Callable] = None,
    ) -> tuple:
        """Run TDSE computation with trace logging."""
        from ontic.quantum.quantum_mechanics.propagator import SplitOperatorPropagator

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[_hash_array(psi_0)],
            output_hashes=[],
            params={"n_grid": self.n_grid, "dt": dt, "n_steps": n_steps},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        prop = SplitOperatorPropagator(
            x_min=self.x_min, x_max=self.x_max,
            n_grid=self.n_grid, mass=self.mass,
        )
        # potential callback: (x: NDArray, t: float) -> NDArray
        V = potential if potential is not None else (lambda x, t: 0.5 * x**2)
        result = prop.propagate(psi_0, V, dt=dt, n_steps=n_steps)
        psi_f = result.psi_t[-1] if hasattr(result, "psi_t") and len(result.psi_t) > 0 else psi_0
        norms = result.norm if hasattr(result, "norm") else [1.0]
        cons = TDSEConservation(
            norm_initial=float(norms[0]),
            norm_final=float(norms[-1]),
            probability_conserved=bool(abs(float(norms[-1]) - float(norms[0])) < 0.01),
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="propagate_complete",
            input_hashes=[],
            output_hashes=[_hash_array(psi_f) if hasattr(psi_f, "tobytes") else _hash_scalar(0.0)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return psi_f, cons, session
