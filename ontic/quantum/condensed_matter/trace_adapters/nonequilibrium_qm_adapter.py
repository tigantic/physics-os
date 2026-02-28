"""
Non-Equilibrium QM Trace Adapter (VII.8)
===========================================

Wraps FloquetSolver for STARK trace logging.
Adapter type: timestep.

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
class NonEqQMConservation:
    """Non-Equilibrium QM conservation quantities."""
    n_quasi_energies: int
    unitarity_error: float
    bounded: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class NonEquilibriumQMTraceAdapter:
    """
    Trace adapter for Non-Equilibrium QM (VII.8).

    Wraps FloquetSolver with STARK-compatible trace logging.
    """

    def __init__(
        self,
        dim: int = 4,
    ) -> None:
        self.dim = dim

    def evaluate(
        self, T_period: float = 1.0,
    ) -> tuple:
        """Run Non-Equilibrium QM computation with trace logging."""
        from ontic.quantum.condensed_matter.nonequilibrium_qm import FloquetSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = FloquetSolver(dim=self.dim)
        H_static = np.diag(np.arange(self.dim, dtype=float))
        V_drive = 0.1 * (np.ones((self.dim, self.dim)) - np.eye(self.dim))
        H_func = lambda t: H_static + V_drive * np.cos(2 * np.pi * t / T_period)
        U_F = solver.stroboscopic_propagator(H_func, T_period)
        qe = solver.quasi_energies(U_F, T_period)
        uu = float(np.max(np.abs(U_F @ U_F.conj().T - np.eye(self.dim))))
        cons = NonEqQMConservation(n_quasi_energies=len(qe), unitarity_error=uu, bounded=bool(uu < 1e-8))
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return qe, cons, session
