"""
Open Quantum Systems Trace Adapter (VII.7)
=============================================

Wraps LindbladSolver for STARK trace logging.
Adapter type: timestep.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class OpenQuantumConservation:
    """Open Quantum Systems conservation quantities."""
    trace_rho_initial: float
    trace_rho_final: float
    positivity: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class OpenQuantumTraceAdapter:
    """
    Trace adapter for Open Quantum Systems (VII.7).

    Wraps LindbladSolver with STARK-compatible trace logging.
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def evaluate(self, n_steps: int = 100, dt: float = 0.1) -> tuple:
        """Run Open Quantum Systems computation with trace logging."""
        from ontic.quantum.condensed_matter.open_quantum import LindbladSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"dim": self.dim, "n_steps": n_steps, "dt": dt},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        H = np.diag(np.arange(self.dim, dtype=float))
        L_ops = [0.1 * np.diag(np.sqrt(np.arange(1, self.dim, dtype=float)), k=-1)]
        solver = LindbladSolver(H, L_ops)
        rho0 = np.zeros((self.dim, self.dim), dtype=complex)
        rho0[0, 0] = 1.0
        # LindbladSolver.evolve(rho_0, t_final, dt, save_interval, method)
        t_final = n_steps * dt
        result = solver.evolve(rho0, t_final=t_final, dt=dt)
        rho_list = result["density_matrices"]
        rho_f = rho_list[-1]
        tr_f = float(np.real(np.trace(rho_f)))
        eigvals = np.real(np.linalg.eigvalsh(rho_f))
        cons = OpenQuantumConservation(
            trace_rho_initial=1.0,
            trace_rho_final=tr_f,
            positivity=bool(np.all(eigvals >= -1e-10)),
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return rho_f, cons, session
