"""
Beyond-DFT Trace Adapter (VIII.2)
====================================

Wraps RestrictedHartreeFock for STARK trace logging.
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
class BeyondDFTConservation:
    """Beyond-DFT conservation quantities."""
    total_energy: float
    converged: bool
    correlation_energy: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class BeyondDFTTraceAdapter:
    """
    Trace adapter for Beyond-DFT (VIII.2).

    Wraps RestrictedHartreeFock with STARK-compatible trace logging.
    """

    def __init__(
        self,
        n_basis: int = 10,
        n_electrons: int = 2,
    ) -> None:
        self.n_basis = n_basis
        self.n_electrons = n_electrons

    def evaluate(self, max_iter: int = 100, tol: float = 1e-8) -> tuple:
        """Run Beyond-DFT computation with trace logging."""
        from tensornet.electronic_structure.beyond_dft import RestrictedHartreeFock

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_basis": self.n_basis, "n_electrons": self.n_electrons},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = RestrictedHartreeFock(
            n_basis=self.n_basis, n_electrons=self.n_electrons,
        )
        # Must provide integrals before SCF
        n = self.n_basis
        rng = np.random.RandomState(42)
        h_core = -np.eye(n) + 0.1 * rng.randn(n, n)
        h_core = 0.5 * (h_core + h_core.T)  # symmetrise
        eri = np.zeros((n, n, n, n))
        for p in range(n):
            for q in range(n):
                eri[p, q, p, q] = 0.5
                eri[p, p, q, q] = 0.3
        solver.set_integrals(h_core, eri)
        result = solver.scf(max_iter=max_iter, tol=tol)
        e_tot = float(result.get("E_electronic", 0.0))
        converged = bool(result.get("converged", False))
        cons = BeyondDFTConservation(
            total_energy=e_tot,
            converged=converged,
            correlation_energy=0.0,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(e_tot)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
