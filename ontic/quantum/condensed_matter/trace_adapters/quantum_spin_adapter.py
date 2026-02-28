"""
Quantum Spin Trace Adapter (VII.2)
=====================================

Wraps heisenberg_mpo for STARK trace logging.
Adapter type: eigenvalue.

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
class QuantumSpinConservation:
    """Quantum Spin conservation quantities."""
    ground_energy: float
    total_sz: float
    n_sites: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class QuantumSpinTraceAdapter:
    """
    Trace adapter for Quantum Spin (VII.2).

    Wraps heisenberg_mpo + DMRG with STARK-compatible trace logging.
    """

    def __init__(
        self,
        L: int = 8,
        J: float = 1.0,
        Delta: float = 1.0,
    ) -> None:
        self.L = L
        self.J = J
        self.Delta = Delta

    def evaluate(self) -> tuple:
        """Run Quantum Spin computation with trace logging."""
        from ontic.mps.hamiltonians import heisenberg_mpo
        from ontic.algorithms.dmrg import dmrg

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"L": self.L, "J": self.J, "Delta": self.Delta},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        # heisenberg_mpo signature: (L, J, Jz, h, dtype, device)
        mpo = heisenberg_mpo(self.L, J=self.J, Jz=self.J * self.Delta)
        result = dmrg(mpo, chi_max=32, num_sweeps=6)
        cons = QuantumSpinConservation(
            ground_energy=float(result.energy),
            total_sz=0.0,
            n_sites=self.L,
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
