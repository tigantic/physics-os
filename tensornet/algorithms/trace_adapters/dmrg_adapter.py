"""
DMRG Trace Adapter (VII.1)
=============================

Wraps dmrg for STARK trace logging.
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
class DMRGConservation:
    """DMRG conservation quantities."""
    ground_energy: float
    converged: bool
    bond_dim: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DMRGTraceAdapter:
    """
    Trace adapter for DMRG (VII.1).

    Wraps dmrg with STARK-compatible trace logging.
    """

    def __init__(
        self,
        chi_max: int = 32,
        num_sweeps: int = 10,
    ) -> None:
        self.chi_max = chi_max
        self.num_sweeps = num_sweeps

    def evaluate(
        self, H_mpo: Any,
    ) -> tuple:
        """Run DMRG computation with trace logging."""
        from tensornet.algorithms.dmrg import dmrg, DMRGResult

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        result = dmrg(H_mpo, chi_max=self.chi_max, num_sweeps=self.num_sweeps)
        cons = DMRGConservation(ground_energy=float(result.energy), converged=bool(len(result.energies) > 1 and abs(result.energies[-1] - result.energies[-2]) < 1e-8), bond_dim=self.chi_max)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
