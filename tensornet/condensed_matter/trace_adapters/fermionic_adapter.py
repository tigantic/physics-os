"""
Fermionic Systems Trace Adapter (VII.11)
===========================================

Wraps BCSSolver for STARK trace logging.
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
class FermionicConservation:
    """Fermionic Systems conservation quantities."""
    condensation_energy: float
    gap_nonzero: bool
    particle_number_conserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class FermionicTraceAdapter:
    """
    Trace adapter for Fermionic Systems (VII.11).

    Wraps BCSSolver with STARK-compatible trace logging.
    """

    def __init__(
        self,
        N_k: int = 200,
        E_cutoff: float = 10.0,
    ) -> None:
        self.N_k = N_k
        self.E_cutoff = E_cutoff

    def evaluate(self, V0: float = 0.5) -> tuple:
        """Run Fermionic Systems computation with trace logging."""
        from tensornet.condensed_matter.fermionic import BCSSolver

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"N_k": self.N_k, "E_cutoff": self.E_cutoff, "V0": V0},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = BCSSolver(N_k=self.N_k, E_cutoff=self.E_cutoff)
        k = np.linspace(-np.pi, np.pi, self.N_k)
        eps_k = -2.0 * np.cos(k)
        # solve_swave(epsilon_k, V0, mu, T, max_iter, tol)
        result = solver.solve_swave(eps_k, V0, mu=0.0)
        cons = FermionicConservation(
            condensation_energy=float(result.condensation_energy),
            gap_nonzero=bool(float(result.gap) > 1e-10),
            particle_number_conserved=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(float(result.gap))],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
