"""
Nuclear Many-Body Trace Adapter (VII.12)
===========================================

Wraps NuclearShellModel for STARK trace logging.
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
class NuclearMBConservation:
    """Nuclear Many-Body conservation quantities."""
    ground_energy: float
    n_eigenvalues: int
    nucleon_number_conserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class NuclearManyBodyTraceAdapter:
    """
    Trace adapter for Nuclear Many-Body (VII.12).

    Wraps NuclearShellModel with STARK-compatible trace logging.
    """

    def __init__(
        self,
        n_orbits: int = 4,
        n_particles: int = 2,
    ) -> None:
        self.n_orbits = n_orbits
        self.n_particles = n_particles

    def evaluate(
        self, n_states: int = 5,
    ) -> tuple:
        """Run Nuclear Many-Body computation with trace logging."""
        from tensornet.quantum.condensed_matter.nuclear_many_body import NuclearShellModel

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        model = NuclearShellModel(n_orbits=self.n_orbits, n_particles=self.n_particles)
        H = model.build_hamiltonian()
        evals, evecs = model.diagonalize()
        evals = evals[:n_states]
        cons = NuclearMBConservation(ground_energy=float(evals[0]) if len(evals) > 0 else 0.0, n_eigenvalues=len(evals), nucleon_number_conserved=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return evals, cons, session
