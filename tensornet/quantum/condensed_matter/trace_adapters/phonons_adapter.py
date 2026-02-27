"""
Phonons Trace Adapter (IX.1)
===============================

Wraps DynamicalMatrix for STARK trace logging.
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

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class PhononConservation:
    """Phonons conservation quantities."""
    n_modes: int
    acoustic_sum_rule_error: float
    all_real_frequencies: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class PhononsTraceAdapter:
    """
    Trace adapter for Phonons (IX.1).

    Wraps DynamicalMatrix with STARK-compatible trace logging.
    """

    def __init__(self, n_atoms: int = 2) -> None:
        self.n_atoms = n_atoms

    def evaluate(self) -> tuple:
        """Run Phonons computation with trace logging."""
        from tensornet.quantum.condensed_matter.phonons import DynamicalMatrix

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_atoms": self.n_atoms},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        n = self.n_atoms
        d = 1  # 1D chain
        masses = np.ones(n)
        spring_k = 10.0
        onsite = np.eye(n * d) * spring_k * 2
        neighbor = -np.eye(n * d) * spring_k
        # force_constants: dict mapping R-tuples to Phi matrices
        fc = {
            (0,): onsite,
            (1,): neighbor,
            (-1,): neighbor,
        }
        lattice_vectors = np.array([[1.0]])
        solver = DynamicalMatrix(
            masses=masses,
            force_constants=fc,
            lattice_vectors=lattice_vectors,
        )
        q_points = np.linspace(0, np.pi, 50).reshape(-1, 1)
        band = solver.dispersion(q_points)
        freqs = band.frequencies
        asr_err = float(np.min(np.abs(freqs[0]))) if len(freqs) > 0 else 0.0
        n_modes = freqs.shape[1] if freqs.ndim == 2 else len(freqs)
        all_real = bool(np.all(np.isreal(freqs)))
        cons = PhononConservation(
            n_modes=n_modes,
            acoustic_sum_rule_error=asr_err,
            all_real_frequencies=all_real,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return band, cons, session
