"""
Classical Magnetism Trace Adapter (IX.3)
===========================================

Wraps LandauLifshitzGilbert for STARK trace logging.
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
class ClassicalMagConservation:
    """Classical Magnetism conservation quantities."""
    m_magnitude_initial: float
    m_magnitude_final: float
    magnitude_conserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ClassicalMagnetismTraceAdapter:
    """
    Trace adapter for Classical Magnetism (IX.3).

    Wraps LandauLifshitzGilbert with STARK-compatible trace logging.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        Ms: float = 8e5,
    ) -> None:
        self.alpha = alpha
        self.Ms = Ms

    def evaluate(
        self, n_steps: int = 500, dt: float = 1e-12,
    ) -> tuple:
        """Run Classical Magnetism computation with trace logging."""
        from ontic.quantum.condensed_matter.classical_magnetism import LandauLifshitzGilbert

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = LandauLifshitzGilbert(alpha=self.alpha, Ms=self.Ms)
        m0 = np.array([0.0, 0.0, 1.0])
        H_ext = np.array([0.0, 0.0, 1e6])
        m_hist = solver.evolve(m0, H_ext, dt=dt, n_steps=n_steps)
        m_f = m_hist[-1] if len(m_hist) > 0 else m0
        mag_i = float(np.linalg.norm(m0))
        mag_f = float(np.linalg.norm(m_f))
        cons = ClassicalMagConservation(m_magnitude_initial=mag_i, m_magnitude_final=mag_f, magnitude_conserved=bool(abs(mag_f - mag_i) < 0.01))
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return m_hist, cons, session
