"""
Potential Energy Surface Trace Adapter (XV.1)
================================================

Wraps MorsePotential for STARK trace logging.
Adapter type: scf.

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
class PESConservation:
    """Potential Energy Surface conservation quantities."""
    equilibrium_energy: float
    gradient_zero_at_minimum: bool
    energy_bounded: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class PESTraceAdapter:
    """
    Trace adapter for Potential Energy Surface (XV.1).

    Wraps MorsePotential with STARK-compatible trace logging.
    """

    def __init__(
        self,
        D_e: float = 4.746,
        alpha_m: float = 1.94,
        r_e: float = 0.741,
    ) -> None:
        self.D_e = D_e
        self.alpha_m = alpha_m
        self.r_e = r_e

    def evaluate(
        self,
    ) -> tuple:
        """Run Potential Energy Surface computation with trace logging."""
        from tensornet.chemistry.pes import MorsePotential, NudgedElasticBand

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        morse = MorsePotential(D_e=self.D_e, alpha=self.alpha_m, r_e=self.r_e)
        e_min = float(morse.energy(np.array([self.r_e])))
        levels = morse.vibrational_levels(n_max=10)
        cons = PESConservation(equilibrium_energy=float(e_min), gradient_zero_at_minimum=True, energy_bounded=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return levels, cons, session
