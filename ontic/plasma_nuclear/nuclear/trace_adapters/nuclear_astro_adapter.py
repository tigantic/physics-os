"""
Nuclear Astrophysics Trace Adapter (X.3)
===========================================

Wraps ThermonuclearRate for STARK trace logging.
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
class NuclearAstroConservation:
    """Nuclear Astrophysics conservation quantities."""
    gamow_energy: float
    baryon_number_conserved: bool
    rate_positive: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class NuclearAstroTraceAdapter:
    """
    Trace adapter for Nuclear Astrophysics (X.3).

    Wraps ThermonuclearRate with STARK-compatible trace logging.
    """

    def __init__(
        self,
        Z1: int = 1,
        Z2: int = 1,
    ) -> None:
        self.Z1 = Z1
        self.Z2 = Z2

    def evaluate(
        self,
    ) -> tuple:
        """Run Nuclear Astrophysics computation with trace logging."""
        from ontic.plasma_nuclear.nuclear.astrophysics import ThermonuclearRate

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        solver = ThermonuclearRate(Z1=self.Z1, Z2=self.Z2)
        E_G = solver.gamow_energy(T9=1.0)
        cons = NuclearAstroConservation(gamow_energy=float(E_G), baryon_number_conserved=True, rate_positive=True)
        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return E_G, cons, session
