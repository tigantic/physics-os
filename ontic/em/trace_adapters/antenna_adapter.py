"""
Antenna & Microwave Trace Adapter (III.7)
==========================================

Wraps ``DipoleAntenna`` and ``UniformLinearArray`` from
``ontic.em.antenna_microwave``.
Conservation: radiated power, directivity consistency.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession
from ontic.em.antenna_microwave import DipoleAntenna


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class AntennaVerification:
    """Verification metrics for antenna pattern computation."""
    directivity: float
    radiation_resistance: float
    max_gain: float
    pattern_integral: float

    def to_dict(self) -> dict[str, float]:
        return {
            "directivity": self.directivity,
            "radiation_resistance": self.radiation_resistance,
            "max_gain": self.max_gain,
            "pattern_integral": self.pattern_integral,
        }


class AntennaTraceAdapter:
    """
    Trace adapter wrapping ``DipoleAntenna``.

    Verifies radiated power consistency and directivity.
    """

    def __init__(self, antenna: DipoleAntenna) -> None:
        self.antenna = antenna

    def compute_pattern(
        self,
        n_theta: int = 361,
    ) -> tuple[NDArray, NDArray, TraceSession]:
        """
        Compute radiation pattern with trace.

        Returns:
            (theta, pattern, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"length": self.antenna.length, "freq": self.antenna.freq,
                    "n_theta": n_theta},
            metrics={},
        )

        theta = np.linspace(0, np.pi, n_theta)
        pattern = self.antenna.pattern(theta)

        t1 = time.perf_counter_ns()

        directivity = self.antenna.directivity(n_theta=n_theta)
        R_rad = self.antenna.radiation_resistance()

        # Pattern integral: ∫U(θ) sin(θ) dθ (normalised → 4π for isotropic)
        d_theta = np.pi / (n_theta - 1)
        pattern_integral = float(np.sum(pattern * np.sin(theta)) * d_theta * 2 * np.pi)
        max_gain = float(np.max(pattern))

        session.log_custom(
            name="pattern_complete",
            input_hashes=[],
            output_hashes=[_hash_array(pattern)],
            params={"compute_time_ns": t1 - t0},
            metrics={
                "directivity": directivity,
                "radiation_resistance": R_rad,
                "max_gain": max_gain,
                "pattern_integral": pattern_integral,
            },
        )

        return theta, pattern, session
