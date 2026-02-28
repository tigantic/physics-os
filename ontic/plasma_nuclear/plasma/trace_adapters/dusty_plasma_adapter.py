"""
Dusty Plasma Trace Adapter (XI.7)
====================================

Wraps ``DustAcousticWave`` / ``OMLGrainCharging`` from
``ontic.plasma.dusty_plasmas``.
Conservation: particle count, total energy (analytical models).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession
from ontic.plasma_nuclear.plasma.dusty_plasmas import DustyPlasmaParams, DustAcousticWave, OMLGrainCharging


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class DustyPlasmaVerification:
    """Verification metrics for dusty plasma models."""
    coupling_parameter: float
    debye_length: float
    dust_plasma_freq: float
    is_crystalline: bool

    def to_dict(self) -> dict[str, float]:
        return {
            "coupling_parameter": self.coupling_parameter,
            "debye_length": self.debye_length,
            "dust_plasma_freq": self.dust_plasma_freq,
            "is_crystalline": float(self.is_crystalline),
        }


class DustyPlasmaTraceAdapter:
    """
    Trace adapter for dusty plasma analytical models.
    """

    def __init__(self, params: DustyPlasmaParams) -> None:
        self.params = params
        self.daw = DustAcousticWave(params)
        self.oml = OMLGrainCharging(params)

    def evaluate(self, k_range: NDArray | None = None) -> tuple[dict[str, float], TraceSession]:
        """
        Evaluate dusty plasma diagnostics with trace.

        Returns:
            (metrics_dict, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        p = self.params

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_d": p.n_d, "T_d": p.T_d, "T_e": p.T_e,
                    "T_i": p.T_i, "r_d": p.r_d, "Z_d": p.Z_d},
            metrics={},
        )

        metrics = {
            "coupling_parameter": p.Gamma,
            "kappa": p.kappa,
            "debye_length": p.lambda_D,
            "dust_plasma_freq": p.omega_pd,
            "is_crystalline": float(p.is_crystalline),
        }

        # Dispersion relation
        if k_range is None:
            k_range = np.linspace(0.01, 5.0, 50) / p.lambda_D

        omega = self.daw.dispersion(k_range)
        metrics["max_omega"] = float(np.max(np.real(omega)))
        metrics["max_growth"] = float(np.max(np.imag(omega)))

        t1 = time.perf_counter_ns()

        session.log_custom(
            name="evaluation_complete",
            input_hashes=[],
            output_hashes=[_hash_array(np.real(omega))],
            params={"compute_time_ns": t1 - t0, "n_k_points": len(k_range)},
            metrics=metrics,
        )

        return metrics, session
