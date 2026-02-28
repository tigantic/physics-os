"""
Laser-Plasma Trace Adapter (XI.6)
====================================

Wraps ``StimulatedRamanScattering`` from ``ontic.plasma.laser_plasma``.
Conservation: photon number, energy (analytical models).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from ontic.core.trace import TraceSession
from ontic.plasma_nuclear.plasma.laser_plasma import LaserPlasmaParams, StimulatedRamanScattering


@dataclass
class LaserPlasmaVerification:
    """Verification metrics for laser-plasma interaction."""
    growth_rate: float
    frequency_ratio: float
    n_over_n_critical: float
    energy_conservation: float

    def to_dict(self) -> dict[str, float]:
        return {
            "growth_rate": self.growth_rate,
            "frequency_ratio": self.frequency_ratio,
            "n_over_n_critical": self.n_over_n_critical,
            "energy_conservation": self.energy_conservation,
        }


class LaserPlasmaTraceAdapter:
    """
    Trace adapter for SRS laser-plasma interaction models.
    """

    def __init__(self, srs: StimulatedRamanScattering) -> None:
        self.srs = srs

    def evaluate(self, L_n: float = 1e-3) -> tuple[dict[str, float], TraceSession]:
        """
        Evaluate laser-plasma interaction with trace.

        Returns:
            (metrics_dict, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        params = self.srs.params

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_e": params.n_e, "T_e": params.T_e,
                    "lambda_0": params.lambda_0, "I_laser": params.I_laser,
                    "L_n": L_n},
            metrics={},
        )

        # Compute key diagnostics
        omega_s, omega_epw = self.srs.frequency_matching()
        gamma = self.srs.growth_rate()
        G = self.srs.convective_gain(L_n)

        metrics = {
            "growth_rate": gamma,
            "omega_scattered": omega_s,
            "omega_epw": omega_epw,
            "convective_gain": G,
            "n_over_n_critical": params.n_e / params.n_critical,
            "frequency_ratio": omega_s / params.omega_0 if params.omega_0 > 0 else 0.0,
        }

        # Energy conservation: ω₀ = ω_s + ω_epw
        freq_sum = omega_s + omega_epw
        energy_conservation = abs(freq_sum - params.omega_0) / max(params.omega_0, 1e-30)
        metrics["energy_conservation"] = energy_conservation

        t1 = time.perf_counter_ns()

        param_hash = hashlib.sha256(
            f"{params.n_e}_{params.T_e}_{params.lambda_0}_{params.I_laser}".encode()
        ).hexdigest()

        session.log_custom(
            name="evaluation_complete",
            input_hashes=[],
            output_hashes=[param_hash],
            params={"compute_time_ns": t1 - t0},
            metrics=metrics,
        )

        return metrics, session
