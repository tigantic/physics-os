"""
Magnetic Reconnection Trace Adapter (XI.5)
=============================================

Wraps ``SweetParkerReconnection`` / ``PetschekReconnection`` from
``ontic.plasma.magnetic_reconnection``.
Conservation: total energy, magnetic flux (analytical models).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from ontic.core.trace import TraceSession
from ontic.plasma_nuclear.plasma.magnetic_reconnection import (
    SweetParkerReconnection,
    PetschekReconnection,
)


@dataclass
class ReconnectionVerification:
    """Verification metrics for reconnection model."""
    reconnection_rate: float
    energy_release_rate: float
    lundquist_number: float
    alfven_speed: float

    def to_dict(self) -> dict[str, float]:
        return {
            "reconnection_rate": self.reconnection_rate,
            "energy_release_rate": self.energy_release_rate,
            "lundquist_number": self.lundquist_number,
            "alfven_speed": self.alfven_speed,
        }


class ReconnectionTraceAdapter:
    """
    Trace adapter for magnetic reconnection analytical models.

    Wraps Sweet-Parker and Petschek models.
    """

    def __init__(self, model: SweetParkerReconnection | PetschekReconnection) -> None:
        self.model = model

    def evaluate(self) -> tuple[dict[str, float], TraceSession]:
        """
        Evaluate reconnection model with trace.

        Returns:
            (metrics_dict, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        model_type = type(self.model).__name__

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"model": model_type,
                    "B0": self.model.B0,
                    "n": self.model.n,
                    "eta": self.model.eta,
                    "L": self.model.L},
            metrics={},
        )

        metrics = {
            "v_A": self.model.v_A,
            "lundquist_number": self.model.lundquist_number,
            "reconnection_rate": self.model.reconnection_rate,
            "energy_release_rate": self.model.energy_release_rate,
        }

        if hasattr(self.model, "current_sheet_thickness"):
            metrics["current_sheet_thickness"] = self.model.current_sheet_thickness
        if hasattr(self.model, "inflow_velocity"):
            metrics["inflow_velocity"] = self.model.inflow_velocity

        t1 = time.perf_counter_ns()

        # Verification: SP rate ~ S^{-1/2}, Petschek rate ~ 1/(ln S)
        S = self.model.lundquist_number
        if model_type == "SweetParkerReconnection":
            expected_rate = 1.0 / np.sqrt(S)
            scaling_error = abs(self.model.reconnection_rate - expected_rate) / max(expected_rate, 1e-30)
        else:
            expected_rate = np.pi / (8 * np.log(S)) if S > 1 else 1.0
            scaling_error = abs(self.model.reconnection_rate - expected_rate) / max(expected_rate, 1e-30)

        metrics["scaling_error"] = scaling_error

        param_hash = hashlib.sha256(
            f"{self.model.B0}_{self.model.n}_{self.model.eta}_{self.model.L}".encode()
        ).hexdigest()

        session.log_custom(
            name="evaluation_complete",
            input_hashes=[],
            output_hashes=[param_hash],
            params={"compute_time_ns": t1 - t0, "model": model_type},
            metrics=metrics,
        )

        return metrics, session
