"""
Computational Photonics Trace Adapter (III.6)
===============================================

Wraps ``TransferMatrix1D`` from ``tensornet.em.computational_photonics``.
Conservation: photon number (|r|² + |t|² = 1), energy flux.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession
from tensornet.em.computational_photonics import TransferMatrix1D


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class PhotonicsVerification:
    """Verification metrics for transfer matrix computation."""
    reflectance: float
    transmittance: float
    unitarity_error: float
    n_layers: int

    def to_dict(self) -> dict[str, float]:
        return {
            "reflectance": self.reflectance,
            "transmittance": self.transmittance,
            "unitarity_error": self.unitarity_error,
            "n_layers": float(self.n_layers),
        }


class PhotonicsTraceAdapter:
    """
    Trace adapter wrapping ``TransferMatrix1D`` photonics solver.

    Verifies photon number conservation: R + T = 1.
    """

    def __init__(self, solver: TransferMatrix1D) -> None:
        self.solver = solver

    def compute(
        self,
        wavelength: float,
        polarisation: str = "TE",
    ) -> tuple[dict[str, float], TraceSession]:
        """
        Run transfer matrix computation with trace.

        Returns:
            (result_dict, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        n_layers = len(self.solver.layers) if hasattr(self.solver, "layers") else 0

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"wavelength": wavelength, "polarisation": polarisation,
                    "n_substrate": self.solver.n_substrate,
                    "n_superstrate": self.solver.n_superstrate,
                    "n_layers": n_layers},
            metrics={},
        )

        result = self.solver.compute(wavelength, polarisation=polarisation)

        t1 = time.perf_counter_ns()

        R = result.get("R", 0.0)
        T = result.get("T", 0.0)
        unitarity_error = abs(R + T - 1.0)

        session.log_custom(
            name="compute_complete",
            input_hashes=[],
            output_hashes=[],
            params={"compute_time_ns": t1 - t0},
            metrics={
                "reflectance": R,
                "transmittance": T,
                "unitarity_error": unitarity_error,
                "n_layers": float(n_layers),
            },
        )

        return result, session

    def sweep(
        self,
        wavelengths: NDArray,
        polarisation: str = "TE",
    ) -> tuple[NDArray, NDArray, TraceSession]:
        """
        Wavelength sweep with trace.

        Returns:
            (R_array, T_array, session)
        """
        session = TraceSession()

        session.log_custom(
            name="sweep_input",
            input_hashes=[],
            output_hashes=[_hash_array(wavelengths)],
            params={"n_wavelengths": len(wavelengths),
                    "wl_min": float(wavelengths[0]),
                    "wl_max": float(wavelengths[-1])},
            metrics={},
        )

        R_arr = np.zeros(len(wavelengths))
        T_arr = np.zeros(len(wavelengths))

        for i, wl in enumerate(wavelengths):
            result = self.solver.compute(wl, polarisation=polarisation)
            R_arr[i] = result.get("R", 0.0)
            T_arr[i] = result.get("T", 0.0)

        max_unitarity_error = float(np.max(np.abs(R_arr + T_arr - 1.0)))

        session.log_custom(
            name="sweep_complete",
            input_hashes=[_hash_array(wavelengths)],
            output_hashes=[_hash_array(R_arr), _hash_array(T_arr)],
            params={"n_wavelengths": len(wavelengths)},
            metrics={
                "mean_R": float(np.mean(R_arr)),
                "mean_T": float(np.mean(T_arr)),
                "max_unitarity_error": max_unitarity_error,
            },
        )

        return R_arr, T_arr, session
