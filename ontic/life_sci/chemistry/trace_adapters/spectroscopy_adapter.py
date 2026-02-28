"""
Spectroscopy Trace Adapter (XV.7)
===================================

Wraps ontic.chemistry.spectroscopy.VibrationalSpectroscopy for STARK tracing.
Conservation: spectral sum rule, peak positions.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class SpectroscopyConservation:
    n_modes: int
    ir_peak_max: float
    raman_peak_max: float
    ir_integral: float
    raman_integral: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n_modes": float(self.n_modes),
            "ir_peak_max": self.ir_peak_max,
            "raman_peak_max": self.raman_peak_max,
            "ir_integral": self.ir_integral,
            "raman_integral": self.raman_integral,
        }


class SpectroscopyTraceAdapter:
    """
    Vibrational spectroscopy adapter with trace logging.

    Modes are added after construction via ``add_mode``.
    """

    def __init__(self) -> None:
        from ontic.life_sci.chemistry.spectroscopy import VibrationalSpectroscopy

        self.solver = VibrationalSpectroscopy()

    def add_mode(
        self,
        k: float,
        mu: float,
        ir_intensity: float = 1.0,
        raman_intensity: float = 1.0,
        label: str = "",
    ) -> None:
        """
        Add a vibrational mode.

        Parameters
        ----------
        k : Force constant (N/m).
        mu : Reduced mass (amu).
        ir_intensity, raman_intensity : Relative intensities.
        label : Mode label.
        """
        self.solver.add_mode(
            k=k, mu=mu,
            ir_intensity=ir_intensity,
            raman_intensity=raman_intensity,
            label=label,
        )

    def evaluate(
        self,
        x_range: tuple[float, float] = (400, 4000),
        sigma: float = 10.0,
        n_pts: int = 1000,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, SpectroscopyConservation, TraceSession]:
        """
        Compute IR and Raman spectra.

        Returns
        -------
        wavenumber_ir, ir_intensity, wavenumber_raman, raman_intensity,
        conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        wn_ir, ir_int = self.solver.ir_spectrum(
            x_range=x_range, sigma=sigma, n_pts=n_pts,
        )
        wn_raman, raman_int = self.solver.raman_spectrum(
            x_range=(max(100, x_range[0] - 300), x_range[1]),
            sigma=sigma, n_pts=n_pts,
        )

        dw_ir = float(wn_ir[1] - wn_ir[0]) if len(wn_ir) > 1 else 1.0
        dw_raman = float(wn_raman[1] - wn_raman[0]) if len(wn_raman) > 1 else 1.0

        cons = SpectroscopyConservation(
            n_modes=len(self.solver.modes),
            ir_peak_max=float(np.max(ir_int)) if len(ir_int) > 0 else 0.0,
            raman_peak_max=float(np.max(raman_int)) if len(raman_int) > 0 else 0.0,
            ir_integral=float(np.sum(ir_int) * dw_ir),
            raman_integral=float(np.sum(raman_int) * dw_raman),
        )

        session.log_custom(


            name="spectroscopy_evaluate",


            input_hashes=[_hash_array(wn_ir)],


            output_hashes=[_hash_array(ir_int), _hash_array(raman_int)],


            metrics=cons.to_dict(),


        )

        return wn_ir, ir_int, wn_raman, raman_int, cons, session
