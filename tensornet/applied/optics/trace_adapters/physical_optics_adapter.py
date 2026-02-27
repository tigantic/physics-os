"""
Physical Optics Trace Adapter (IV.1)
=====================================

Wraps tensornet.optics.physical_optics.FresnelPropagator for STARK tracing.
Conservation: Poynting flux (total intensity), optical coherence.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class OpticsConservation:
    total_intensity: float
    peak_intensity: float
    coherence_area: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_intensity": self.total_intensity,
            "peak_intensity": self.peak_intensity,
            "coherence_area": self.coherence_area,
        }


class PhysicalOpticsTraceAdapter:
    """
    Fresnel propagation adapter with trace logging.

    Wraps FresnelPropagator from tensornet.applied.optics.physical_optics.

    Parameters
    ----------
    wavelength : float
        Wavelength (m).
    grid_size : int
        Number of grid points per side.
    pixel_pitch : float
        Grid spacing (m).
    """

    def __init__(
        self,
        wavelength: float = 632.8e-9,
        grid_size: int = 256,
        pixel_pitch: float = 10e-6,
    ) -> None:
        from tensornet.applied.optics.physical_optics import FresnelPropagator

        self.propagator = FresnelPropagator(
            wavelength=wavelength,
            grid_size=grid_size,
            pixel_pitch=pixel_pitch,
        )
        self.grid_size = grid_size

    def propagate(
        self,
        U0: NDArray,
        distances: list[float],
    ) -> tuple[NDArray, list[OpticsConservation], TraceSession]:
        """
        Propagate field through multiple distances.

        Parameters
        ----------
        U0 : (grid_size, grid_size) complex input field.
        distances : list of propagation distances (m).

        Returns
        -------
        U_final, conservation_list, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )
        U = U0.copy().astype(np.complex128)
        cons_list: list[OpticsConservation] = []

        I0 = np.abs(U) ** 2
        cons = OpticsConservation(
            total_intensity=float(np.sum(I0)),
            peak_intensity=float(np.max(I0)),
            coherence_area=float(np.sum(I0 > 0.5 * np.max(I0))),
        )
        cons_list.append(cons)
        _record(session, 0, 0.0, U, cons)

        for i, z in enumerate(distances):
            U = self.propagator.propagate(U, z)
            I_out = np.abs(U) ** 2
            cons = OpticsConservation(
                total_intensity=float(np.sum(I_out)),
                peak_intensity=float(np.max(I_out)),
                coherence_area=float(np.sum(I_out > 0.5 * np.max(I_out))),
            )
            cons_list.append(cons)
            _record(session, i + 1, z, U, cons)

        return U, cons_list, session


def _record(
    session: TraceSession,
    step: int,
    z: float,
    U: NDArray,
    cons: OpticsConservation,
) -> None:
    session.log_custom(

        name="physical_optics_propagate",

        input_hashes=[_hash_array(np.abs(U))],

        output_hashes=[_hash_array(np.abs(U))],

        metrics={"step": step, "distance": z, **cons.to_dict()},

    )
