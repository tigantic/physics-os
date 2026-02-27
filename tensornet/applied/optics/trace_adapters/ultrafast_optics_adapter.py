"""
Ultrafast Optics Trace Adapter (IV.4)
=======================================

Wraps tensornet.optics.ultrafast_optics.SplitStepFourier for STARK tracing.
Conservation: pulse energy, photon number.

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
class UltrafastConservation:
    pulse_energy: float
    peak_power: float
    spectral_width: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pulse_energy": self.pulse_energy,
            "peak_power": self.peak_power,
            "spectral_width": self.spectral_width,
        }


class UltrafastOpticsTraceAdapter:
    """
    Split-step Fourier NLSE propagation adapter with trace logging.

    Parameters
    ----------
    n_t : int
        Number of temporal grid points.
    t_window : float
        Temporal window (ps).
    beta2 : float
        GVD parameter (ps²/km).
    gamma : float
        Nonlinear parameter (1/(W·km)).
    n_z : int
        Number of propagation steps.
    z_max : float
        Total propagation distance (km).
    """

    def __init__(
        self,
        n_t: int = 2**10,
        t_window: float = 10.0,
        beta2: float = -20.0,
        gamma: float = 1.0,
        n_z: int = 200,
        z_max: float = 1.0,
    ) -> None:
        from tensornet.applied.optics.ultrafast_optics import SplitStepFourier

        self.ssf = SplitStepFourier(
            n_t=n_t, t_window=t_window, beta2=beta2,
            gamma=gamma, n_z=n_z, z_max=z_max,
        )
        self.n_t = n_t
        self.n_z = n_z

    def solve(
        self,
        A0: NDArray,
    ) -> tuple[NDArray, UltrafastConservation, TraceSession]:
        """
        Propagate pulse through fibre.

        Parameters
        ----------
        A0 : (n_t,) complex pulse envelope.

        Returns
        -------
        A_final, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        E0 = float(np.sum(np.abs(A0) ** 2))
        cons0 = UltrafastConservation(
            pulse_energy=E0,
            peak_power=float(np.max(np.abs(A0) ** 2)),
            spectral_width=float(np.std(np.abs(np.fft.fft(A0)) ** 2)),
        )
        _record(session, 0, 0.0, A0, cons0)

        result = self.ssf.propagate(A0)
        if isinstance(result, tuple):
            # propagate returns (z_array, A_final)
            A_final = result[-1]
        else:
            A_final = result

        Ef = float(np.sum(np.abs(A_final) ** 2))
        cons_f = UltrafastConservation(
            pulse_energy=Ef,
            peak_power=float(np.max(np.abs(A_final) ** 2)),
            spectral_width=float(np.std(np.abs(np.fft.fft(A_final)) ** 2)),
        )
        _record(session, 1, self.ssf.z_max if hasattr(self.ssf, "z_max") else 1.0,
                A_final, cons_f)

        return A_final, cons_f, session


def _record(
    session: TraceSession,
    step: int,
    z: float,
    A: NDArray,
    cons: UltrafastConservation,
) -> None:
    session.log_custom(

        name="ultrafast_optics_propagate",

        input_hashes=[_hash_array(np.abs(A))],

        output_hashes=[_hash_array(np.abs(A))],

        metrics={"step": step, "distance": z, **cons.to_dict()},

    )
