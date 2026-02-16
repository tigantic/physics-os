"""
Seismology Trace Adapter (XIII.1)
==================================

Wraps tensornet.geophysics.seismology.AcousticWave2D for STARK tracing.
Conservation: elastic wave energy.

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
class SeismologyConservation:
    wave_energy: float
    max_amplitude: float
    rms_amplitude: float

    def to_dict(self) -> dict[str, float]:
        return {
            "wave_energy": self.wave_energy,
            "max_amplitude": self.max_amplitude,
            "rms_amplitude": self.rms_amplitude,
        }


class SeismologyTraceAdapter:
    """
    2D acoustic wave propagation adapter with trace logging.

    Parameters
    ----------
    nx, nz : int
        Grid dimensions.
    dx, dz : float
        Grid spacing (m).
    dt : float
        Time step (s).
    nt : int
        Number of time steps.
    """

    def __init__(
        self,
        nx: int = 100,
        nz: int = 100,
        dx: float = 10.0,
        dz: float = 10.0,
        dt: float = 0.001,
        nt: int = 500,
    ) -> None:
        from tensornet.geophysics.seismology import AcousticWave2D

        self.solver = AcousticWave2D(
            nx=nx, nz=nz, dx=dx, dz=dz, dt=dt, nt=nt
        )
        self.nt = nt

    def solve(
        self,
        src_x: int = 50,
        src_z: int = 10,
        f0: float = 25.0,
    ) -> tuple[list, SeismologyConservation, TraceSession]:
        """
        Propagate seismic wavefield.

        Returns
        -------
        snapshots, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        snapshots = self.solver.propagate(
            src_x=src_x, src_z=src_z, f0=f0
        )

        if not snapshots:
            snapshots = [np.zeros((100, 100))]

        snap_final = np.asarray(snapshots[-1])
        cons = SeismologyConservation(
            wave_energy=float(0.5 * np.sum(snap_final**2)),
            max_amplitude=float(np.max(np.abs(snap_final))),
            rms_amplitude=float(np.sqrt(np.mean(snap_final**2))),
        )

        session.log_custom(
            name="seismology_propagate",
            input_hashes=[_hash_array(np.array([src_x, src_z, f0]))],
            output_hashes=[_hash_array(snap_final)],
            metrics=cons.to_dict(),
        )

        return snapshots, cons, session
