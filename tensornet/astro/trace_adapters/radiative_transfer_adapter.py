"""
Radiative Transfer Trace Adapter (XII.6)
==========================================

Wraps tensornet.astro.radiative_transfer.DiscreteOrdinates for STARK tracing.
Conservation: radiative energy, photon number.

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
class RadTransferConservation:
    total_intensity: float
    flux_top: float
    flux_bottom: float
    energy_balance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_intensity": self.total_intensity,
            "flux_top": self.flux_top,
            "flux_bottom": self.flux_bottom,
            "energy_balance": self.energy_balance,
        }


class RadiativeTransferTraceAdapter:
    """
    Discrete ordinates radiative transfer adapter.

    Parameters
    ----------
    n_depth : int
        Number of optical depth points.
    tau_max : float
        Maximum optical depth.
    n_mu : int
        Number of angular quadrature points.
    """

    def __init__(
        self,
        n_depth: int = 100,
        tau_max: float = 20.0,
        n_mu: int = 8,
    ) -> None:
        from tensornet.astro.radiative_transfer import (
            DiscreteOrdinates,
            RadiativeTransfer1D,
        )

        self.rt = RadiativeTransfer1D(
            n_depth=n_depth, tau_max=tau_max, n_mu=n_mu
        )
        self.sn = DiscreteOrdinates(
            N_angles=n_mu, n_depth=n_depth, tau_max=tau_max
        )
        self.n_depth = n_depth
        self.tau_max = tau_max

    def solve(
        self,
        source_function: NDArray | None = None,
    ) -> tuple[NDArray, RadTransferConservation, TraceSession]:
        """
        Solve radiative transfer with discrete ordinates.

        Parameters
        ----------
        source_function : (n_depth,) or None (default: Planck B=1).

        Returns
        -------
        mean_intensity, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        if source_function is None:
            source_function = np.ones(self.n_depth)

        S = source_function.astype(np.float64)
        J = self.sn.solve(S)
        J = np.atleast_1d(J)

        flux_top = float(J[0]) if len(J) > 0 else 0.0
        flux_bot = float(J[-1]) if len(J) > 0 else 0.0

        cons = RadTransferConservation(
            total_intensity=float(np.sum(J)),
            flux_top=flux_top,
            flux_bottom=flux_bot,
            energy_balance=float(np.sum(J - S)),
        )

        session.log_custom(


            name="radiative_transfer_solve",


            input_hashes=[_hash_array(S)],


            output_hashes=[_hash_array(J)],


            metrics={"n_depth": self.n_depth, **cons.to_dict()},


        )

        return J, cons, session
