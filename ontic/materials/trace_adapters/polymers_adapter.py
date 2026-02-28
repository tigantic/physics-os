"""
Polymers / Soft Matter Trace Adapter (XIV.6)
==============================================

Wraps ontic.materials.polymers_soft_matter.SCFT1D for STARK tracing.
Conservation: incompressibility (φ_A + φ_B = 1).

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
class PolymersConservation:
    phi_sum_max: float
    phi_sum_min: float
    phi_sum_mean: float
    incompressibility_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "phi_sum_max": self.phi_sum_max,
            "phi_sum_min": self.phi_sum_min,
            "phi_sum_mean": self.phi_sum_mean,
            "incompressibility_error": self.incompressibility_error,
        }


class PolymersTraceAdapter:
    """
    SCFT (self-consistent field theory) adapter with trace logging.

    Parameters
    ----------
    n_grid : int
        Number of grid points.
    L : float
        Box length.
    N : int
        Chain length.
    f : float
        A-block fraction.
    chi_N : float
        Flory-Huggins interaction parameter × chain length.
    """

    def __init__(
        self,
        n_grid: int = 64,
        L: float = 10.0,
        N: int = 100,
        f: float = 0.5,
        chi_N: float = 20.0,
    ) -> None:
        from ontic.materials.polymers_soft_matter import SCFT1D

        self.solver = SCFT1D(n_grid=n_grid, L=L, N=N, f=f, chi_N=chi_N)

    def solve(
        self,
        max_iter: int = 200,
        tol: float = 1e-5,
    ) -> tuple[NDArray, NDArray, int, PolymersConservation, TraceSession]:
        """
        Run SCFT to self-consistency.

        Returns
        -------
        phiA, phiB, n_iterations, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        n_iter = self.solver.iterate(max_iter=max_iter, tol=tol)

        phiA = self.solver.phiA
        phiB = self.solver.phiB
        phi_sum = phiA + phiB
        cons = PolymersConservation(
            phi_sum_max=float(np.max(phi_sum)),
            phi_sum_min=float(np.min(phi_sum)),
            phi_sum_mean=float(np.mean(phi_sum)),
            incompressibility_error=float(np.max(np.abs(phi_sum - 1.0))),
        )

        session.log_custom(
            name="scft_solve",
            input_hashes=[_hash_array(self.solver.wA), _hash_array(self.solver.wB)],
            output_hashes=[_hash_array(phiA), _hash_array(phiB)],
            metrics=cons.to_dict(),
        )

        return phiA, phiB, n_iter, cons, session
