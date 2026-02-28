"""
Magnetostatics Trace Adapter (III.2)
======================================

Wraps ``MagneticVectorPotential2D`` from ``ontic.em.magnetostatics``.
Conservation: ∇·B = 0 (solenoidal constraint), magnetic flux.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession
from ontic.em.magnetostatics import MagneticVectorPotential2D


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class MagnetostaticsVerification:
    """Verification metrics for magnetostatics solve."""
    total_flux: float
    max_B: float
    div_B_residual: float
    energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_flux": self.total_flux,
            "max_B": self.max_B,
            "div_B_residual": self.div_B_residual,
            "energy": self.energy,
        }


class MagnetostaticsTraceAdapter:
    """
    Trace adapter wrapping ``MagneticVectorPotential2D``.

    Logs ∇·B residual, magnetic flux, energy.
    """

    def __init__(self, solver: MagneticVectorPotential2D) -> None:
        self.solver = solver

    def solve(
        self,
        Jz: NDArray,
        **kwargs,
    ) -> tuple[NDArray, NDArray, NDArray, TraceSession]:
        """
        Solve for vector potential A and magnetic field B = ∇×A.

        Returns:
            (A, Bx, By, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[_hash_array(Jz)],
            params={"nx": self.solver.nx, "ny": self.solver.ny,
                    "Lx": self.solver.Lx, "Ly": self.solver.Ly},
            metrics={"total_current": float(np.sum(Jz) * self.solver.Lx / self.solver.nx *
                                            self.solver.Ly / self.solver.ny)},
        )

        A = self.solver.solve_poisson(Jz, **kwargs)
        Bx, By = self.solver.compute_B()

        t1 = time.perf_counter_ns()

        # Verification: ∇·B = ∂Bx/∂x + ∂By/∂y ≈ 0
        dx = self.solver.Lx / self.solver.nx
        dy = self.solver.Ly / self.solver.ny
        dBx_dx = np.gradient(Bx, dx, axis=0)
        dBy_dy = np.gradient(By, dy, axis=1)
        div_B = dBx_dx + dBy_dy
        div_B_res = float(np.max(np.abs(div_B)))

        dA = dx * dy
        max_B = float(np.max(np.sqrt(Bx**2 + By**2)))
        total_flux = float(np.sum(np.sqrt(Bx**2 + By**2)) * dA)
        mu_0 = 4 * np.pi * 1e-7
        energy = float(np.sum(Bx**2 + By**2) / (2 * mu_0) * dA)

        session.log_custom(
            name="solve_complete",
            input_hashes=[_hash_array(Jz)],
            output_hashes=[_hash_array(A), _hash_array(Bx)],
            params={"solve_time_ns": t1 - t0},
            metrics={
                "total_flux": total_flux,
                "max_B": max_B,
                "div_B_residual": div_B_res,
                "energy": energy,
            },
        )

        return A, Bx, By, session
