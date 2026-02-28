"""
Electrostatics Trace Adapter (III.1)
======================================

Wraps ``PoissonBoltzmannSolver`` from ``ontic.em.electrostatics``.
Conservation: Gauss's law (∇·E = ρ/ε₀), total charge.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession
from ontic.em.electrostatics import PoissonBoltzmannSolver, ElectrostaticResult


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class ElectrostaticsVerification:
    """Verification metrics for electrostatics solve."""
    total_charge: float
    total_energy: float
    gauss_law_residual: float
    max_field: float
    iterations: int

    def to_dict(self) -> dict[str, float]:
        return {
            "total_charge": self.total_charge,
            "total_energy": self.total_energy,
            "gauss_law_residual": self.gauss_law_residual,
            "max_field": self.max_field,
            "iterations": float(self.iterations),
        }


class ElectrostaticsTraceAdapter:
    """
    Trace adapter wrapping ``PoissonBoltzmannSolver``.

    Logs Gauss's law residual, total charge, energy.
    """

    def __init__(self, solver: PoissonBoltzmannSolver) -> None:
        self.solver = solver

    def _verify(self, rho: NDArray, result: ElectrostaticResult) -> ElectrostaticsVerification:
        dx = self.solver.dx
        dV = dx ** len(self.solver.grid_shape)
        total_charge = float(np.sum(rho) * dV)
        total_energy = float(result.energy) if result.energy is not None else 0.0

        # Gauss's law: ∇·E = ρ/ε₀  →  residual = ∫|∇·E - ρ/ε₀| dV
        E = result.electric_field
        if E is not None and len(E) >= 2:
            div_E = np.zeros_like(rho)
            for dim in range(len(E)):
                div_E += np.gradient(E[dim], dx, axis=dim)
            eps_0 = 8.854187817e-12
            gauss_res = float(np.sum(np.abs(div_E - rho / (eps_0 * self.solver.epsilon_r))) * dV)
        else:
            gauss_res = 0.0

        max_field = float(max(np.max(np.abs(e)) for e in E)) if E is not None and len(E) > 0 else 0.0

        return ElectrostaticsVerification(
            total_charge=total_charge,
            total_energy=total_energy,
            gauss_law_residual=gauss_res,
            max_field=max_field,
            iterations=result.iterations if result.iterations is not None else 0,
        )

    def solve(
        self,
        rho: NDArray,
        phi_boundary: NDArray | None = None,
        boundary_mask: NDArray | None = None,
        **kwargs,
    ) -> tuple[ElectrostaticResult, TraceSession]:
        """
        Run Poisson-Boltzmann solve with trace.

        Returns:
            (result, session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[_hash_array(rho)],
            params={"grid_shape": list(self.solver.grid_shape),
                    "dx": self.solver.dx,
                    "epsilon_r": self.solver.epsilon_r},
            metrics={"total_charge": float(np.sum(rho) * self.solver.dx ** len(self.solver.grid_shape))},
        )

        result = self.solver.solve(rho, phi_boundary, boundary_mask, **kwargs)

        t1 = time.perf_counter_ns()
        verification = self._verify(rho, result)

        session.log_custom(
            name="solve_complete",
            input_hashes=[_hash_array(rho)],
            output_hashes=[_hash_array(result.potential)],
            params={"solve_time_ns": t1 - t0},
            metrics=verification.to_dict(),
        )

        return result, session
