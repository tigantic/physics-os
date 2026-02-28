"""
Thermo-Mechanical Trace Adapter (XVIII.2)
============================================

Wraps ontic.coupled.thermo_mechanical.ThermoelasticSolver for STARK tracing.
Conservation: thermal energy, displacement continuity, stress equilibrium.

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
class ThermoMechanicalConservation:
    vm_stress_max: float
    vm_stress_mean: float
    displacement_max: float
    T_max: float
    T_min: float

    def to_dict(self) -> dict[str, float]:
        return {
            "vm_stress_max": self.vm_stress_max,
            "vm_stress_mean": self.vm_stress_mean,
            "displacement_max": self.displacement_max,
            "T_max": self.T_max,
            "T_min": self.T_min,
        }


class ThermoMechanicalTraceAdapter:
    """
    Thermoelastic solver adapter with trace logging.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    Lx, Ly : float
        Domain size (m).
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha_th : float
        Thermal expansion coefficient (1/K).
    """

    def __init__(
        self,
        nx: int = 50,
        ny: int = 50,
        Lx: float = 1.0,
        Ly: float = 1.0,
        E: float = 200e9,
        nu: float = 0.3,
        alpha_th: float = 12e-6,
    ) -> None:
        from ontic.fluids.coupled.thermo_mechanical import ThermoelasticSolver

        self.solver = ThermoelasticSolver(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, E=E, nu=nu, alpha_th=alpha_th,
        )

    def solve(
        self,
        T_field: NDArray,
        n_iter: int = 3000,
        tol: float = 1e-6,
    ) -> tuple[NDArray, NDArray, ThermoMechanicalConservation, TraceSession]:
        """
        Solve thermoelastic problem for a given temperature field.

        Parameters
        ----------
        T_field : 2-D temperature field (K).
        n_iter : Maximum iterations.
        tol : Convergence tolerance.

        Returns
        -------
        ux, uy, conservation, session
        """
        session = TraceSession()

        self.solver.set_temperature(T_field)

        session.log_custom(
            name="thermoelastic_setup",
            input_hashes=[_hash_array(T_field)],
            output_hashes=[],
            metrics={"T_max": float(np.max(T_field)), "T_min": float(np.min(T_field))},
        )

        n_converged = self.solver.solve(n_iter=n_iter, tol=tol)

        ux = self.solver.ux
        uy = self.solver.uy
        vm = self.solver.von_mises()

        cons = ThermoMechanicalConservation(
            vm_stress_max=float(np.max(vm)),
            vm_stress_mean=float(np.mean(vm)),
            displacement_max=float(np.max(np.sqrt(ux**2 + uy**2))),
            T_max=float(np.max(T_field)),
            T_min=float(np.min(T_field)),
        )

        session.log_custom(
            name="thermoelastic_solve",
            input_hashes=[_hash_array(T_field)],
            output_hashes=[_hash_array(ux), _hash_array(uy)],
            metrics=cons.to_dict(),
        )

        return ux, uy, cons, session
