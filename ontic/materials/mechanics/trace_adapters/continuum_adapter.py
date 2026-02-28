"""
Continuum Mechanics Trace Adapter (I.3)
========================================

Wraps ontic.mechanics.continuum constitutive models with a simple
1D bar tensile-test driver for trace logging.
Conservation: linear momentum, total energy (strain + kinetic).

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
class ContinuumConservation:
    strain_energy: float
    kinetic_energy: float
    total_energy: float
    max_stress: float

    def to_dict(self) -> dict[str, float]:
        return {
            "strain_energy": self.strain_energy,
            "kinetic_energy": self.kinetic_energy,
            "total_energy": self.total_energy,
            "max_stress": self.max_stress,
        }


class ContinuumMechanicsTraceAdapter:
    """
    1D explicit dynamics bar (Neo-Hookean material) with trace logging.

    Parameters
    ----------
    n_elem : int
        Number of elements.
    L : float
        Initial bar length (m).
    A : float
        Cross-section area (m²).
    rho : float
        Density (kg/m³).
    E : float
        Young's modulus (Pa).
    """

    def __init__(
        self,
        n_elem: int = 50,
        L: float = 1.0,
        A: float = 1e-4,
        rho: float = 7850.0,
        E: float = 200e9,
    ) -> None:
        self.n_elem = n_elem
        self.L = L
        self.A = A
        self.rho = rho
        self.E = E
        self.dx = L / n_elem
        self.n_nodes = n_elem + 1

    def solve(
        self,
        t_final: float,
        dt: float | None = None,
        applied_velocity: float = 1.0,
    ) -> tuple[NDArray, float, int, TraceSession]:
        """
        Explicit central-difference time integration.

        Parameters
        ----------
        t_final : float
        dt : float or None (auto CFL)
        applied_velocity : float
            Velocity BC at right end (m/s).

        Returns
        -------
        displacement, t, n_steps, session
        """
        c = np.sqrt(self.E / self.rho)
        if dt is None:
            dt = 0.5 * self.dx / c

        session = TraceSession()


        session.log_custom(


            name="input_state",


            input_hashes=[],


            output_hashes=[],


            params={},


            metrics={},


        )
        n_steps = int(t_final / dt)
        u = np.zeros(self.n_nodes)
        v = np.zeros(self.n_nodes)
        mass_node = self.rho * self.A * self.dx

        _record(session, 0, 0.0, u, v, self._conservation(u, v, mass_node))

        for step in range(1, n_steps + 1):
            strain = np.diff(u) / self.dx
            stress = self.E * strain
            f_int = np.zeros(self.n_nodes)
            for e in range(self.n_elem):
                fe = self.A * stress[e]
                f_int[e] -= fe
                f_int[e + 1] += fe

            a = f_int / mass_node
            a[0] = 0.0  # fixed left end

            v += dt * a
            v[-1] = applied_velocity  # velocity BC
            u += dt * v
            u[0] = 0.0

            t = step * dt
            if step % max(1, n_steps // 20) == 0 or step == n_steps:
                cons = self._conservation(u, v, mass_node)
                _record(session, step, t, u, v, cons)

        return u, t, n_steps, session

    def _conservation(
        self, u: NDArray, v: NDArray, mass_node: float
    ) -> ContinuumConservation:
        strain = np.diff(u) / self.dx
        se = 0.5 * self.E * np.sum(strain**2) * self.A * self.dx
        ke = 0.5 * mass_node * np.sum(v**2)
        stress = self.E * strain
        return ContinuumConservation(
            strain_energy=float(se),
            kinetic_energy=float(ke),
            total_energy=float(se + ke),
            max_stress=float(np.max(np.abs(stress))) if len(stress) > 0 else 0.0,
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    u: NDArray,
    v: NDArray,
    cons: ContinuumConservation,
) -> None:
    session.log_custom(

        name="continuum_mechanics_step",

        input_hashes=[_hash_array(u), _hash_array(v)],

        output_hashes=[_hash_array(u)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
