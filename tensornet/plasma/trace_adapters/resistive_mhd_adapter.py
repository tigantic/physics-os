"""
Resistive MHD Trace Adapter (XI.2)
====================================

Wraps ``HallMHDSolver1D`` (with η>0) from ``tensornet.plasma.extended_mhd``.
Conservation: magnetic helicity, total energy (with Ohmic dissipation).

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession
from tensornet.plasma.extended_mhd import HallMHDSolver1D


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class ResistiveMHDConservation:
    """Conservation quantities for resistive MHD."""
    total_mass: float
    total_energy: float
    magnetic_energy: float
    kinetic_energy: float
    magnetic_helicity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "total_energy": self.total_energy,
            "magnetic_energy": self.magnetic_energy,
            "kinetic_energy": self.kinetic_energy,
            "magnetic_helicity": self.magnetic_helicity,
        }


class ResistiveMHDTraceAdapter:
    """
    Trace adapter wrapping ``HallMHDSolver1D`` with η > 0.
    """

    def __init__(self, solver: HallMHDSolver1D) -> None:
        self.solver = solver

    def _compute_conservation(self) -> ResistiveMHDConservation:
        U = self.solver.U
        dx = self.solver.Lx / self.solver.nx
        rho = U[:, 0]
        mom_x, mom_y, mom_z = U[:, 1], U[:, 2], U[:, 3]
        Bx, By, Bz = U[:, 4], U[:, 5], U[:, 6]
        e = U[:, 7]

        total_mass = float(np.sum(rho) * dx)
        total_E = float(np.sum(e) * dx)
        mag_E = float(0.5 * np.sum(Bx**2 + By**2 + Bz**2) * dx)

        # KE = ½ρv² = (ρv)²/(2ρ)
        rho_safe = np.maximum(rho, 1e-30)
        kin_E = float(0.5 * np.sum((mom_x**2 + mom_y**2 + mom_z**2) / rho_safe) * dx)

        # Magnetic helicity proxy in 1D: ∫A·B dx ≈ cumsum(By)·By + cumsum(Bz)·Bz
        Ay = np.cumsum(Bz) * dx  # A_y ~ ∫B_z dx
        Az = -np.cumsum(By) * dx  # A_z ~ -∫B_y dx
        helicity = float(np.sum(Ay * By + Az * Bz) * dx)

        return ResistiveMHDConservation(
            total_mass=total_mass,
            total_energy=total_E,
            magnetic_energy=mag_E,
            kinetic_energy=kin_E,
            magnetic_helicity=helicity,
        )

    def step(self, dt: float, session: TraceSession | None = None) -> None:
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(self.solver.U)

        self.solver.step(dt)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation()
            session.log_custom(
                name="resistive_mhd_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(self.solver.U)],
                params={"dt": dt, "eta": self.solver.eta},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

    def solve(
        self,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[float, int, TraceSession]:
        if dt is None:
            dx = self.solver.Lx / self.solver.nx
            dt = 0.3 * dx

        session = TraceSession()

        cons0 = self._compute_conservation()
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(self.solver.U)],
            params={"nx": self.solver.nx, "Lx": self.solver.Lx, "eta": self.solver.eta},
            metrics=cons0.to_dict(),
        )

        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            self.step(dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation()
        session.log_custom(
            name="final_state",
            input_hashes=[_hash_array(self.solver.U)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "mass_drift": abs(cons_f.total_mass - cons0.total_mass) /
                                   max(abs(cons0.total_mass), 1e-30)},
        )

        return t, n_steps, session
