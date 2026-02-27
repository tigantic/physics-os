"""
Ideal MHD Trace Adapter (XI.1)
=================================

Wraps ``HallMHDSolver1D`` (with η=0) from ``tensornet.plasma.extended_mhd``.
Conservation: mass, momentum, energy, ∇·B = 0.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession
from tensornet.plasma_nuclear.plasma.extended_mhd import HallMHDSolver1D


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class IdealMHDConservation:
    """Conservation quantities for ideal MHD."""
    total_mass: float
    total_momentum: float
    total_energy: float
    max_divB: float
    magnetic_energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "total_momentum": self.total_momentum,
            "total_energy": self.total_energy,
            "max_divB": self.max_divB,
            "magnetic_energy": self.magnetic_energy,
        }


class IdealMHDTraceAdapter:
    """
    Trace adapter wrapping ``HallMHDSolver1D`` with η=0 (ideal limit).

    State vector U[i, :] = [ρ, ρvx, ρvy, ρvz, Bx, By, Bz, e].
    """

    def __init__(self, solver: HallMHDSolver1D) -> None:
        self.solver = solver

    def _compute_conservation(self) -> IdealMHDConservation:
        U = self.solver.U
        dx = self.solver.Lx / self.solver.nx
        rho = U[:, 0]
        mom_x = U[:, 1]
        Bx, By, Bz = U[:, 4], U[:, 5], U[:, 6]
        e = U[:, 7]

        total_mass = float(np.sum(rho) * dx)
        total_mom = float(np.sum(mom_x) * dx)
        total_E = float(np.sum(e) * dx)
        mag_E = float(0.5 * np.sum(Bx**2 + By**2 + Bz**2) * dx)

        # ∇·B in 1D: dBx/dx
        divB = np.gradient(Bx, dx)
        max_divB = float(np.max(np.abs(divB)))

        return IdealMHDConservation(
            total_mass=total_mass,
            total_momentum=total_mom,
            total_energy=total_E,
            max_divB=max_divB,
            magnetic_energy=mag_E,
        )

    def step(self, dt: float, session: TraceSession | None = None) -> None:
        """Advance one ideal MHD step."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(self.solver.U)

        self.solver.step(dt)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation()
            session.log_custom(
                name="ideal_mhd_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(self.solver.U)],
                params={"dt": dt},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

    def solve(
        self,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[float, int, TraceSession]:
        """
        Run ideal MHD.

        Returns:
            (t, n_steps, session)
        """
        if dt is None:
            dx = self.solver.Lx / self.solver.nx
            dt = 0.3 * dx  # CFL-like

        session = TraceSession()

        cons0 = self._compute_conservation()
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(self.solver.U)],
            params={"nx": self.solver.nx, "Lx": self.solver.Lx},
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
                                   max(abs(cons0.total_mass), 1e-30),
                     "energy_drift": abs(cons_f.total_energy - cons0.total_energy) /
                                     max(abs(cons0.total_energy), 1e-30)},
        )

        return t, n_steps, session
