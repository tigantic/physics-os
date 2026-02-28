"""
Shallow Water Trace Adapter (II.7)
====================================

Wraps ``ShallowWaterEquations`` from ``ontic.geophysics.oceanography``
to emit deterministic trace entries.

Conservation: mass (water height), momentum, potential vorticity.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class ShallowWaterConservation:
    """Conservation quantities for shallow water equations."""
    total_mass: float
    momentum_x: float
    momentum_y: float
    total_energy: float
    potential_vorticity_mean: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "momentum_x": self.momentum_x,
            "momentum_y": self.momentum_y,
            "total_energy": self.total_energy,
            "potential_vorticity_mean": self.potential_vorticity_mean,
        }


class ShallowWaterTraceAdapter:
    """
    Trace adapter wrapping ``ShallowWaterEquations`` from oceanography.

    Logs mass, momentum, energy, and potential vorticity per step.
    """

    def __init__(self, solver: object) -> None:
        """
        Parameters:
            solver: Instance of ``ShallowWaterEquations`` from
                    ``ontic.geophysics.oceanography``.
        """
        self.solver = solver

    def _state_hash(self) -> str:
        h = hashlib.sha256()
        for field_name in ("u", "v", "eta"):
            arr = getattr(self.solver, field_name, None)
            if arr is not None:
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def _compute_conservation(self) -> ShallowWaterConservation:
        s = self.solver
        dx = s.Lx / s.nx
        dy = s.Ly / s.ny
        dA = dx * dy
        H = s.H

        h_total = H + s.eta  # total water depth
        total_mass = float(np.sum(h_total) * dA)
        momentum_x = float(np.sum(h_total * s.u) * dA)
        momentum_y = float(np.sum(h_total * s.v) * dA)

        # Total energy: KE + PE
        g = 9.81
        KE = 0.5 * h_total * (s.u**2 + s.v**2)
        PE = 0.5 * g * s.eta**2
        total_energy = float(np.sum(KE + PE) * dA)

        # Potential vorticity: (f + ζ) / h, with ζ = ∂v/∂x - ∂u/∂y
        dvdx = np.gradient(s.v, dx, axis=0)
        dudy = np.gradient(s.u, dy, axis=1)
        zeta = dvdx - dudy
        f = s.f0 if hasattr(s, "f0") else 1e-4
        PV = (f + zeta) / np.maximum(h_total, 1e-10)
        pv_mean = float(np.mean(np.abs(PV)))

        return ShallowWaterConservation(
            total_mass=total_mass,
            momentum_x=momentum_x,
            momentum_y=momentum_y,
            total_energy=total_energy,
            potential_vorticity_mean=pv_mean,
        )

    def step(self, dt: float, session: TraceSession | None = None) -> None:
        """Advance one shallow-water step."""
        t0 = time.perf_counter_ns()
        input_hash = self._state_hash()

        self.solver.step(dt)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation()
            session.log_custom(
                name="shallow_water_step",
                input_hashes=[input_hash],
                output_hashes=[self._state_hash()],
                params={"dt": dt},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

    def solve(
        self,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[float, int, TraceSession]:
        """
        Run shallow water from current state to t_final.

        Returns:
            (t, n_steps, session)
        """
        if dt is None:
            dt = self.solver.cfl_timestep(0.5) if hasattr(self.solver, "cfl_timestep") else 1.0

        session = TraceSession()

        cons0 = self._compute_conservation()
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[self._state_hash()],
            params={"nx": self.solver.nx, "ny": self.solver.ny,
                    "Lx": self.solver.Lx, "Ly": self.solver.Ly,
                    "H": self.solver.H},
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
            input_hashes=[self._state_hash()],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "mass_drift": abs(cons_f.total_mass - cons0.total_mass) /
                                   max(abs(cons0.total_mass), 1e-30),
                     "energy_drift": abs(cons_f.total_energy - cons0.total_energy) /
                                     max(abs(cons0.total_energy), 1e-30)},
        )

        return t, n_steps, session
