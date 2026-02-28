"""
Maxwell FDTD Trace Adapter (III.3)
====================================

Wraps ``FDTD2D_TM`` from ``ontic.em.wave_propagation``.
Conservation: Poynting energy (electromagnetic energy density), CFL stability.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ontic.core.trace import TraceSession
from ontic.em.wave_propagation import FDTD2D_TM


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class MaxwellFDTDConservation:
    """Conservation quantities for FDTD."""
    em_energy: float
    max_Ez: float
    max_Hx: float
    max_Hy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "em_energy": self.em_energy,
            "max_Ez": self.max_Ez,
            "max_Hx": self.max_Hx,
            "max_Hy": self.max_Hy,
        }


class MaxwellFDTDTraceAdapter:
    """
    Trace adapter wrapping ``FDTD2D_TM`` Maxwell solver.

    Logs electromagnetic energy, field maxima per step.
    """

    def __init__(self, solver: FDTD2D_TM) -> None:
        self.solver = solver

    def _compute_conservation(self) -> MaxwellFDTDConservation:
        eps_0 = 8.854187817e-12
        mu_0 = 4 * np.pi * 1e-7
        dx = self.solver.dx
        dy = self.solver.dy
        dA = dx * dy

        Ez = self.solver.Ez
        Hx = self.solver.Hx
        Hy = self.solver.Hy

        # EM energy density: u = ½ε₀|E|² + ½μ₀|H|²
        u_E = 0.5 * eps_0 * Ez**2
        u_H = 0.5 * mu_0 * (Hx**2 + Hy**2)
        em_energy = float(np.sum(u_E + u_H) * dA)

        return MaxwellFDTDConservation(
            em_energy=em_energy,
            max_Ez=float(np.max(np.abs(Ez))),
            max_Hx=float(np.max(np.abs(Hx))),
            max_Hy=float(np.max(np.abs(Hy))),
        )

    def _state_hash(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.solver.Ez).tobytes())
        h.update(np.ascontiguousarray(self.solver.Hx).tobytes())
        h.update(np.ascontiguousarray(self.solver.Hy).tobytes())
        return h.hexdigest()

    def step(self, session: TraceSession | None = None) -> None:
        """Advance one FDTD leapfrog step."""
        t0 = time.perf_counter_ns()
        input_hash = self._state_hash()

        self.solver.step()

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation()
            session.log_custom(
                name="fdtd_step",
                input_hashes=[input_hash],
                output_hashes=[self._state_hash()],
                params={"dx": self.solver.dx, "dy": self.solver.dy},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

    def solve(
        self,
        n_steps: int,
        source_pos: tuple[int, int] | None = None,
        freq: float = 10e9,
    ) -> tuple[NDArray, int, TraceSession]:
        """
        Run FDTD for n_steps with optional point source.

        Returns:
            (Ez_final, n_steps, session)
        """
        session = TraceSession()

        cons0 = self._compute_conservation()
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[self._state_hash()],
            params={"nx": self.solver.nx, "ny": self.solver.ny,
                    "dx": self.solver.dx, "dy": self.solver.dy,
                    "n_steps": n_steps, "freq": freq},
            metrics=cons0.to_dict(),
        )

        c = 3e8
        dt = self.solver.dx / (c * np.sqrt(2)) * 0.99  # CFL

        for i in range(n_steps):
            if source_pos is not None:
                self.solver.Ez[source_pos[0], source_pos[1]] += np.sin(2 * np.pi * freq * i * dt)
            self.step(session)

        cons_f = self._compute_conservation()
        session.log_custom(
            name="final_state",
            input_hashes=[self._state_hash()],
            output_hashes=[],
            params={"n_steps_completed": n_steps},
            metrics={**cons_f.to_dict(),
                     "energy_growth": cons_f.em_energy / max(cons0.em_energy, 1e-30)},
        )

        return self.solver.Ez.copy(), n_steps, session
