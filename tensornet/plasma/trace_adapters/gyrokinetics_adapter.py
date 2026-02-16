"""
Gyrokinetics Trace Adapter (XI.4)
===================================

Wraps ``GyrokineticVlasov1D`` from ``tensornet.plasma.gyrokinetics``.
Conservation: phase-space volume (∫f dz dv), total energy.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession
from tensornet.plasma.gyrokinetics import GyrokineticVlasov1D


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class GyrokineticsConservation:
    """Conservation quantities for gyrokinetic solver."""
    particle_count: float
    kinetic_energy: float
    l2_norm: float
    entropy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "particle_count": self.particle_count,
            "kinetic_energy": self.kinetic_energy,
            "l2_norm": self.l2_norm,
            "entropy": self.entropy,
        }


class GyrokineticsTraceAdapter:
    """
    Trace adapter wrapping ``GyrokineticVlasov1D``.
    """

    def __init__(self, solver: GyrokineticVlasov1D) -> None:
        self.solver = solver

    def _compute_conservation(self) -> GyrokineticsConservation:
        f = self.solver.f
        dz = self.solver.Lz / self.solver.nz
        dv = 2 * self.solver.v_max / self.solver.nv
        v = np.linspace(-self.solver.v_max, self.solver.v_max,
                        self.solver.nv, endpoint=False) + dv / 2

        dV = dz * dv
        particle_count = float(np.sum(f) * dV)
        KE = float(0.5 * np.sum(f * v[None, :]**2) * dV)
        l2 = float(np.sum(f**2) * dV)
        f_safe = np.maximum(f, 1e-300)
        entropy = float(-np.sum(f_safe * np.log(f_safe)) * dV)

        return GyrokineticsConservation(
            particle_count=particle_count,
            kinetic_energy=KE,
            l2_norm=l2,
            entropy=entropy,
        )

    def step(self, dt: float, session: TraceSession | None = None) -> float:
        """Advance one GK step. Returns kinetic energy."""
        t0 = time.perf_counter_ns()
        input_hash = _hash_array(self.solver.f)

        KE = self.solver.step(dt)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation()
            session.log_custom(
                name="gk_step",
                input_hashes=[input_hash],
                output_hashes=[_hash_array(self.solver.f)],
                params={"dt": dt},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return KE

    def solve(
        self,
        t_final: float,
        dt: float | None = None,
    ) -> tuple[float, int, TraceSession]:
        if dt is None:
            dz = self.solver.Lz / self.solver.nz
            dt = 0.5 * dz / self.solver.v_max

        session = TraceSession()

        cons0 = self._compute_conservation()
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_array(self.solver.f)],
            params={"nz": self.solver.nz, "nv": self.solver.nv,
                    "Lz": self.solver.Lz, "v_max": self.solver.v_max},
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
            input_hashes=[_hash_array(self.solver.f)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "particle_drift": abs(cons_f.particle_count - cons0.particle_count) /
                                       max(abs(cons0.particle_count), 1e-30)},
        )

        return t, n_steps, session
