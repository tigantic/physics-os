"""
Euler 3D Trace Adapter
======================

Wraps ``Euler3D.step()`` / ``Euler3D.solve()`` to emit deterministic
computation-trace entries for every time-step.

Conservation quantities logged per step:
    - Total mass:              ∫ρ dV
    - x-momentum:              ∫ρu dV
    - y-momentum:              ∫ρv dV
    - z-momentum:              ∫ρw dV
    - Total energy:            ∫E dV

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ontic.cfd.euler_3d import Euler3D, Euler3DState
from ontic.core.trace import TraceSession, _hash_tensor


@dataclass
class Euler3DConservation:
    """Conservation quantities for one timestep."""
    mass: float
    momentum_x: float
    momentum_y: float
    momentum_z: float
    energy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mass": self.mass,
            "momentum_x": self.momentum_x,
            "momentum_y": self.momentum_y,
            "momentum_z": self.momentum_z,
            "energy": self.energy,
        }


def _compute_conservation(state: Euler3DState, dV: float) -> Euler3DConservation:
    """Compute Euler conservation integrals from primitive variables."""
    with torch.no_grad():
        U = state.to_conservative()
        return Euler3DConservation(
            mass=float(U[0].sum().item()) * dV,
            momentum_x=float(U[1].sum().item()) * dV,
            momentum_y=float(U[2].sum().item()) * dV,
            momentum_z=float(U[3].sum().item()) * dV,
            energy=float(U[4].sum().item()) * dV,
        )


def _state_hash(state: Euler3DState) -> str:
    """Deterministic SHA-256 of the full Euler3D state."""
    h = hashlib.sha256()
    for fld in (state.rho, state.u, state.v, state.w, state.p):
        h.update(_hash_tensor(fld).encode("ascii"))
    return h.hexdigest()


class Euler3DTraceAdapter:
    """
    Wraps an ``Euler3D`` solver to emit computation trace entries.

    Usage::

        solver = Euler3D(Nx=64, Ny=64, Nz=64, Lx=1.0, Ly=1.0, Lz=1.0)
        adapter = Euler3DTraceAdapter(solver)
        state, t, steps, session = adapter.solve(initial, t_final=0.5)
        session.save("traces/euler3d.json")
    """

    def __init__(self, solver: Euler3D) -> None:
        self.solver = solver
        self.dV = solver.dx * solver.dy * solver.dz

    def step(
        self,
        state: Euler3DState,
        dt: float,
        session: TraceSession,
    ) -> Euler3DState:
        """Single timestep with trace recording."""
        cons_before = _compute_conservation(state, self.dV)
        hash_before = _state_hash(state)

        t0 = time.perf_counter_ns()
        state_next = self.solver.step(state, dt)
        duration_ns = time.perf_counter_ns() - t0

        cons_after = _compute_conservation(state_next, self.dV)
        hash_after = _state_hash(state_next)

        session.log_custom(
            name="euler3d_timestep",
            input_hashes={"state": hash_before},
            output_hashes={"state": hash_after},
            params={
                "dt": dt,
                "gamma": self.solver.gamma,
                "grid": [self.solver.Nx, self.solver.Ny, self.solver.Nz],
                "cfl": self.solver.cfl,
            },
            metrics={
                "conservation_before": cons_before.to_dict(),
                "conservation_after": cons_after.to_dict(),
                "mass_drift": abs(cons_after.mass - cons_before.mass),
                "momentum_x_drift": abs(cons_after.momentum_x - cons_before.momentum_x),
                "momentum_y_drift": abs(cons_after.momentum_y - cons_before.momentum_y),
                "momentum_z_drift": abs(cons_after.momentum_z - cons_before.momentum_z),
                "energy_drift": abs(cons_after.energy - cons_before.energy),
                "wall_time_ns": duration_ns,
            },
        )

        return state_next

    def solve(
        self,
        initial_state: Euler3DState,
        t_final: float,
        max_steps: int = 100_000,
        callback: Callable[..., Any] | None = None,
    ) -> tuple[Euler3DState, float, int, TraceSession]:
        """
        Solve to final time with full trace recording.

        Returns:
            (final_state, final_time, num_steps, trace_session)
        """
        session = TraceSession()
        state = initial_state
        t = 0.0
        step_count = 0

        # Record initial state
        cons_init = _compute_conservation(state, self.dV)
        session.log_custom(
            name="euler3d_initial",
            input_hashes={},
            output_hashes={"state": _state_hash(state)},
            params={
                "solver": "euler3d",
                "grid": [self.solver.Nx, self.solver.Ny, self.solver.Nz],
                "domain": [self.solver.Lx, self.solver.Ly, self.solver.Lz],
                "gamma": self.solver.gamma,
                "cfl": self.solver.cfl,
                "t_final": t_final,
            },
            metrics={"conservation_initial": cons_init.to_dict()},
        )

        while t < t_final and step_count < max_steps:
            dt = self.solver.compute_timestep(state)
            dt = min(dt, t_final - t)

            state = self.step(state, dt, session)

            t += dt
            step_count += 1

            if callback is not None and callback(state, t, step_count):
                break

        # Record final conservation check
        cons_final = _compute_conservation(state, self.dV)
        session.log_custom(
            name="euler3d_final",
            input_hashes={"state": _state_hash(state)},
            output_hashes={},
            params={"t_final_actual": t, "num_steps": step_count},
            metrics={
                "conservation_final": cons_final.to_dict(),
                "conservation_drift": {
                    "mass": abs(cons_final.mass - cons_init.mass),
                    "momentum_x": abs(cons_final.momentum_x - cons_init.momentum_x),
                    "momentum_y": abs(cons_final.momentum_y - cons_init.momentum_y),
                    "momentum_z": abs(cons_final.momentum_z - cons_init.momentum_z),
                    "energy": abs(cons_final.energy - cons_init.energy),
                },
            },
        )

        return state, t, step_count, session
