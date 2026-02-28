"""
Navier-Stokes 2D Trace Adapter
================================

Wraps ``NS2DSolver.step_rk4()`` / ``NS2DSolver.solve()`` to emit deterministic
computation-trace entries for every time-step.

Conservation quantities logged per step:
    - Kinetic energy:          0.5 ∫(u² + v²) dA
    - Enstrophy:               0.5 ∫ω² dA
    - Max divergence:          max|∇·u| (should be ~0 for incompressible)

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ontic.cfd.ns_2d import NS2DSolver, NSState
from ontic.core.trace import TraceSession, _hash_tensor


@dataclass
class NS2DConservation:
    """Conservation quantities for incompressible NS."""
    kinetic_energy: float
    enstrophy: float
    max_divergence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kinetic_energy": self.kinetic_energy,
            "enstrophy": self.enstrophy,
            "max_divergence": self.max_divergence,
        }


def _compute_ns_conservation(
    solver: NS2DSolver, state: NSState
) -> NS2DConservation:
    """Compute NS conservation quantities."""
    with torch.no_grad():
        dx = solver.Lx / solver.Nx
        dy = solver.Ly / solver.Ny
        dA = dx * dy

        # Kinetic energy: 0.5 * ∫(u² + v²) dA
        ke = 0.5 * float((state.u ** 2 + state.v ** 2).sum().item()) * dA

        # Vorticity via spectral differentiation
        kx = torch.fft.fftfreq(solver.Nx, d=dx, device=state.u.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(solver.Ny, d=dy, device=state.u.device) * 2 * torch.pi
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")

        u_hat = torch.fft.fft2(state.u)
        v_hat = torch.fft.fft2(state.v)

        omega = torch.fft.ifft2(1j * KX * v_hat - 1j * KY * u_hat).real

        # Enstrophy: 0.5 * ∫ω² dA
        enstrophy = 0.5 * float((omega ** 2).sum().item()) * dA

        # Divergence: ∇·u (should be ~0)
        div = torch.fft.ifft2(1j * KX * u_hat + 1j * KY * v_hat).real
        max_div = float(torch.abs(div).max().item())

        return NS2DConservation(
            kinetic_energy=ke,
            enstrophy=enstrophy,
            max_divergence=max_div,
        )


def _ns_state_hash(state: NSState) -> str:
    """Deterministic SHA-256 of NS state."""
    h = hashlib.sha256()
    h.update(_hash_tensor(state.u).encode("ascii"))
    h.update(_hash_tensor(state.v).encode("ascii"))
    return h.hexdigest()


class NS2DTraceAdapter:
    """
    Wraps an ``NS2DSolver`` to emit computation trace entries.

    Usage::

        solver = NS2DSolver(Nx=128, Ny=128, Lx=2*pi, Ly=2*pi, nu=0.01)
        adapter = NS2DTraceAdapter(solver)
        result, session = adapter.solve(initial, t_final=1.0)
        session.save("traces/ns2d.json")
    """

    def __init__(self, solver: NS2DSolver) -> None:
        self.solver = solver

    def step(
        self,
        state: NSState,
        dt: float,
        session: TraceSession,
        method: str = "rk4",
    ) -> NSState:
        """Single timestep with trace recording."""
        cons_before = _compute_ns_conservation(self.solver, state)
        hash_before = _ns_state_hash(state)

        t0 = time.perf_counter_ns()
        if method == "rk4":
            state_next, proj_result = self.solver.step_rk4(state, dt)
        else:
            state_next, proj_result = self.solver.step_forward_euler(state, dt)
        duration_ns = time.perf_counter_ns() - t0

        cons_after = _compute_ns_conservation(self.solver, state_next)
        hash_after = _ns_state_hash(state_next)

        session.log_custom(
            name="ns2d_timestep",
            input_hashes={"state": hash_before},
            output_hashes={"state": hash_after},
            params={
                "dt": dt,
                "nu": self.solver.nu,
                "method": method,
                "grid": [self.solver.Nx, self.solver.Ny],
            },
            metrics={
                "conservation_before": cons_before.to_dict(),
                "conservation_after": cons_after.to_dict(),
                "ke_drift": abs(cons_after.kinetic_energy - cons_before.kinetic_energy),
                "max_divergence": cons_after.max_divergence,
                "wall_time_ns": duration_ns,
            },
        )

        return state_next

    def solve(
        self,
        initial_state: NSState,
        t_final: float,
        dt: float | None = None,
        cfl_target: float = 0.5,
        max_steps: int = 100_000,
        method: str = "rk4",
    ) -> tuple[NSState, float, int, TraceSession]:
        """
        Solve to final time with full trace recording.

        Returns:
            (final_state, final_time, num_steps, trace_session)
        """
        session = TraceSession()
        state = initial_state
        t = state.t
        step_count = 0

        cons_init = _compute_ns_conservation(self.solver, state)
        session.log_custom(
            name="ns2d_initial",
            input_hashes={},
            output_hashes={"state": _ns_state_hash(state)},
            params={
                "solver": "ns2d",
                "grid": [self.solver.Nx, self.solver.Ny],
                "domain": [self.solver.Lx, self.solver.Ly],
                "nu": self.solver.nu,
                "t_final": t_final,
                "method": method,
            },
            metrics={"conservation_initial": cons_init.to_dict()},
        )

        while t < t_final and step_count < max_steps:
            if dt is not None:
                step_dt = dt
            else:
                step_dt = self.solver.compute_stable_dt(state, cfl_target)
            step_dt = min(step_dt, t_final - t)

            state = self.step(state, step_dt, session, method=method)

            t += step_dt
            step_count += 1

        cons_final = _compute_ns_conservation(self.solver, state)
        session.log_custom(
            name="ns2d_final",
            input_hashes={"state": _ns_state_hash(state)},
            output_hashes={},
            params={"t_final_actual": t, "num_steps": step_count},
            metrics={
                "conservation_final": cons_final.to_dict(),
                "conservation_drift": {
                    "kinetic_energy": abs(
                        cons_final.kinetic_energy - cons_init.kinetic_energy
                    ),
                    "max_divergence": cons_final.max_divergence,
                },
            },
        )

        return state, t, step_count, session
