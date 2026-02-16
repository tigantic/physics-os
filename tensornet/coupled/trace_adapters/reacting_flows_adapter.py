"""
Reacting Flows Trace Adapter (XVIII.5)
========================================

Wraps tensornet.cfd.reactive_ns.ReactiveNS for STARK tracing.
Conservation: total mass, species sum (Σ Yᵢ = 1), total energy.

NOTE: ReactiveNS is Torch-based. This adapter handles tensor↔numpy conversion
and re-exports the trace for Phase 7 coupled-physics integration.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class ReactingFlowsConservation:
    total_mass: float
    species_sum_error: float
    total_energy: float
    max_temperature: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "species_sum_error": self.species_sum_error,
            "total_energy": self.total_energy,
            "max_temperature": self.max_temperature,
        }


class ReactingFlowsTraceAdapter:
    """
    Reactive Navier-Stokes adapter (coupled-physics variant) with trace logging.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    Lx, Ly : float
        Domain size (m).
    cfl : float
        CFL number.
    """

    def __init__(
        self,
        Nx: int = 50,
        Ny: int = 50,
        Lx: float = 1.0,
        Ly: float = 1.0,
        cfl: float = 0.3,
    ) -> None:
        import torch

        from tensornet.cfd.reactive_ns import ReactiveConfig, ReactiveNS

        self.config = ReactiveConfig(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, cfl=cfl)
        self.solver = ReactiveNS(config=self.config)
        self._torch = torch

    def solve(
        self,
        state: object,
        n_steps: int = 100,
        dt: float | None = None,
    ) -> tuple[object, float, int, ReactingFlowsConservation, TraceSession]:
        """
        Time-step reactive Navier-Stokes.

        Parameters
        ----------
        state : ReactiveState (Torch tensors).
        n_steps : Number of time steps.
        dt : Time step; if None, computed from CFL.

        Returns
        -------
        state, t_final, n_steps, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        if dt is None:
            dt = float(self.solver.compute_timestep(state))

        cons = self._conservation(state)
        _record(session, 0, 0.0, state, cons)

        t = 0.0
        log_stride = max(1, n_steps // 20)
        for step in range(1, n_steps + 1):
            state = self.solver.step(state, dt)
            t += dt
            if step % log_stride == 0 or step == n_steps:
                cons = self._conservation(state)
                _record(session, step, t, state, cons)

        return state, t, n_steps, cons, session

    def _conservation(self, state: object) -> ReactingFlowsConservation:
        rho = state.rho
        Y = state.Y
        T = state.T

        total_mass = float(rho.sum().item())
        species_sum = sum(yi for yi in Y.values())
        species_err = float((species_sum - 1.0).abs().max().item())
        total_e = float((rho * (state.p / (rho + 1e-30))).sum().item())
        max_T = float(T.max().item())

        return ReactingFlowsConservation(
            total_mass=total_mass,
            species_sum_error=species_err,
            total_energy=total_e,
            max_temperature=max_T,
        )


def _record(
    session: TraceSession,
    step: int,
    t: float,
    state: object,
    cons: ReactingFlowsConservation,
) -> None:
    rho_np = state.rho.detach().cpu().numpy()
    session.log_custom(

        name="reactive_step",

        input_hashes=[_hash_array(rho_np)],

        output_hashes=[_hash_array(rho_np)],

        metrics={"step": step, "time": t, **cons.to_dict()},

    )
