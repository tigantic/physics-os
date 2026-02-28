"""
Reactive Flow Trace Adapter (II.5)
====================================

Wraps ``ReactiveNS.step()`` to emit deterministic trace entries.
Conservation: species mass fractions (ΣY_i = 1), total energy.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ontic.core.trace import TraceSession, _hash_tensor


@dataclass
class ReactiveConservation:
    """Conservation quantities per step."""
    total_mass: float
    total_energy: float
    species_sum: float        # Should be 1.0 everywhere
    max_temperature: float
    min_temperature: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_mass": self.total_mass,
            "total_energy": self.total_energy,
            "species_sum": self.species_sum,
            "max_temperature": self.max_temperature,
            "min_temperature": self.min_temperature,
        }


class ReactiveFlowTraceAdapter:
    """
    Trace adapter wrapping ``ReactiveNS`` from ``ontic.cfd.reactive_ns``.

    Logs species mass conservation and total energy per step.
    """

    def __init__(self, solver: object) -> None:
        """
        Parameters:
            solver: Instance of ``ReactiveNS`` (from ``ontic.cfd.reactive_ns``).
        """
        self.solver = solver

    def _state_hash(self, state: object) -> str:
        """Hash the reactive state fields."""
        h = hashlib.sha256()
        for field_name in ("rho", "u", "v", "p"):
            val = getattr(state, field_name, None)
            if val is not None:
                h.update(_hash_tensor(val).encode())
        # Hash species mass fractions
        Y = getattr(state, "Y", {})
        for species_key in sorted(Y.keys(), key=str):
            h.update(_hash_tensor(Y[species_key]).encode())
        return h.hexdigest()

    def _compute_conservation(self, state: object) -> ReactiveConservation:
        rho = state.rho
        p = state.p
        u, v = state.u, state.v
        gamma = 1.4  # default air

        # Total mass
        total_mass = float(torch.sum(rho).item())

        # Total energy: E = p/(γ-1) + ½ρ(u²+v²)
        KE = 0.5 * rho * (u**2 + v**2)
        IE = p / (gamma - 1.0)
        total_energy = float(torch.sum(KE + IE).item())

        # Species sum check
        Y = getattr(state, "Y", {})
        if Y:
            Y_sum = sum(Y[k] for k in Y)
            species_sum = float(torch.mean(Y_sum).item())
        else:
            species_sum = 1.0

        # Temperature estimate: T = p / (ρ R_mix)
        T = state.T if hasattr(state, "T") else p / (rho * 287.0)
        max_T = float(torch.max(T).item())
        min_T = float(torch.min(T).item())

        return ReactiveConservation(
            total_mass=total_mass,
            total_energy=total_energy,
            species_sum=species_sum,
            max_temperature=max_T,
            min_temperature=min_T,
        )

    def step(self, state: object, dt: float,
             session: TraceSession | None = None) -> object:
        """Advance one reactive step via solver.step()."""
        t0 = time.perf_counter_ns()
        input_hash = self._state_hash(state)

        new_state = self.solver.step(state, dt)

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(new_state)
            session.log_custom(
                name="reactive_step",
                input_hashes=[input_hash],
                output_hashes=[self._state_hash(new_state)],
                params={"dt": dt},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return new_state

    def solve(
        self,
        state0: object,
        t_final: float,
        dt: float = 1e-5,
    ) -> tuple[object, float, int, TraceSession]:
        """
        Run reactive flow from initial condition.

        Returns:
            (final_state, t, n_steps, session)
        """
        session = TraceSession()

        cons0 = self._compute_conservation(state0)
        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[self._state_hash(state0)],
            params={"t_final": t_final, "dt": dt},
            metrics=cons0.to_dict(),
        )

        state = state0
        t = 0.0
        n_steps = 0

        while t < t_final - 1e-14:
            dt_actual = min(dt, t_final - t)
            state = self.step(state, dt_actual, session)
            t += dt_actual
            n_steps += 1

        cons_f = self._compute_conservation(state)
        session.log_custom(
            name="final_state",
            input_hashes=[self._state_hash(state)],
            output_hashes=[],
            params={"t_final": t, "n_steps": n_steps},
            metrics={**cons_f.to_dict(),
                     "mass_drift": abs(cons_f.total_mass - cons0.total_mass) /
                                   max(abs(cons0.total_mass), 1e-30),
                     "energy_drift": abs(cons_f.total_energy - cons0.total_energy) /
                                     max(abs(cons0.total_energy), 1e-30)},
        )

        return state, t, n_steps, session
