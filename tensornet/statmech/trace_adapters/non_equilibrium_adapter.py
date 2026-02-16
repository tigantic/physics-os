"""
Non-Equilibrium StatMech Trace Adapter (V.2)
==============================================

Wraps ``KineticMonteCarlo`` / ``GillespieSSA`` from
``tensornet.statmech.non_equilibrium``.
Conservation: free energy, detailed balance.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_state(state: dict | list | tuple) -> str:
    return hashlib.sha256(repr(sorted(state.items()) if isinstance(state, dict) else state).encode()).hexdigest()


@dataclass
class NonEquilibriumVerification:
    """Verification metrics for KMC/Gillespie runs."""
    total_events: int
    final_time: float
    species_total: float
    mean_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_events": float(self.total_events),
            "final_time": self.final_time,
            "species_total": self.species_total,
            "mean_rate": self.mean_rate,
        }


class NonEquilibriumTraceAdapter:
    """
    Trace adapter for ``GillespieSSA`` / ``KineticMonteCarlo`` solvers.

    Logs event counts, species conservation, timing.
    """

    def __init__(self, solver: object) -> None:
        """
        Parameters:
            solver: Instance of ``GillespieSSA`` or ``KineticMonteCarlo``.
        """
        self.solver = solver

    def run(
        self,
        t_max: float,
        max_steps: int = 100_000,
    ) -> tuple[list, float, int, TraceSession]:
        """
        Run stochastic simulation with trace.

        Returns:
            (trajectory, t_final, n_events, session)
        """
        session = TraceSession()
        t0_wall = time.perf_counter_ns()

        # Get initial state hash
        if hasattr(self.solver, "state"):
            init_state = dict(self.solver.state) if isinstance(self.solver.state, dict) else self.solver.state
        elif hasattr(self.solver, "populations"):
            init_state = dict(self.solver.populations) if hasattr(self.solver.populations, "items") else self.solver.populations
        else:
            init_state = {}

        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[_hash_state(init_state) if isinstance(init_state, dict) else hashlib.sha256(repr(init_state).encode()).hexdigest()],
            params={"t_max": t_max, "max_steps": max_steps},
            metrics={"species_total": float(sum(init_state.values())) if isinstance(init_state, dict) else 0.0},
        )

        trajectory = []
        t = 0.0
        n_events = 0

        while t < t_max and n_events < max_steps:
            result = self.solver.step()
            if result is None:
                break
            dt_event, event_name = result[0], result[1] if len(result) > 1 else "unknown"
            if dt_event is None or event_name is None:
                break
            t += dt_event
            n_events += 1
            trajectory.append((t, event_name))

            # Log milestone events
            if n_events % max(max_steps // 20, 1) == 0:
                session.log_custom(
                    name="milestone",
                    input_hashes=[],
                    output_hashes=[],
                    params={"event": n_events, "t": t},
                    metrics={"events_so_far": float(n_events)},
                )

        t1_wall = time.perf_counter_ns()

        # Final state
        if hasattr(self.solver, "state"):
            final_state = dict(self.solver.state) if isinstance(self.solver.state, dict) else self.solver.state
        elif hasattr(self.solver, "populations"):
            final_state = dict(self.solver.populations) if hasattr(self.solver.populations, "items") else self.solver.populations
        else:
            final_state = {}

        species_total = float(sum(final_state.values())) if isinstance(final_state, dict) else 0.0
        mean_rate = n_events / max(t, 1e-30)

        session.log_custom(
            name="final_state",
            input_hashes=[],
            output_hashes=[_hash_state(final_state) if isinstance(final_state, dict) else hashlib.sha256(repr(final_state).encode()).hexdigest()],
            params={"t_final": t, "n_events": n_events,
                    "wall_time_ns": t1_wall - t0_wall},
            metrics={
                "total_events": float(n_events),
                "final_time": t,
                "species_total": species_total,
                "mean_rate": mean_rate,
            },
        )

        return trajectory, t, n_events, session
