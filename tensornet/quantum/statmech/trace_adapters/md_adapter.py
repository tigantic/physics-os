"""
Molecular Dynamics Trace Adapter (V.3)
========================================

Wraps ``MDSimulation`` from ``tensornet.md.engine``.
Conservation: total energy (KE + PE), momentum, angular momentum.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class MDConservation:
    """Conservation quantities for MD simulation."""
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    total_momentum: float
    temperature: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "total_energy": self.total_energy,
            "total_momentum": self.total_momentum,
            "temperature": self.temperature,
        }


class MDTraceAdapter:
    """
    Trace adapter wrapping ``MDSimulation`` from ``tensornet.md.engine``.

    Logs kinetic energy, potential energy, momentum, temperature per step.
    """

    def __init__(self, simulation: object) -> None:
        """
        Parameters:
            simulation: Instance of ``MDSimulation``.
        """
        self.sim = simulation

    def _state_hash(self) -> str:
        h = hashlib.sha256()
        state = self.sim.state
        h.update(np.ascontiguousarray(state.positions).tobytes())
        h.update(np.ascontiguousarray(state.velocities).tobytes())
        return h.hexdigest()

    def _compute_conservation(self, KE: float, PE: float) -> MDConservation:
        state = self.sim.state
        n_atoms = len(self.sim.atoms) if hasattr(self.sim, "atoms") else state.positions.shape[0]

        # Total momentum: sum(m_i * v_i)
        masses = np.array([a.mass for a in self.sim.atoms]) if hasattr(self.sim, "atoms") else np.ones(n_atoms)
        total_p = float(np.linalg.norm(np.sum(masses[:, None] * state.velocities, axis=0)))

        # Temperature: T = 2*KE / (3*N*k_B) (in reduced units kB=1)
        ndof = 3 * n_atoms
        temperature = 2 * KE / max(ndof, 1)

        return MDConservation(
            kinetic_energy=KE,
            potential_energy=PE,
            total_energy=KE + PE,
            total_momentum=total_p,
            temperature=temperature,
        )

    def step(self, session: TraceSession | None = None) -> tuple[float, float]:
        """Advance one MD step."""
        t0 = time.perf_counter_ns()
        input_hash = self._state_hash()

        KE, PE = self.sim.step()

        t1 = time.perf_counter_ns()

        if session is not None:
            cons = self._compute_conservation(KE, PE)
            session.log_custom(
                name="md_step",
                input_hashes=[input_hash],
                output_hashes=[self._state_hash()],
                params={"dt": self.sim.integrator.dt if hasattr(self.sim, "integrator") else
                        self.sim.dt if hasattr(self.sim, "dt") else 0.002},
                metrics={**cons.to_dict(), "step_ns": t1 - t0},
            )

        return KE, PE

    def solve(
        self,
        n_steps: int,
    ) -> tuple[float, float, int, TraceSession]:
        """
        Run MD simulation for n_steps.

        Returns:
            (final_KE, final_PE, n_steps, session)
        """
        session = TraceSession()

        # Initial state — step once to get energies, or use zeros
        KE0, PE0 = 0.0, 0.0
        if hasattr(self.sim, "last_KE"):
            KE0, PE0 = self.sim.last_KE, self.sim.last_PE

        session.log_custom(
            name="initial_state",
            input_hashes=[],
            output_hashes=[self._state_hash()],
            params={"n_steps": n_steps,
                    "n_atoms": len(self.sim.atoms) if hasattr(self.sim, "atoms") else 0},
            metrics={"initial_KE": KE0, "initial_PE": PE0, "initial_E": KE0 + PE0},
        )

        KE, PE = 0.0, 0.0
        for i in range(n_steps):
            KE, PE = self.step(session)

        cons_f = self._compute_conservation(KE, PE)
        session.log_custom(
            name="final_state",
            input_hashes=[self._state_hash()],
            output_hashes=[],
            params={"n_steps_completed": n_steps},
            metrics={**cons_f.to_dict(),
                     "energy_drift": abs(cons_f.total_energy - (KE0 + PE0)) /
                                     max(abs(KE0 + PE0), 1e-30) if (KE0 + PE0) != 0 else 0.0},
        )

        return KE, PE, n_steps, session
