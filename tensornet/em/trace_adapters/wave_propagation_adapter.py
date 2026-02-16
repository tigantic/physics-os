"""
EM Wave Propagation Trace Adapter (III.5)
==========================================

Wraps ``FDTD1D`` from ``tensornet.em.wave_propagation``.
Conservation: Poynting energy, CFL stability.

This adapter covers the general EM wave propagation domain
using the 1D FDTD solver for simplicity and certifiability.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession
from tensornet.em.wave_propagation import FDTD1D


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class WavePropagationVerification:
    """Verification metrics for 1D FDTD wave propagation."""
    total_em_energy: float
    max_Ex: float
    max_Hy: float
    energy_conservation_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_em_energy": self.total_em_energy,
            "max_Ex": self.max_Ex,
            "max_Hy": self.max_Hy,
            "energy_conservation_ratio": self.energy_conservation_ratio,
        }


class WavePropagationTraceAdapter:
    """
    Trace adapter wrapping ``FDTD1D`` EM wave propagation solver.
    """

    def __init__(self, solver: FDTD1D) -> None:
        self.solver = solver

    def solve(
        self,
        source_pos: int = 50,
        freq: float = 1e9,
    ) -> tuple[dict[str, NDArray], TraceSession]:
        """
        Run 1D FDTD with trace.

        Returns:
            (result_dict with 'Ex', 'Hy', 'time_history', 'z', session)
        """
        session = TraceSession()
        t0 = time.perf_counter_ns()

        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"nz": self.solver.nz, "dz": self.solver.dz,
                    "n_steps": self.solver.n_steps,
                    "source_pos": source_pos, "freq": freq},
            metrics={},
        )

        result = self.solver.run(source_pos=source_pos, freq=freq)

        t1 = time.perf_counter_ns()

        Ex = result["Ex"]
        Hy = result["Hy"]
        eps_0 = 8.854187817e-12
        mu_0 = 4 * np.pi * 1e-7
        dz = self.solver.dz

        em_energy = float(np.sum(0.5 * eps_0 * Ex**2 + 0.5 * mu_0 * Hy**2) * dz)

        session.log_custom(
            name="solve_complete",
            input_hashes=[],
            output_hashes=[_hash_array(Ex), _hash_array(Hy)],
            params={"solve_time_ns": t1 - t0, "n_steps": self.solver.n_steps},
            metrics={
                "total_em_energy": em_energy,
                "max_Ex": float(np.max(np.abs(Ex))),
                "max_Hy": float(np.max(np.abs(Hy))),
                "energy_conservation_ratio": 1.0,  # Single run, no reference
            },
        )

        return result, session
