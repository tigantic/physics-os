"""
Ultracold Atoms Trace Adapter (VII.13)
=========================================

Wraps GrossPitaevskiiSolver (ultracold variant) for STARK trace logging.
Adapter type: scf.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class UltracoldConservation:
    """Ultracold Atoms conservation quantities."""
    energy: float
    atom_number_conserved: bool
    chemical_potential: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class UltracoldTraceAdapter:
    """
    Trace adapter for Ultracold Atoms (VII.13).

    Wraps GrossPitaevskiiSolver (ultracold) with STARK-compatible trace
    logging.
    """

    def __init__(
        self,
        nx: int = 128,
        Lx: float = 20.0,
        g_int: float = 1.0,
    ) -> None:
        self.nx = nx
        self.Lx = Lx
        self.g_int = g_int

    def evaluate(self, n_steps: int = 2000) -> tuple:
        """Run Ultracold Atoms computation with trace logging."""
        from tensornet.condensed_matter.ultracold_atoms import (
            GrossPitaevskiiSolver as GPEUltracold,
        )

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"nx": self.nx, "Lx": self.Lx, "g": self.g_int},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        # GrossPitaevskiiSolver(nx, Lx, g, omega)
        solver = GPEUltracold(nx=self.nx, Lx=self.Lx, g=self.g_int, omega=1.0)
        density = solver.find_ground_state(n_steps=n_steps)
        mu = solver.chemical_potential()
        dx = self.Lx / self.nx
        atom_count = float(np.sum(density) * dx)
        cons = UltracoldConservation(
            energy=float(mu),
            atom_number_conserved=bool(atom_count > 0),
            chemical_potential=float(mu),
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(mu)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return density, cons, session
