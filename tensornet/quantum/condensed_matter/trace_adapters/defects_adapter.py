"""
Defects Trace Adapter (IX.7)
===============================

Wraps PointDefectCalculator for STARK trace logging.
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
class DefectsConservation:
    """Defects conservation quantities."""
    formation_energy: float
    relaxation_energy: float
    charge_balanced: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DefectsTraceAdapter:
    """
    Trace adapter for Defects (IX.7).

    Wraps PointDefectCalculator with STARK-compatible trace logging.
    """

    def __init__(self, n_atoms: int = 8) -> None:
        self.n_atoms = n_atoms

    def evaluate(self) -> tuple:
        """Run Defects computation with trace logging."""
        from tensornet.quantum.condensed_matter.defects import PointDefectCalculator

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_atoms": self.n_atoms},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        # Simple cubic lattice positions
        n_side = max(2, int(round(self.n_atoms ** (1.0 / 3.0))))
        positions = np.array(
            [[i, j, k] for i in range(n_side) for j in range(n_side) for k in range(n_side)],
            dtype=float,
        )[:self.n_atoms]
        box = np.array([float(n_side)] * 3)
        # PointDefectCalculator(positions, box, pair_potential, params)
        solver = PointDefectCalculator(positions=positions, box=box)
        result = solver.vacancy_formation_energy(site=0)
        cons = DefectsConservation(
            formation_energy=float(result.formation_energy),
            relaxation_energy=float(result.relaxation_energy),
            charge_balanced=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(float(result.formation_energy))],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
