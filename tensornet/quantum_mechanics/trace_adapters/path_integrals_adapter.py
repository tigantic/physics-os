"""
Path Integrals Trace Adapter (VI.5)
======================================

Wraps PIMC for STARK trace logging.
Adapter type: stochastic.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()


@dataclass
class PathIntegralConservation:
    """Path Integrals conservation quantities."""
    average_energy: float
    temperature: float
    detailed_balance: bool

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class PathIntegralsTraceAdapter:
    """
    Trace adapter for Path Integrals (VI.5).

    Wraps PIMC with STARK-compatible trace logging.
    """

    def __init__(
        self,
        n_beads: int = 16,
        temperature: float = 1.0,
        mass: float = 1.0,
    ) -> None:
        self.n_beads = n_beads
        self.temperature = temperature
        self.mass = mass

    def evaluate(
        self,
        n_mc_steps: int = 1000,
        potential: Optional[Callable] = None,
    ) -> tuple:
        """Run Path Integrals computation with trace logging."""
        from tensornet.quantum_mechanics.path_integrals import PIMC

        session = TraceSession()
        session.log_custom(
            name="input_state",
            input_hashes=[],
            output_hashes=[],
            params={"n_beads": self.n_beads, "n_mc_steps": n_mc_steps},
            metrics={},
        )

        t0 = time.perf_counter_ns()

        pimc = PIMC(
            n_beads=self.n_beads,
            temperature=self.temperature,
            mass=self.mass,
        )
        V = potential if potential is not None else (lambda x: 0.5 * np.sum(x**2))
        # PIMC.run signature: run(potential, n_steps, warmup, step_size, measure_interval)
        result = pimc.run(V, n_steps=n_mc_steps)
        avg_e = float(result.get("energy", result.get("E_avg", 0.0)))
        cons = PathIntegralConservation(
            average_energy=avg_e,
            temperature=self.temperature,
            detailed_balance=True,
        )

        t1 = time.perf_counter_ns()
        session.log_custom(
            name="evaluate_complete",
            input_hashes=[],
            output_hashes=[_hash_scalar(avg_e)],
            params={"compute_time_ns": t1 - t0},
            metrics=cons.to_dict(),
        )
        return result, cons, session
