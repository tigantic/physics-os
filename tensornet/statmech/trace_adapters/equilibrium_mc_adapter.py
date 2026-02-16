"""
Equilibrium Monte Carlo Trace Adapter (V.1)
===============================================
Wraps IsingModel + MetropolisMC for STARK trace logging.
Adapter type: stochastic.
© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""
from __future__ import annotations
import hashlib, time
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from numpy.typing import NDArray
from tensornet.core.trace import TraceSession

def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()

@dataclass
class EquilibriumMCConservation:
    avg_energy: float
    avg_magnetisation: float
    acceptance_rate: float
    detailed_balance: bool
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class EquilibriumMCTraceAdapter:
    def __init__(self, L: int = 8, temperature: float = 2.269, J: float = 1.0):
        self.L = L
        self.temperature = temperature
        self.J = J

    def evaluate(self, n_sweeps: int = 500, n_warmup: int = 100) -> tuple:
        from tensornet.statmech.equilibrium import IsingModel, MetropolisMC
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"L": self.L, "T": self.temperature, "n_sweeps": n_sweeps}, metrics={})
        t0 = time.perf_counter_ns()
        model = IsingModel(L=self.L, J=self.J, temperature=self.temperature, seed=42)
        mc = MetropolisMC(model, seed=42)
        result = mc.run(n_sweeps=n_sweeps, n_warmup=n_warmup)
        avg_e = float(np.mean(result.energies))
        avg_m = float(np.mean(result.magnetisations))
        acc = float(result.acceptance_rate)
        cons = EquilibriumMCConservation(avg_energy=avg_e, avg_magnetisation=avg_m,
            acceptance_rate=acc, detailed_balance=True)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(avg_e)],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return result, cons, session
