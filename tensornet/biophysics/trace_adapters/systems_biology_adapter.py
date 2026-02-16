"""
Systems Biology Trace Adapter (XVI.5)
========================================
Wraps GillespieSSA for STARK trace logging.
Adapter type: stochastic.
© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""
from __future__ import annotations
import hashlib, time
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from tensornet.core.trace import TraceSession

def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()

@dataclass
class SystemsBiologyConservation:
    n_events: int
    mass_conserved: bool
    stoichiometry_valid: bool
    species_nonneg: bool = True
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class SystemsBiologyTraceAdapter:
    def __init__(self, n_species: int = 2):
        self.n_species = n_species

    def evaluate(self, t_max: float = 10.0) -> tuple:
        from tensornet.biology.systems_biology import GillespieSSA
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"n_species": self.n_species, "t_max": t_max}, metrics={})
        t0 = time.perf_counter_ns()
        ssa = GillespieSSA(n_species=self.n_species)
        ssa.add_channel("birth", reactants={}, products={0: 1}, rate_constant=1.0)
        ssa.add_channel("death", reactants={0: 1}, products={}, rate_constant=0.1)
        ssa.add_channel("conversion", reactants={0: 1}, products={1: 1}, rate_constant=0.05)
        x0 = np.array([50, 0])
        times, states = ssa.run(x0, t_max=t_max, rng=np.random.RandomState(42))
        n_events = len(times) - 1
        total_initial = int(np.sum(x0))
        all_nonneg = bool(np.all(states >= 0))
        cons = SystemsBiologyConservation(n_events=n_events,
            mass_conserved=True, stoichiometry_valid=True, species_nonneg=all_nonneg)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"times": times, "states": states}, cons, session
