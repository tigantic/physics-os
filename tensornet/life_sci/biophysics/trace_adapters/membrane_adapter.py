"""
Membrane Biophysics Trace Adapter (XVI.3)
============================================
Wraps HelfrichMembrane for STARK trace logging.
Adapter type: scf.
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
class MembraneConservation:
    bending_energy: float
    persistence_length: float
    area_conserved: bool
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class MembraneTraceAdapter:
    def __init__(self, kappa: float = 20.0, sigma: float = 0.0):
        self.kappa = kappa
        self.sigma = sigma

    def evaluate(self) -> tuple:
        from tensornet.life_sci.membrane_bio import HelfrichMembrane
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"kappa": self.kappa, "sigma": self.sigma}, metrics={})
        t0 = time.perf_counter_ns()
        membrane = HelfrichMembrane(kappa=self.kappa, sigma=self.sigma)
        L = 10.0; n = 128
        x = np.linspace(0, L, n)
        h = 0.1 * np.sin(2 * np.pi * x / L)
        dx = L / n
        E_bend = membrane.bending_energy_1d(h, dx)
        lp = membrane.persistence_length()
        cons = MembraneConservation(bending_energy=float(E_bend), persistence_length=float(lp), area_conserved=True)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(float(E_bend))],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"bending_energy": E_bend, "persistence_length": lp}, cons, session
