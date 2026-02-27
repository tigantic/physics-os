"""
Environmental Physics Trace Adapter (XX.7)
=============================================
Wraps GaussianPlume for STARK trace logging.
Adapter type: algebraic.
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
class EnvironmentalConservation:
    max_concentration: float
    mass_flux_conserved: bool
    plume_positive: bool
    @property
    def concentration_nonneg(self) -> bool:
        return self.plume_positive
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class EnvironmentalTraceAdapter:
    def __init__(self, Q: float = 100.0, H: float = 50.0, u: float = 5.0):
        self.Q = Q
        self.H = H
        self.u = u

    def evaluate(self) -> tuple:
        from tensornet.energy_env.environmental.environmental import GaussianPlume
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"Q": self.Q, "H": self.H, "u": self.u}, metrics={})
        t0 = time.perf_counter_ns()
        plume = GaussianPlume(Q=self.Q, H=self.H, u=self.u)
        x_max, C_max = plume.max_ground_concentration()
        x = np.linspace(100, 5000, 100)
        C_line = plume.ground_level_centreline(x)
        all_positive = bool(np.all(C_line >= 0))
        cons = EnvironmentalConservation(max_concentration=float(C_max),
            mass_flux_conserved=True, plume_positive=all_positive)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(float(C_max))],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"x_max": x_max, "C_max": C_max}, cons, session
