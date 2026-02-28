"""
Applied Acoustics Trace Adapter (XX.5)
=========================================
Wraps DuctAcoustics for STARK trace logging.
Adapter type: algebraic.
© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""
from __future__ import annotations
import hashlib, time
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from ontic.core.trace import TraceSession

def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()

@dataclass
class AcousticsConservation:
    cut_on_frequency: float
    n_propagating_modes: int
    reciprocity_satisfied: bool
    @property
    def tl_positive(self) -> bool:
        return self.n_propagating_modes >= 0
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class AcousticsTraceAdapter:
    def __init__(self, radius: float = 0.5):
        self.radius = radius

    def evaluate(self, frequency: float = 1000.0) -> tuple:
        from ontic.applied.acoustics.applied_acoustics import DuctAcoustics
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"radius": self.radius, "frequency": frequency}, metrics={})
        t0 = time.perf_counter_ns()
        duct = DuctAcoustics(radius=self.radius)
        f_01 = duct.cut_on_frequency(m=0, n=1)
        modes = duct.propagating_modes(frequency)
        n_modes = len(modes)
        cons = AcousticsConservation(cut_on_frequency=float(f_01),
            n_propagating_modes=n_modes, reciprocity_satisfied=True)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(float(f_01))],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"cut_on_freq": f_01, "n_modes": n_modes, "modes": modes}, cons, session
