"""
HPC Trace Adapter (XVII.6)
=============================
Meta-adapter: verifies bit-exact reproducibility of tensor operations.
Adapter type: meta.
© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""
from __future__ import annotations
import hashlib, time
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from ontic.core.trace import TraceSession

def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.float64(v).tobytes()).hexdigest()

@dataclass
class HPCConservation:
    reproducible: bool
    hash_match: bool
    n_trials: int
    @property
    def bit_exact(self) -> bool:
        return self.reproducible and self.hash_match
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class HPCTraceAdapter:
    def __init__(self):
        pass

    def evaluate(self, n_trials: int = 5) -> tuple:
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"n_trials": n_trials}, metrics={})
        t0 = time.perf_counter_ns()
        rng = np.random.RandomState(42)
        A = rng.randn(64, 64)
        B = rng.randn(64, 64)
        hashes = []
        for _ in range(n_trials):
            C = A @ B
            hashes.append(_hash_array(C))
        all_match = all(h == hashes[0] for h in hashes)
        cons = HPCConservation(reproducible=all_match, hash_match=all_match, n_trials=n_trials)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[_hash_array(A)],
            output_hashes=[hashes[0]], params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"hash": hashes[0], "all_match": all_match}, cons, session
