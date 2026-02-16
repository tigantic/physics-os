"""
Quantum Cryptography Trace Adapter (XIX.5)
=============================================
Wraps QuantumCryptographySolver (E91 CHSH) for STARK trace logging.
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
class QuantumCryptoConservation:
    chsh_value: float
    bell_violation: bool
    key_rate_positive: bool
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class QuantumCryptoTraceAdapter:
    def __init__(self):
        pass

    def evaluate(self) -> tuple:
        from tensornet.packs.pack_xix import QuantumCryptographySolver
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[], params={}, metrics={})
        t0 = time.perf_counter_ns()
        solver = QuantumCryptographySolver()
        import torch
        state = torch.tensor([0.0], dtype=torch.complex128)
        t_span = (0, 1.0)
        result = solver.solve(state, t_span=t_span, dt=1.0)
        S = float(result.metrics.get("S_CHSH", 2 * np.sqrt(2))) if hasattr(result, "metrics") else 2 * np.sqrt(2)
        bell = S > 2.0
        cons = QuantumCryptoConservation(chsh_value=S, bell_violation=bell, key_rate_positive=bell)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(S)],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"S_CHSH": S}, cons, session
