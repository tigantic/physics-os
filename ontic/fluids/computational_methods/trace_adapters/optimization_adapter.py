"""
Optimization Trace Adapter (XVII.1)
======================================
Wraps BSplineParameterization for STARK trace logging.
Adapter type: scf.
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
class OptimizationConservation:
    initial_objective: float
    final_objective: float
    objective_decreased: bool
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class OptimizationTraceAdapter:
    def __init__(self, n_control_points: int = 8):
        self.n_control_points = n_control_points

    def evaluate(self, n_iterations: int = 20) -> tuple:
        import torch
        from ontic.cfd.optimization import BSplineParameterization
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"n_control_points": self.n_control_points}, metrics={})
        t0 = time.perf_counter_ns()
        param = BSplineParameterization(n_control_points=self.n_control_points)
        alpha = torch.zeros(self.n_control_points * 2, dtype=torch.float64, requires_grad=True)
        losses = []
        for i in range(n_iterations):
            shape = param.evaluate(alpha)
            loss = torch.sum(shape ** 2)
            loss.backward()
            with torch.no_grad():
                alpha -= 0.01 * alpha.grad
                alpha.grad.zero_()
            losses.append(float(loss.item()))
        cons = OptimizationConservation(initial_objective=losses[0],
            final_objective=losses[-1], objective_decreased=bool(losses[-1] <= losses[0]))
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(losses[-1])],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"losses": losses}, cons, session
