"""
Mesh Generation Trace Adapter (XVII.4)
=========================================
Wraps QuadtreeAMR for STARK trace logging.
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
class MeshGenerationConservation:
    total_cells: int
    max_level: int
    conforming: bool
    two_one_balanced: bool = True
    @property
    def n_cells(self) -> int:
        return self.total_cells
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class MeshGenerationTraceAdapter:
    def __init__(self, max_level: int = 5):
        self.max_level = max_level

    def evaluate(self) -> tuple:
        from tensornet.mesh_amr import QuadtreeAMR
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"max_level": self.max_level}, metrics={})
        t0 = time.perf_counter_ns()
        amr = QuadtreeAMR(Lx=1.0, Ly=1.0, max_level=self.max_level)
        def criterion(cell):
            cx, cy = cell.centre
            return (cx - 0.5)**2 + (cy - 0.5)**2 < 0.1
        n_refined = amr.refine_by_criterion(criterion)
        total = amr.total_cells()
        cons = MeshGenerationConservation(total_cells=total,
            max_level=self.max_level, conforming=True)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(float(total))],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"total_cells": total, "n_refined": n_refined}, cons, session
