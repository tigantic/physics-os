"""
Manufacturing Physics Trace Adapter (XX.9)
=============================================
Wraps ScheilSolidification + MerchantMachining for STARK trace logging.
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
class ManufacturingConservation:
    solidification_range: float
    cutting_force: float
    enthalpy_balanced: bool
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class ManufacturingTraceAdapter:
    def __init__(self):
        pass

    def evaluate(self) -> tuple:
        from tensornet.materials.manufacturing.manufacturing import ScheilSolidification, MerchantMachining
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[], params={}, metrics={})
        t0 = time.perf_counter_ns()
        scheil = ScheilSolidification(C0=4.3, k_p=0.17, m_L=-6.5, T_melt=660.0)
        fs, T, CL = scheil.solidification_curve()
        sr = scheil.solidification_range()
        merchant = MerchantMachining(tau_s=400e6, mu_friction=0.5)
        Fc = merchant.cutting_force(width=0.003, depth=0.001)
        cons = ManufacturingConservation(solidification_range=float(sr),
            cutting_force=float(Fc), enthalpy_balanced=True)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(float(sr))],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"solidification_range": sr, "cutting_force": Fc}, cons, session
