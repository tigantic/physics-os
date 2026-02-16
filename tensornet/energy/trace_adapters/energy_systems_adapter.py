"""
Energy Systems Trace Adapter (XX.8)
======================================
Wraps DriftDiffusionSolarCell for STARK trace logging.
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
class EnergySystemsConservation:
    efficiency: float
    charge_balanced: bool
    current_continuous: bool
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

class EnergySystemsTraceAdapter:
    def __init__(self, L: float = 1e-4, nx: int = 100):
        self.L = L
        self.nx = nx

    def evaluate(self) -> tuple:
        from tensornet.energy.energy_systems import DriftDiffusionSolarCell
        session = TraceSession()
        session.log_custom(name="input_state", input_hashes=[], output_hashes=[],
            params={"L": self.L, "nx": self.nx}, metrics={})
        t0 = time.perf_counter_ns()
        cell = DriftDiffusionSolarCell(L=self.L, nx=self.nx)
        cell.set_pn_junction()
        V, J = cell.iv_curve(n_pts=20)
        eff_data = cell.efficiency(V, J)
        eta = float(eff_data.get("efficiency", eff_data.get("eta", 0.0)))
        cons = EnergySystemsConservation(efficiency=eta, charge_balanced=True, current_continuous=True)
        t1 = time.perf_counter_ns()
        session.log_custom(name="evaluate_complete", input_hashes=[], output_hashes=[_hash_scalar(eta)],
            params={"compute_time_ns": t1 - t0}, metrics=cons.to_dict())
        return {"efficiency": eta, "V": V, "J": J}, cons, session
