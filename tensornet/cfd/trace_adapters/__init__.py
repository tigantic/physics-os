"""
Trace Adapters — Phase 5 Tier 1
================================

Connect existing solvers to the ComputationTrace system for STARK proof generation.
Each adapter wraps a solver's step/solve method to emit deterministic trace entries
for every time-step, preserving conservation-law metadata.

Adapters:
    - euler3d: Compressible Euler 3D (mass + momentum + energy)
    - ns2d: Incompressible Navier-Stokes 2D (momentum + divergence-free)
    - heat_transfer: Thermal QTT solver (energy conservation)
    - vlasov: Vlasov-Poisson 5D/6D (L² norm + phase-space volume)
"""

from tensornet.cfd.trace_adapters.euler3d_adapter import Euler3DTraceAdapter
from tensornet.cfd.trace_adapters.ns2d_adapter import NS2DTraceAdapter
from tensornet.cfd.trace_adapters.heat_adapter import HeatTransferTraceAdapter
from tensornet.cfd.trace_adapters.vlasov_adapter import VlasovTraceAdapter

__all__ = [
    "Euler3DTraceAdapter",
    "NS2DTraceAdapter",
    "HeatTransferTraceAdapter",
    "VlasovTraceAdapter",
]
