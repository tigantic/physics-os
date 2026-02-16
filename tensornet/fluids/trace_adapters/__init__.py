"""
Fluids Trace Adapters — Phase 6 Tier 2A
=========================================

Trace adapters for 8 remaining fluid dynamics domains:
  - Turbulence (II.3)
  - Multiphase (II.4)
  - Reactive Flow (II.5)
  - Rarefied Gas (II.6)
  - Shallow Water (II.7)
  - Non-Newtonian (II.8)
  - Porous Media (II.9)
  - Free Surface (II.10)

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from tensornet.fluids.trace_adapters.turbulence_adapter import TurbulenceTraceAdapter
from tensornet.fluids.trace_adapters.multiphase_adapter import MultiphaseTraceAdapter
from tensornet.fluids.trace_adapters.reactive_adapter import ReactiveFlowTraceAdapter
from tensornet.fluids.trace_adapters.rarefied_adapter import RarefiedGasTraceAdapter
from tensornet.fluids.trace_adapters.shallow_water_adapter import ShallowWaterTraceAdapter
from tensornet.fluids.trace_adapters.non_newtonian_adapter import NonNewtonianTraceAdapter
from tensornet.fluids.trace_adapters.porous_media_adapter import PorousMediaTraceAdapter
from tensornet.fluids.trace_adapters.free_surface_adapter import FreeSurfaceTraceAdapter

__all__ = [
    "TurbulenceTraceAdapter",
    "MultiphaseTraceAdapter",
    "ReactiveFlowTraceAdapter",
    "RarefiedGasTraceAdapter",
    "ShallowWaterTraceAdapter",
    "NonNewtonianTraceAdapter",
    "PorousMediaTraceAdapter",
    "FreeSurfaceTraceAdapter",
]
