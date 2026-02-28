"""
Coupled Physics Trace Adapters (Category XVIII)
=================================================

7 adapters covering all coupled-physics sub-domains:
  XVIII.1 — Fluid-Structure Interaction
  XVIII.2 — Thermo-Mechanical
  XVIII.3 — Electro-Mechanical
  XVIII.4 — Coupled MHD
  XVIII.5 — Reacting Flows
  XVIII.6 — Radiation Hydrodynamics
  XVIII.7 — Multiscale

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from ontic.fluids.coupled.trace_adapters.fsi_adapter import (
    FSITraceAdapter,
)
from ontic.fluids.coupled.trace_adapters.thermo_mechanical_adapter import (
    ThermoMechanicalTraceAdapter,
)
from ontic.fluids.coupled.trace_adapters.electro_mechanical_adapter import (
    ElectroMechanicalTraceAdapter,
)
from ontic.fluids.coupled.trace_adapters.coupled_mhd_adapter import (
    CoupledMHDTraceAdapter,
)
from ontic.fluids.coupled.trace_adapters.reacting_flows_adapter import (
    ReactingFlowsTraceAdapter,
)
from ontic.fluids.coupled.trace_adapters.radiation_hydro_adapter import (
    RadiationHydroTraceAdapter,
)
from ontic.fluids.coupled.trace_adapters.multiscale_adapter import (
    MultiscaleTraceAdapter,
)

__all__ = [
    "FSITraceAdapter",
    "ThermoMechanicalTraceAdapter",
    "ElectroMechanicalTraceAdapter",
    "CoupledMHDTraceAdapter",
    "ReactingFlowsTraceAdapter",
    "RadiationHydroTraceAdapter",
    "MultiscaleTraceAdapter",
]
