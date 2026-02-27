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

from tensornet.fluids.coupled.trace_adapters.fsi_adapter import (
    FSITraceAdapter,
)
from tensornet.fluids.coupled.trace_adapters.thermo_mechanical_adapter import (
    ThermoMechanicalTraceAdapter,
)
from tensornet.fluids.coupled.trace_adapters.electro_mechanical_adapter import (
    ElectroMechanicalTraceAdapter,
)
from tensornet.fluids.coupled.trace_adapters.coupled_mhd_adapter import (
    CoupledMHDTraceAdapter,
)
from tensornet.fluids.coupled.trace_adapters.reacting_flows_adapter import (
    ReactingFlowsTraceAdapter,
)
from tensornet.fluids.coupled.trace_adapters.radiation_hydro_adapter import (
    RadiationHydroTraceAdapter,
)
from tensornet.fluids.coupled.trace_adapters.multiscale_adapter import (
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
