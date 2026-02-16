"""
Geophysics Trace Adapters (Category XIII)
==========================================

6 adapters covering all geophysics sub-domains:
  XIII.1 — Seismology
  XIII.2 — Mantle Convection
  XIII.3 — Geodynamo
  XIII.4 — Atmospheric Physics
  XIII.5 — Oceanography
  XIII.6 — Glaciology

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from tensornet.geophysics.trace_adapters.seismology_adapter import (
    SeismologyTraceAdapter,
)
from tensornet.geophysics.trace_adapters.mantle_convection_adapter import (
    MantleConvectionTraceAdapter,
)
from tensornet.geophysics.trace_adapters.geodynamo_adapter import (
    GeodynamoTraceAdapter,
)
from tensornet.geophysics.trace_adapters.atmospheric_adapter import (
    AtmosphericPhysicsTraceAdapter,
)
from tensornet.geophysics.trace_adapters.oceanography_adapter import (
    OceanographyTraceAdapter,
)
from tensornet.geophysics.trace_adapters.glaciology_adapter import (
    GlaciologyTraceAdapter,
)

__all__ = [
    "SeismologyTraceAdapter",
    "MantleConvectionTraceAdapter",
    "GeodynamoTraceAdapter",
    "AtmosphericPhysicsTraceAdapter",
    "OceanographyTraceAdapter",
    "GlaciologyTraceAdapter",
]
