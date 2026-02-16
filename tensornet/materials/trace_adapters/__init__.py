"""
Materials Science Trace Adapters (Category XIV)
=================================================

7 adapters covering all materials sub-domains:
  XIV.1 — First Principles (Birch-Murnaghan EOS)
  XIV.2 — Mechanical Properties (Elastic Tensor)
  XIV.3 — Phase Field (Cahn-Hilliard)
  XIV.4 — Microstructure (Multi-Phase-Field Grain Growth)
  XIV.5 — Radiation Damage (NRT Displacements)
  XIV.6 — Polymers / Soft Matter (SCFT)
  XIV.7 — Ceramics (Sintering)

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from tensornet.materials.trace_adapters.first_principles_adapter import (
    FirstPrinciplesTraceAdapter,
)
from tensornet.materials.trace_adapters.mechanical_properties_adapter import (
    MechanicalPropertiesTraceAdapter,
)
from tensornet.materials.trace_adapters.phase_field_adapter import (
    PhaseFieldTraceAdapter,
)
from tensornet.materials.trace_adapters.microstructure_adapter import (
    MicrostructureTraceAdapter,
)
from tensornet.materials.trace_adapters.radiation_damage_adapter import (
    RadiationDamageTraceAdapter,
)
from tensornet.materials.trace_adapters.polymers_adapter import (
    PolymersTraceAdapter,
)
from tensornet.materials.trace_adapters.ceramics_adapter import (
    CeramicsTraceAdapter,
)

__all__ = [
    "FirstPrinciplesTraceAdapter",
    "MechanicalPropertiesTraceAdapter",
    "PhaseFieldTraceAdapter",
    "MicrostructureTraceAdapter",
    "RadiationDamageTraceAdapter",
    "PolymersTraceAdapter",
    "CeramicsTraceAdapter",
]
