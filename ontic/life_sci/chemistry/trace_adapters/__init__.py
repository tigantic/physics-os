"""
Chemical Physics Trace Adapters (Category XV)
================================================

4 adapters covering chemical physics sub-domains:
  XV.3 — Nonadiabatic Dynamics
  XV.4 — Photochemistry  
  XV.5 — Quantum Reactive Scattering
  XV.7 — Spectroscopy

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from ontic.life_sci.chemistry.trace_adapters.nonadiabatic_adapter import (
    NonadiabaticTraceAdapter,
)
from ontic.life_sci.chemistry.trace_adapters.photochemistry_adapter import (
    PhotochemistryTraceAdapter,
)
from ontic.life_sci.chemistry.trace_adapters.quantum_reactive_adapter import (
    QuantumReactiveTraceAdapter,
)
from ontic.life_sci.chemistry.trace_adapters.spectroscopy_adapter import (
    SpectroscopyTraceAdapter,
)

__all__ = [
    "NonadiabaticTraceAdapter",
    "PhotochemistryTraceAdapter",
    "QuantumReactiveTraceAdapter",
    "SpectroscopyTraceAdapter",
]
