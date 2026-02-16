"""
EM Trace Adapters — Phase 6 Tier 2A
======================================

Trace adapters for 7 electromagnetism domains:
  - Electrostatics (III.1)
  - Magnetostatics (III.2)
  - Full Maxwell FDTD (III.3)
  - Frequency-Domain EM (III.4)
  - EM Wave Propagation (III.5)
  - Computational Photonics (III.6)
  - Antenna & Microwave (III.7)

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from tensornet.em.trace_adapters.electrostatics_adapter import ElectrostaticsTraceAdapter
from tensornet.em.trace_adapters.magnetostatics_adapter import MagnetostaticsTraceAdapter
from tensornet.em.trace_adapters.maxwell_fdtd_adapter import MaxwellFDTDTraceAdapter
from tensornet.em.trace_adapters.frequency_domain_adapter import FrequencyDomainTraceAdapter
from tensornet.em.trace_adapters.wave_propagation_adapter import WavePropagationTraceAdapter
from tensornet.em.trace_adapters.photonics_adapter import PhotonicsTraceAdapter
from tensornet.em.trace_adapters.antenna_adapter import AntennaTraceAdapter

__all__ = [
    "ElectrostaticsTraceAdapter",
    "MagnetostaticsTraceAdapter",
    "MaxwellFDTDTraceAdapter",
    "FrequencyDomainTraceAdapter",
    "WavePropagationTraceAdapter",
    "PhotonicsTraceAdapter",
    "AntennaTraceAdapter",
]
