"""
TensorNet Energy Module - Wind Farm Physics & Optimization

Phase 5: Commercial Energy Simulation
Target Market: Offshore Wind Developers (Orsted, Shell Energy)

Capabilities:
- Jensen Park Wake Model (industry standard)
- Velocity deficit calculation
- Multi-turbine wake superposition
- Power output optimization
- Real-time Unreal Engine visualization
"""

from tensornet.energy.turbine import WindFarm
from tensornet.energy.energy_systems import (
    DriftDiffusionSolarCell,
    NewmanP2D,
    NeutronDiffusion,
)

__all__ = [
    "WindFarm",
    # XX.8 Energy Systems
    "DriftDiffusionSolarCell",
    "NewmanP2D",
    "NeutronDiffusion",
]
