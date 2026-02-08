"""
Materials science package: mechanical properties, ceramics.

Domains: XIV.2, XIV.7.
"""

from .mechanical_properties import (
    ElasticTensor,
    FrenkelStrength,
    GriffithFracture,
    ParisFatigue,
    CreepModel,
)
from .ceramics import (
    SinteringModel,
    UHTCOxidation,
    ThermalBarrierCoating,
)

__all__ = [
    "ElasticTensor",
    "FrenkelStrength",
    "GriffithFracture",
    "ParisFatigue",
    "CreepModel",
    "SinteringModel",
    "UHTCOxidation",
    "ThermalBarrierCoating",
]
