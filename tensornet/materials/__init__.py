"""
Materials science package.

Domains: XIV.1 First-Principles Design, XIV.2 Mechanical Properties,
XIV.4 Microstructure, XIV.5 Radiation Damage, XIV.6 Polymers & Soft Matter,
XIV.7 Ceramics.
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
from .first_principles_design import (
    BirchMurnaghanEOS,
    ElasticConstants,
    ConvexHullStability,
    PhononDispersion1D,
)
from .microstructure import (
    CahnHilliard2D,
    AllenCahn2D,
    MultiPhaseFieldGrainGrowth,
    ClassicalNucleation,
)
from .radiation_damage import (
    NRTDisplacements,
    BinaryCollisionApproximation,
    StoppingPower,
    FrenkelPairThermodynamics,
)
from .polymers_soft_matter import (
    IdealChainStatistics,
    FloryHuggins,
    SCFT1D,
    ReptationModel,
    RubberElasticity,
)

__all__ = [
    # XIV.1 First-Principles Design
    "BirchMurnaghanEOS",
    "ElasticConstants",
    "ConvexHullStability",
    "PhononDispersion1D",
    # XIV.2 Mechanical Properties
    "ElasticTensor",
    "FrenkelStrength",
    "GriffithFracture",
    "ParisFatigue",
    "CreepModel",
    # XIV.4 Microstructure
    "CahnHilliard2D",
    "AllenCahn2D",
    "MultiPhaseFieldGrainGrowth",
    "ClassicalNucleation",
    # XIV.5 Radiation Damage
    "NRTDisplacements",
    "BinaryCollisionApproximation",
    "StoppingPower",
    "FrenkelPairThermodynamics",
    # XIV.6 Polymers & Soft Matter
    "IdealChainStatistics",
    "FloryHuggins",
    "SCFT1D",
    "ReptationModel",
    "RubberElasticity",
    # XIV.7 Ceramics
    "SinteringModel",
    "UHTCOxidation",
    "ThermalBarrierCoating",
]
