"""
Astrophysics package.

Domains: XII.1 Stellar Structure, XII.2 Compact Objects, XII.3 Gravitational Waves,
XII.4 Cosmological Simulations, XII.5 CMB & Early Universe, XII.6 Radiative Transfer.
"""

from .compact_objects import (
    NeutronStarEOS,
    TOVSolver,
    KerrBlackHole,
    ShakuraSunyaevDisk,
)
from .stellar_structure import (
    StellarEOS,
    StellarStructure,
    MixingLengthConvection,
    StellarOpacity,
    NuclearBurning,
)
from .gravitational_waves import (
    PostNewtonianInspiral,
    QuasiNormalRingdown,
    MatchedFilter,
    GWEnergy,
)
from .cosmological_sims import (
    FriedmannCosmology,
    MatterPowerSpectrum,
    ParticleMeshNBody,
    HaloMassFunction,
)
from .cmb_early_universe import (
    Recombination,
    CMBPowerSpectrum,
    SlowRollInflation,
    BoltzmannHierarchy,
)
from .radiative_transfer import (
    RadiativeTransfer1D,
    LambdaIteration,
    DiscreteOrdinates,
    MonteCarloRT,
    EddingtonApproximation,
)

__all__ = [
    # XII.1 Stellar Structure
    "StellarEOS",
    "StellarStructure",
    "MixingLengthConvection",
    "StellarOpacity",
    "NuclearBurning",
    # XII.2 Compact Objects
    "NeutronStarEOS",
    "TOVSolver",
    "KerrBlackHole",
    "ShakuraSunyaevDisk",
    # XII.3 Gravitational Waves
    "PostNewtonianInspiral",
    "QuasiNormalRingdown",
    "MatchedFilter",
    "GWEnergy",
    # XII.4 Cosmological Simulations
    "FriedmannCosmology",
    "MatterPowerSpectrum",
    "ParticleMeshNBody",
    "HaloMassFunction",
    # XII.5 CMB & Early Universe
    "Recombination",
    "CMBPowerSpectrum",
    "SlowRollInflation",
    "BoltzmannHierarchy",
    # XII.6 Radiative Transfer
    "RadiativeTransfer1D",
    "LambdaIteration",
    "DiscreteOrdinates",
    "MonteCarloRT",
    "EddingtonApproximation",
]
