"""
ontic.statmech — Statistical mechanics.

Modules:
    equilibrium       Classical Monte Carlo, partition functions, phase transitions
    non_equilibrium   Jarzynski, Crooks, Gillespie SSA, kinetic MC
"""

from ontic.quantum.statmech.equilibrium import (
    MetropolisMC,
    WolffClusterMC,
    WangLandauMC,
    PartitionFunction,
    LandauMeanField,
    IsingModel,
    PottsModel,
    XYModel,
)

from ontic.quantum.statmech.non_equilibrium import (
    JarzynskiEstimator,
    CrooksEstimator,
    KuboResponse,
    KineticMonteCarlo,
    GillespieSSA,
    ChemicalMasterEquation,
)

from ontic.quantum.statmech.monte_carlo import (
    SwendsenWangCluster,
    ParallelTempering,
    HistogramReweighting,
    MulticanonicalMC,
)

__all__ = [
    "MetropolisMC",
    "WolffClusterMC",
    "WangLandauMC",
    "PartitionFunction",
    "LandauMeanField",
    "IsingModel",
    "PottsModel",
    "XYModel",
    "JarzynskiEstimator",
    "CrooksEstimator",
    "KuboResponse",
    "KineticMonteCarlo",
    "GillespieSSA",
    "ChemicalMasterEquation",
    "SwendsenWangCluster",
    "ParallelTempering",
    "HistogramReweighting",
    "MulticanonicalMC",
]
