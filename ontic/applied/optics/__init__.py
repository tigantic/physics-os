"""
ontic.optics — Physical & wave optics.

Modules:
    physical_optics  Diffraction, polarization, angular spectrum propagation
"""

from ontic.applied.optics.physical_optics import (
    FresnelPropagator,
    FraunhoferPropagator,
    AngularSpectrumPropagator,
    JonesVector,
    JonesMatrix,
    MuellerMatrix,
    StokesVector,
    ThinFilmStack,
    GaussianBeam,
)
from ontic.applied.optics.quantum_optics import (
    JaynesCummingsModel,
    PhotonStatistics,
    SqueezedState,
    HongOuMandel,
)
from ontic.applied.optics.laser_physics import (
    FourLevelLaser,
    GaussianBeam as LaserGaussianBeam,
    FabryPerotCavity,
)
from ontic.applied.optics.ultrafast_optics import (
    UltrafastPulse,
    SplitStepFourier,
    SelfPhaseModulation,
    Autocorrelation,
)

__all__ = [
    "FresnelPropagator",
    "FraunhoferPropagator",
    "AngularSpectrumPropagator",
    "JonesVector",
    "JonesMatrix",
    "MuellerMatrix",
    "StokesVector",
    "ThinFilmStack",
    "GaussianBeam",
    "JaynesCummingsModel",
    "PhotonStatistics",
    "SqueezedState",
    "HongOuMandel",
    "FourLevelLaser",
    "LaserGaussianBeam",
    "FabryPerotCavity",
    "UltrafastPulse",
    "SplitStepFourier",
    "SelfPhaseModulation",
    "Autocorrelation",
]
