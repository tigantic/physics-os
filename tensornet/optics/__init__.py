"""
tensornet.optics — Physical & wave optics.

Modules:
    physical_optics  Diffraction, polarization, angular spectrum propagation
"""

from tensornet.optics.physical_optics import (
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
from tensornet.optics.quantum_optics import (
    JaynesCummingsModel,
    PhotonStatistics,
    SqueezedState,
    HongOuMandel,
)
from tensornet.optics.laser_physics import (
    FourLevelLaser,
    GaussianBeam as LaserGaussianBeam,
    FabryPerotCavity,
)
from tensornet.optics.ultrafast_optics import (
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
