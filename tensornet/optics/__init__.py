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
]
