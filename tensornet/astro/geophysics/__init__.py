"""
Geophysics package.

Domains: XIII.1 Seismology, XIII.2 Mantle Convection, XIII.3 Geodynamo,
XIII.4 Atmosphere, XIII.5 Oceanography, XIII.6 Glaciology.
"""

from .atmosphere import (
    ChapmanOzone,
    KesslerMicrophysics,
    ClausiusClapeyron,
    RadiativeConvectiveEquilibrium,
)
from .oceanography import (
    SeawaterEOS,
    ShallowWaterEquations,
    StommelBoxModel,
    InternalWaves,
    TidalConstituent,
    TidalHarmonics,
)
from .seismology import (
    AcousticWave2D,
    SeismicRayTracing,
    TravelTimeTomography,
    MomentTensorInversion,
)
from .mantle_convection import (
    StokesFlow2D,
    MantleConvection2D,
    MantleViscosity,
)
from .geodynamo import (
    MagneticInduction2D,
    AlphaOmegaDynamo,
    DynamoParameters,
)
from .glaciology import (
    GlenFlowLaw,
    ShallowIceApproximation,
    GlacialIsostaticAdjustment,
    IceThermodynamics1D,
)

__all__ = [
    # XIII.1 Seismology
    "AcousticWave2D",
    "SeismicRayTracing",
    "TravelTimeTomography",
    "MomentTensorInversion",
    # XIII.2 Mantle Convection
    "StokesFlow2D",
    "MantleConvection2D",
    "MantleViscosity",
    # XIII.3 Geodynamo
    "MagneticInduction2D",
    "AlphaOmegaDynamo",
    "DynamoParameters",
    # XIII.4 Atmosphere
    "ChapmanOzone",
    "KesslerMicrophysics",
    "ClausiusClapeyron",
    "RadiativeConvectiveEquilibrium",
    # XIII.5 Oceanography
    "SeawaterEOS",
    "ShallowWaterEquations",
    "StommelBoxModel",
    "InternalWaves",
    "TidalConstituent",
    "TidalHarmonics",
    # XIII.6 Glaciology
    "GlenFlowLaw",
    "ShallowIceApproximation",
    "GlacialIsostaticAdjustment",
    "IceThermodynamics1D",
]
