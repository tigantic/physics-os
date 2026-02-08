"""
Geophysics package: atmospheric physics, physical oceanography.

Domains: XIII.4, XIII.5.
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

__all__ = [
    "ChapmanOzone",
    "KesslerMicrophysics",
    "ClausiusClapeyron",
    "RadiativeConvectiveEquilibrium",
    "SeawaterEOS",
    "ShallowWaterEquations",
    "StommelBoxModel",
    "InternalWaves",
    "TidalConstituent",
    "TidalHarmonics",
]
