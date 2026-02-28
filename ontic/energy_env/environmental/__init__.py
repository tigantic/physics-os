"""
Environmental physics package: atmospheric dispersion, hydrology,
storm surge, wildfire modeling.

Domains: XX.7.
"""

from .environmental import (
    GaussianPlume,
    SCSCurveNumber,
    StormSurge1D,
    FireAtmosphere,
)

__all__ = [
    "GaussianPlume",
    "SCSCurveNumber",
    "StormSurge1D",
    "FireAtmosphere",
]
