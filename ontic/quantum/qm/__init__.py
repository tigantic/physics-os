"""
Quantum mechanics package: scattering theory, semiclassical (WKB) methods.

Domains: VI.3, VI.4.
"""

from .scattering import (
    PartialWaveScattering,
    BornApproximation,
    RMatrixScattering,
    BreitWignerResonance,
)
from .semiclassical_wkb import (
    WKBSolver,
    TullySurfaceHopping,
    HermanKlukPropagator,
)

__all__ = [
    # Scattering (VI.3)
    "PartialWaveScattering", "BornApproximation",
    "RMatrixScattering", "BreitWignerResonance",
    # Semiclassical / WKB (VI.4)
    "WKBSolver", "TullySurfaceHopping", "HermanKlukPropagator",
]
