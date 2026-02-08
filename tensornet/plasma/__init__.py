"""
Plasma physics package: extended MHD, laser-plasma interactions,
space & astrophysical plasma.

Domains: XI.2, XI.6, XI.8.
"""

from .extended_mhd import (
    GeneralisedOhm,
    HallMHDSolver1D,
    TwoFluidPlasma,
    GyroviscousMHD,
)
from .laser_plasma import (
    LaserPlasmaParams,
    StimulatedRamanScattering,
    StimulatedBrillouinScattering,
    RelativisticSelfFocusing,
    ParametricInstability,
)
from .space_plasma import (
    ParkerSolarWind,
    ParkerTransportEquation,
    BlandfordZnajek,
    MeanFieldDynamo,
)

__all__ = [
    "GeneralisedOhm",
    "HallMHDSolver1D",
    "TwoFluidPlasma",
    "GyroviscousMHD",
    "LaserPlasmaParams",
    "StimulatedRamanScattering",
    "StimulatedBrillouinScattering",
    "RelativisticSelfFocusing",
    "ParametricInstability",
    "ParkerSolarWind",
    "ParkerTransportEquation",
    "BlandfordZnajek",
    "MeanFieldDynamo",
]
