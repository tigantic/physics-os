"""
Plasma physics package: extended MHD, laser-plasma interactions,
space & astrophysical plasma, gyrokinetics, reconnection, dusty plasmas.

Domains: XI.2, XI.4, XI.5, XI.6, XI.7, XI.8.
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
from .gyrokinetics import (
    GKParameters,
    ITGDispersion,
    TEMDispersion,
    ETGDispersion,
    GyrokineticVlasov1D,
)
from .magnetic_reconnection import (
    SweetParkerReconnection,
    PetschekReconnection,
    PlasmoidInstability,
    TearingMode,
)
from .dusty_plasmas import (
    DustyPlasmaParams,
    OMLGrainCharging,
    DustAcousticWave,
    YukawaOCP,
)

__all__ = [
    # Extended MHD (XI.2)
    "GeneralisedOhm", "HallMHDSolver1D", "TwoFluidPlasma", "GyroviscousMHD",
    # Laser-Plasma (XI.6)
    "LaserPlasmaParams", "StimulatedRamanScattering",
    "StimulatedBrillouinScattering", "RelativisticSelfFocusing",
    "ParametricInstability",
    # Space Plasma (XI.8)
    "ParkerSolarWind", "ParkerTransportEquation",
    "BlandfordZnajek", "MeanFieldDynamo",
    # Gyrokinetics (XI.4)
    "GKParameters", "ITGDispersion", "TEMDispersion",
    "ETGDispersion", "GyrokineticVlasov1D",
    # Magnetic Reconnection (XI.5)
    "SweetParkerReconnection", "PetschekReconnection",
    "PlasmoidInstability", "TearingMode",
    # Dusty Plasmas (XI.7)
    "DustyPlasmaParams", "OMLGrainCharging",
    "DustAcousticWave", "YukawaOCP",
]
