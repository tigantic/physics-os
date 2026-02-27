"""
Coupled physics package: MHD-coupled flows, magnetoconvection, EM pumps,
thermo-mechanical coupling, electro-mechanical coupling.

Domains: XVIII.2, XVIII.3, XVIII.4.
"""

from .coupled_mhd import (
    HartmannFlow,
    CzochralskiMHD,
    EMPump,
    Magnetoconvection,
)
from .thermo_mechanical import (
    ThermoelasticSolver,
    ThermalBuckling,
    WeldingResidualStress,
    CastingSolidificationStress,
)
from .electro_mechanical import (
    PiezoelectricSolver,
    MEMSPullIn,
    ElectrostrictiveMaterial,
    CombDriveActuator,
)

__all__ = [
    # MHD
    "HartmannFlow", "CzochralskiMHD", "EMPump", "Magnetoconvection",
    # Thermo-Mechanical (XVIII.2)
    "ThermoelasticSolver", "ThermalBuckling",
    "WeldingResidualStress", "CastingSolidificationStress",
    # Electro-Mechanical (XVIII.3)
    "PiezoelectricSolver", "MEMSPullIn",
    "ElectrostrictiveMaterial", "CombDriveActuator",
]
