"""
Chemistry package: PES construction, reaction rate theory, photochemistry.

Domains: XV.1, XV.2, XV.5.
"""

from .pes import (
    MorsePotential,
    LEPSPotential,
    NudgedElasticBand,
    IntrinsicReactionCoordinate,
)
from .reaction_rate import (
    TransitionStateTheory,
    VariationalTST,
    RRKMTheory,
    KramersRate,
)
from .photochemistry import (
    FranckCondonFactors,
    InternalConversion,
    IntersystemCrossing,
    Photodissociation,
    FluorescenceLifetime,
)
from .quantum_reactive import (
    TransitionStateTheory as QReactiveTST,
    CollinearReactiveScattering,
    QuantumBarrierTransmission,
)
from .nonadiabatic import (
    LandauZener,
    FewestSwitchesSurfaceHopping,
    SpinBosonModel,
)
from .spectroscopy import (
    VibrationalSpectroscopy,
    ElectronicSpectroscopy,
    FranckCondonFactors as FCFactorsSpectro,
    RotationalSpectroscopy,
    NMRChemicalShift,
)

__all__ = [
    "MorsePotential",
    "LEPSPotential",
    "NudgedElasticBand",
    "IntrinsicReactionCoordinate",
    "TransitionStateTheory",
    "VariationalTST",
    "RRKMTheory",
    "KramersRate",
    "FranckCondonFactors",
    "InternalConversion",
    "IntersystemCrossing",
    "Photodissociation",
    "FluorescenceLifetime",
    "QReactiveTST",
    "CollinearReactiveScattering",
    "QuantumBarrierTransmission",
    "LandauZener",
    "FewestSwitchesSurfaceHopping",
    "SpinBosonModel",
    "VibrationalSpectroscopy",
    "ElectronicSpectroscopy",
    "FCFactorsSpectro",
    "RotationalSpectroscopy",
    "NMRChemicalShift",
]
