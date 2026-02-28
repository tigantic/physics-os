"""
ontic.em — Computational Electromagnetics (Python layer)

Modules:
    electrostatics           Poisson-Boltzmann, multipole expansion, capacitance extraction
    magnetostatics           Biot-Savart, vector potential, magnetic dipoles
    frequency_domain         FDFD, Method of Moments, RCS
    wave_propagation         FDTD, Mie scattering
    computational_photonics  Transfer matrix, coupled-mode theory, slab waveguides
    antenna_microwave        Dipole antenna, ULA, microstrip, transmission lines
"""

from ontic.em.electrostatics import (
    PoissonBoltzmannSolver,
    MultipoleExpansion,
    CapacitanceExtractor,
    ChargeDistribution,
    DebyeHuckelSolver,
    PoissonNernstPlanck,
)
from ontic.em.magnetostatics import (
    BiotSavart,
    MagneticVectorPotential2D,
    MagneticDipole,
)
from ontic.em.frequency_domain import (
    FDFD2D_TM,
    MethodOfMoments2D,
)
from ontic.em.wave_propagation import (
    FDTD1D,
    FDTD2D_TM,
    MieScattering,
)
from ontic.em.computational_photonics import (
    TransferMatrix1D,
    CoupledModeTheory,
    SlabWaveguide,
)
from ontic.em.antenna_microwave import (
    DipoleAntenna,
    UniformLinearArray,
    MicrostripPatch,
    TransmissionLine,
)

__all__ = [
    "PoissonBoltzmannSolver",
    "MultipoleExpansion",
    "CapacitanceExtractor",
    "ChargeDistribution",
    "DebyeHuckelSolver",
    "PoissonNernstPlanck",
    "BiotSavart",
    "MagneticVectorPotential2D",
    "MagneticDipole",
    "FDFD2D_TM",
    "MethodOfMoments2D",
    "FDTD1D",
    "FDTD2D_TM",
    "MieScattering",
    "TransferMatrix1D",
    "CoupledModeTheory",
    "SlabWaveguide",
    "DipoleAntenna",
    "UniformLinearArray",
    "MicrostripPatch",
    "TransmissionLine",
]
