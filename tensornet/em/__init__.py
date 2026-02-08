"""
tensornet.em — Computational Electromagnetics (Python layer)

Modules:
    electrostatics           Poisson-Boltzmann, multipole expansion, capacitance extraction
    magnetostatics           Biot-Savart, vector potential, magnetic dipoles
    frequency_domain         FDFD, Method of Moments, RCS
    wave_propagation         FDTD, Mie scattering
    computational_photonics  Transfer matrix, coupled-mode theory, slab waveguides
    antenna_microwave        Dipole antenna, ULA, microstrip, transmission lines
"""

from tensornet.em.electrostatics import (
    PoissonBoltzmannSolver,
    MultipoleExpansion,
    CapacitanceExtractor,
    ChargeDistribution,
    DebyeHuckelSolver,
    PoissonNernstPlanck,
)
from tensornet.em.magnetostatics import (
    BiotSavart,
    MagneticVectorPotential2D,
    MagneticDipole,
)
from tensornet.em.frequency_domain import (
    FDFD2D_TM,
    MethodOfMoments2D,
)
from tensornet.em.wave_propagation import (
    FDTD1D,
    FDTD2D_TM,
    MieScattering,
)
from tensornet.em.computational_photonics import (
    TransferMatrix1D,
    CoupledModeTheory,
    SlabWaveguide,
)
from tensornet.em.antenna_microwave import (
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
