"""Quantum mechanics solvers for single-particle Schrödinger equation."""

from .stationary import (
    DVRSolver,
    ShootingMethodSolver,
    SpectralSolver,
    WKBApproximation,
    HydrogenAtom,
    HarmonicOscillator,
    EigenResult,
)
from .propagator import (
    SplitOperatorPropagator,
    CrankNicolsonPropagator,
    ChebyshevPropagator,
    WavepacketTunneling,
    PropagationResult,
)
from .path_integrals import (
    PIMC,
    RPMD,
    InstantonSolver,
    ThermodynamicIntegration,
)

__all__ = [
    # Stationary
    "DVRSolver",
    "ShootingMethodSolver",
    "SpectralSolver",
    "WKBApproximation",
    "HydrogenAtom",
    "HarmonicOscillator",
    "EigenResult",
    # Time-dependent
    "SplitOperatorPropagator",
    "CrankNicolsonPropagator",
    "ChebyshevPropagator",
    "WavepacketTunneling",
    "PropagationResult",
    # Path integrals
    "PIMC",
    "RPMD",
    "InstantonSolver",
    "ThermodynamicIntegration",
]
