"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              C L A S S I C A L   M E C H A N I C S   M O D U L E          ║
║                                                                            ║
║  Production-grade classical mechanics solvers for HyperTensor.             ║
║  Covers I.2 (Lagrangian/Hamiltonian), I.3 (Continuum),                    ║
║  I.4 (Structural) of the 140-domain taxonomy.                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from ontic.materials.mechanics.symplectic import (
    ruth4,
    yoshida6,
    yoshida8,
    composition_integrator,
    SymplecticIntegratorSuite,
)

from ontic.materials.mechanics.noether import (
    NoetherVerifier,
    ConservationLaw,
    SymmetryGenerator,
    verify_energy_conservation,
    verify_momentum_conservation,
    verify_angular_momentum_conservation,
)

from ontic.materials.mechanics.variational import (
    DiscreteVariationalIntegrator,
    DEL_Integrator,
    ActionMinimizer,
    VariationalPrincipleVerifier,
)

from ontic.materials.mechanics.structural import (
    TimoshenkoBeam,
    MindlinReissnerPlate,
    lanczos_eigensolver,
    eigenvalue_buckling,
    CompositeLamina,
    CompositeLaminate,
    assemble_beam_system,
)

__all__ = [
    # Symplectic integrators
    "ruth4",
    "yoshida6",
    "yoshida8",
    "composition_integrator",
    "SymplecticIntegratorSuite",
    # Noether conservation
    "NoetherVerifier",
    "ConservationLaw",
    "SymmetryGenerator",
    "verify_energy_conservation",
    "verify_momentum_conservation",
    "verify_angular_momentum_conservation",
    # Variational
    "DiscreteVariationalIntegrator",
    "DEL_Integrator",
    "ActionMinimizer",
    "VariationalPrincipleVerifier",
    # Structural
    "TimoshenkoBeam",
    "MindlinReissnerPlate",
    "lanczos_eigensolver",
    "eigenvalue_buckling",
    "CompositeLamina",
    "CompositeLaminate",
    "assemble_beam_system",
]
