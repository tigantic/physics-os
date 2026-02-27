"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YANG-MILLS QTT MODULE                              ║
║                                                                              ║
║                    Lattice Gauge Theory in Tensor Train Form                 ║
║                                                                              ║
║                         Target: Clay Millennium Prize                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module implements SU(N) lattice gauge theory using Quantized Tensor Train
(QTT) representation. The goal is to provide a constructive proof of the 
Yang-Mills existence and mass gap problem.

Key Components:
    - su2: SU(2) group operations and algebra
    - lattice: Lattice geometry and indexing
    - operators: Link operators, plaquettes, electric field
    - hamiltonian: Kogut-Susskind Hamiltonian
    - gauss: Gauss law operators and gauge invariance

Mathematical Foundation:
    - Wilson's lattice gauge theory (1974)
    - Kogut-Susskind Hamiltonian formulation (1975)
    - Tensor network / MPS / QTT representation

Author: HyperTensor Yang-Mills Project
Date: 2026-01-15
Status: Sprint 1 - Infrastructure
"""

from .su2 import (
    pauli_matrices,
    random_su2,
    su2_generators,
    spin_j_generators,
    casimir_eigenvalue,
    PAULI, TAU, SIGMA_0,
)

from .lattice import (
    Lattice,
    LatticeLink,
    LatticeSite,
    GaugeConfiguration,
)

from .operators import (
    TruncatedHilbertSpace,
    LinkOperator,
    PlaquetteOperator,
    ElectricFieldOperator,
)

from .hamiltonian import (
    SinglePlaquetteHamiltonian,
    LatticeHamiltonian,
)

from .gauss import (
    GaussOperator,
    SinglePlaquetteGauss,
)

__version__ = "0.1.0"
__all__ = [
    # SU(2)
    "pauli_matrices", 
    "random_su2",
    "su2_generators",
    "spin_j_generators",
    "casimir_eigenvalue",
    "PAULI", "TAU", "SIGMA_0",
    # Lattice
    "Lattice",
    "LatticeLink",
    "LatticeSite",
    "GaugeConfiguration",
    # Operators
    "TruncatedHilbertSpace",
    "LinkOperator",
    "PlaquetteOperator",
    "ElectricFieldOperator",
    # Hamiltonian
    "SinglePlaquetteHamiltonian",
    "LatticeHamiltonian",
    # Gauss
    "GaussOperator",
    "SinglePlaquetteGauss",
]
