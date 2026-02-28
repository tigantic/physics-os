"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║              G E O M E T R I C   T Y P E   S Y S T E M                                  ║
║                                                                                          ║
║     "QTT solved compression. Geometric Types solve correctness."                        ║
║                                                                                          ║
║     Mathematical objects as first-class citizens with compile-time invariants.          ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

The Stack:
┌─────────────────────────────────────┐
│  INTENT                             │  "simulate turbulent flow"
├─────────────────────────────────────┤
│  GEOMETRIC TYPES                    │  VectorField, Measure, Manifold, Spinor  ← YOU ARE HERE
├─────────────────────────────────────┤
│  GENESIS PRIMITIVES                 │  OT, SGW, RKHS, PH, GA
├─────────────────────────────────────┤
│  QTT                                │  Compressed tensor trains
├─────────────────────────────────────┤
│  ontic                          │  MPS/MPO operations
├─────────────────────────────────────┤
│  pytorch                            │  Dense tensor ops, autodiff
├─────────────────────────────────────┤
│  python                             │  Host language
└─────────────────────────────────────┘

This module provides:
- Type-safe mathematical objects (VectorField, Measure, Manifold, Spinor, etc.)
- Compile-time constraint checking (Divergence=0, Symplectic, etc.)
- Structure-preserving operation verification
- QTT as the universal runtime encoding

Author: TiganticLabz Geometric Types Protocol
Date: January 27, 2026
"""

from ontic.types.spaces import (
    Space,
    EuclideanSpace,
    R1, R2, R3, R4,
    Sphere, S1, S2,
    Torus, T2, T3,
    Manifold,
    TangentSpace,
    CotangentSpace,
)

from ontic.types.constraints import (
    Constraint,
    Divergence,
    Curl,
    Gradient,
    Laplacian,
    Conserved,
    Symplectic,
    Unitary,
    Orthogonal,
    Normalized,
    Positive,
    Symmetric,
    Antisymmetric,
    Traceless,
    Hermitian,
)

from ontic.types.fields import (
    Field,
    ScalarField,
    VectorField,
    TensorField,
    SpinorField,
    DifferentialForm,
    OneForm,
    TwoForm,
    TopForm,
)

from ontic.types.operators import (
    Operator,
    LinearOperator,
    DifferentialOperator,
    IntegralOperator,
    GreenFunction,
    Propagator,
)

from ontic.types.measures import (
    Measure,
    ProbabilityMeasure,
    LebesgueMeasure,
    DiracMeasure,
    GaussianMeasure,
    HaarMeasure,
)

from ontic.types.algebraic import (
    LieGroup,
    LieAlgebra,
    FiberBundle,
    PrincipalBundle,
    AssociatedBundle,
    Connection,
    Curvature,
)

from ontic.types.invariants import (
    verify_invariant,
    InvariantViolation,
    TypeChecker,
)

from ontic.types.evolution import (
    evolve,
    Flow,
    Hamiltonian,
    Lagrangian,
    ActionFunctional,
)

__all__ = [
    # Spaces
    "Space", "EuclideanSpace", "R1", "R2", "R3", "R4",
    "Sphere", "S1", "S2", "Torus", "T2", "T3",
    "Manifold", "TangentSpace", "CotangentSpace",
    # Constraints
    "Constraint", "Divergence", "Curl", "Gradient", "Laplacian",
    "Conserved", "Symplectic", "Unitary", "Orthogonal", "Normalized",
    "Positive", "Symmetric", "Antisymmetric", "Traceless", "Hermitian",
    # Fields
    "Field", "ScalarField", "VectorField", "TensorField", "SpinorField",
    "DifferentialForm", "OneForm", "TwoForm", "TopForm",
    # Operators
    "Operator", "LinearOperator", "DifferentialOperator",
    "IntegralOperator", "GreenFunction", "Propagator",
    # Measures
    "Measure", "ProbabilityMeasure", "LebesgueMeasure",
    "DiracMeasure", "GaussianMeasure", "HaarMeasure",
    # Algebraic
    "LieGroup", "LieAlgebra", "FiberBundle", "PrincipalBundle",
    "AssociatedBundle", "Connection", "Curvature",
    # Invariants
    "verify_invariant", "InvariantViolation", "TypeChecker",
    # Evolution
    "evolve", "Flow", "Hamiltonian", "Lagrangian", "ActionFunctional",
]

__version__ = "1.0.0"
