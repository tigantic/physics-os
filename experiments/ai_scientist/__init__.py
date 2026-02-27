"""
AI Scientist v1.0
=================

The machine that solves physics.

Architecture:
    Phase 1: Conjecturer - Neural Symbolic Regression (PySR)
             Input: QTT tensor data
             Output: Analytic formula σ_k ≤ Ce^{-γk}
    
    Phase 2: Formalizer - PhysLean (Lean 4 + Mathlib)
             Input: Physics definitions
             Output: Formal axioms in dependent type theory
    
    Phase 3: Prover - LLM-aided Formalization
             Input: Conjecture + Axioms
             Output: Machine-verified proof

The Goal: Not to solve Yang-Mills, but to solve HOW to solve Yang-Mills.

This is the difference between:
    - Measuring 1000 triangles (Science/Checking)
    - Deriving a² + b² = c² (Mathematics/Proving)

The Hard Wall (Cubitt et al. 2015):
    The spectral gap problem is UNDECIDABLE.
    No finite computation can prove the infinite limit.
    We need LOGICAL INDUCTION, not numerical checking.

Usage:
    from ai_scientist import AIScientist, Conjecturer, Formalizer
    
    # Full pipeline
    scientist = AIScientist()
    result = scientist.solve_yang_mills(gap_data, L_values)
    
    # Or individual phases
    conjecturer = Conjecturer()
    formula = conjecturer.discover_scaling_law(L, gap)
    
    formalizer = Formalizer()
    lean_code = formalizer.generate_yang_mills_theory()
"""

__version__ = "1.0.0"

from .conjecturer import Conjecturer, DiscoveredFormula
from .formalizer import Formalizer, LeanDefinition, LeanTheorem, LeanAxiom
from .prover import AIScientistProver, ProofObligation, ProverSession
from .pipeline import AIScientist, ScientificResult

__all__ = [
    # Phase 1
    "Conjecturer",
    "DiscoveredFormula",
    # Phase 2
    "Formalizer",
    "LeanDefinition",
    "LeanTheorem",
    "LeanAxiom",
    # Phase 3
    "AIScientistProver",
    "ProofObligation",
    "ProverSession",
    # Integration
    "AIScientist",
    "ScientificResult",
]
