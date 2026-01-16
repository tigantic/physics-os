"""
HyperTensor Proof Engine
========================

Computer-Assisted Proof Generator for Mathematical Physics.

This module transforms numerical simulations into rigorous proofs:

    LAYER 1 (Intuition): QTT Solver
        - Fast numerical computation
        - Finds approximate solutions
        - Output: Float64 results
    
    LAYER 2 (Certifier): Interval Arithmetic
        - Wraps computations in rigorous bounds
        - Every number becomes [lower, upper]
        - Output: Guaranteed bounds
    
    LAYER 3 (Formalizer): Lean 4 Integration
        - Exports certificates to formal proofs
        - Type-checked by Lean compiler
        - Output: Verified theorem

The key insight: we don't need to prove HOW we found the answer,
we only need to prove THAT the answer is correct.

Usage:
    from proof_engine import Interval, Certificate, LeanExporter
    
    # Create rigorous interval
    g = Interval.from_float(0.2)
    
    # Generate witness
    witness = WitnessGenerator()
    cert = witness.generate_mass_gap_witness(g, L=32, χ=100)
    
    # Verify certificate
    checker = RigorousChecker()
    assert cert.verify(checker)
    
    # Export to Lean 4
    exporter = LeanExporter()
    lean_code = exporter.export_full_proof(cert)
"""

from .interval import Interval, IntervalTensor, interval, itensor
from .certificate import Certificate, MassGapCertificate, WitnessGenerator, RigorousChecker
from .lean_export import LeanExporter, LeanTheorem
from .constructive_qft import BetaFunction, RGFlow, DimensionalTransmutation, RGStep

__version__ = "1.0.0"
__all__ = [
    # Interval arithmetic
    'Interval',
    'IntervalTensor',
    'interval',
    'itensor',
    # Certificates
    'Certificate',
    'MassGapCertificate',
    'WitnessGenerator',
    'RigorousChecker',
    # Lean export
    'LeanExporter',
    'LeanTheorem',
    # Constructive QFT
    'BetaFunction',
    'RGFlow',
    'DimensionalTransmutation',
    'RGStep',
]
