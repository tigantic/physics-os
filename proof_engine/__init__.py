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
from .convergence import (
    ConvergenceRate, classify_rate, ConvergenceCertificate,
    MonotoneEnergyCertificate, ContractionMapCertificate,
    LanczosConvergenceCertificate, FixedPointCertificate,
    CauchyCriterionCertificate,
    convergence_to_lean, convergence_to_coq, convergence_to_isabelle,
)
from .well_posedness import (
    PDEType, FunctionSpace, WellPosednessCertificate,
    LerayHopfCertificate, LaxMilgramCertificate,
    EnergyEstimateCertificate, GronwallCertificate,
    StokesRegularityCertificate,
    wellposedness_to_lean, wellposedness_to_coq, wellposedness_to_isabelle,
)
from .coq_export import CoqTheorem, CoqDefinition, CoqLemma, CoqExporter
from .isabelle_export import (
    IsabelleLemma, IsabelleDefinition, IsabelleTheory, IsabelleExporter,
)
from .proof_carrying import (
    ProofTag, ProofAnnotation, PCCPayload,
    verify_conservation, verify_bound, verify_monotone, verify_positivity,
    verify_payload, annotate, PCCRegistry,
)
from .dashboard import (
    Verdict, ProofLayer, ProofStatus, CoverageEntry, Anomaly, ProofDashboard,
)
from .thermodynamic import (
    ThermoState, FirstLawCertificate, SecondLawCertificate, ThirdLawCertificate,
    check_maxwell_relations, check_onsager_reciprocity,
    ThermodynamicAudit, run_thermodynamic_audit, thermodynamic_to_lean,
)
from .cross_proof import (
    ProofSystem, NodeStatus, ProofNode, InterfaceContract,
    ProofGraph, check_transitivity, link_lean_to_interval,
)

__version__ = "2.0.0"
__all__ = [
    # Interval arithmetic
    'Interval', 'IntervalTensor', 'interval', 'itensor',
    # Certificates
    'Certificate', 'MassGapCertificate', 'WitnessGenerator', 'RigorousChecker',
    # Lean export
    'LeanExporter', 'LeanTheorem',
    # Constructive QFT
    'BetaFunction', 'RGFlow', 'DimensionalTransmutation', 'RGStep',
    # Convergence proofs (4.3)
    'ConvergenceRate', 'classify_rate', 'ConvergenceCertificate',
    'MonotoneEnergyCertificate', 'ContractionMapCertificate',
    'LanczosConvergenceCertificate', 'FixedPointCertificate',
    'CauchyCriterionCertificate',
    'convergence_to_lean', 'convergence_to_coq', 'convergence_to_isabelle',
    # Well-posedness proofs (4.4)
    'PDEType', 'FunctionSpace', 'WellPosednessCertificate',
    'LerayHopfCertificate', 'LaxMilgramCertificate',
    'EnergyEstimateCertificate', 'GronwallCertificate',
    'StokesRegularityCertificate',
    'wellposedness_to_lean', 'wellposedness_to_coq', 'wellposedness_to_isabelle',
    # Coq export (4.5)
    'CoqTheorem', 'CoqDefinition', 'CoqLemma', 'CoqExporter',
    # Isabelle/HOL export (4.6)
    'IsabelleLemma', 'IsabelleDefinition', 'IsabelleTheory', 'IsabelleExporter',
    # Proof-carrying code (4.8)
    'ProofTag', 'ProofAnnotation', 'PCCPayload',
    'verify_conservation', 'verify_bound', 'verify_monotone', 'verify_positivity',
    'verify_payload', 'annotate', 'PCCRegistry',
    # Dashboard (4.10)
    'Verdict', 'ProofLayer', 'ProofStatus', 'CoverageEntry', 'Anomaly', 'ProofDashboard',
    # Thermodynamic consistency (4.13)
    'ThermoState', 'FirstLawCertificate', 'SecondLawCertificate', 'ThirdLawCertificate',
    'check_maxwell_relations', 'check_onsager_reciprocity',
    'ThermodynamicAudit', 'run_thermodynamic_audit', 'thermodynamic_to_lean',
    # Cross-proof linking (4.14)
    'ProofSystem', 'NodeStatus', 'ProofNode', 'InterfaceContract',
    'ProofGraph', 'check_transitivity', 'link_lean_to_interval',
]
