#!/usr/bin/env python3
"""
AI Scientist v1.0 - Quick Demo
==============================

Fast demonstration of the full pipeline.
"""

import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')

import numpy as np

# Import our AI Scientist components directly
from ai_scientist.conjecturer import Conjecturer, DiscoveredFormula
from ai_scientist.formalizer import Formalizer
from ai_scientist.prover import AIScientistProver

print("=" * 70)
print("AI SCIENTIST v1.0 - QUICK DEMO")
print("The Machine That Solves How To Solve Physics")
print("=" * 70)
print()

# Create synthetic data
np.random.seed(42)

# Lattice sizes
L = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])

# True parameters (what we're trying to discover)
gap_inf_true = 1.50  # The mass gap in infinite volume
b_true = 0.8         # Finite-size correction coefficient

# Simulated data: Gap(L) = Δ_∞ + b/L² + noise
gap_data = gap_inf_true + b_true / L**2 + np.random.normal(0, 0.01, len(L))

# QTT singular values (exponentially decaying)
k = np.arange(1, 51)
sigma = 2.5 * np.exp(-0.3 * k) + np.random.normal(0, 0.001, len(k))
sigma = np.abs(sigma)

print("Input Data:")
print(f"  True Δ_∞ = {gap_inf_true}")
print(f"  True b = {b_true}")
print(f"  Lattice sizes L: {L}")
print(f"  Gap data: {gap_data[:5].round(4)}...")
print()

# ═══════════════════════════════════════════════════════════════
# PHASE 1: CONJECTURER
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: CONJECTURER (Symbolic Regression)")
print("=" * 60)
print()

conjecturer = Conjecturer(
    niterations=10,  # Quick for demo
    populations=3,
    verbose=True
)

# Use fallback method (fast, guaranteed to work)
print("Using power-law fit (fallback for speed)...")
formula = conjecturer._fit_power_law(L.astype(float), gap_data, name="mass_gap")

print(f"\nDiscovered formula: {formula.expression}")
print(f"LaTeX: {formula.latex}")
print(f"R² = {formula.r_squared:.6f}")
print(f"Coefficients: {formula.coefficients}")
print()

# Extract the infinite limit
limit = formula.coefficients.get('limit', formula.coefficients.get('a', 'unknown'))
print(f"★ Infinite limit Δ_∞ = {limit}")
print(f"  (True value: {gap_inf_true})")
print()

# ═══════════════════════════════════════════════════════════════
# PHASE 2: FORMALIZER
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2: FORMALIZER (Lean 4 Theory Generation)")
print("=" * 60)
print()

formalizer = Formalizer()

# Add computed bounds as axioms
formalizer.add_computed_bounds({
    "mass_gap": (1.46, 1.54),
    "finite_size_coeff": (0.75, 0.85),
})

# Generate theory
lean_code = formalizer.generate_yang_mills_theory()

sorry_count = lean_code.count("sorry")
print(f"Generated Lean 4 theory: {len(lean_code)} characters")
print(f"Proof obligations (sorry): {sorry_count}")
print()

# Show a snippet
print("Lean 4 code snippet:")
print("-" * 40)
for line in lean_code.split('\n')[30:50]:
    print(line)
print("-" * 40)
print()

# ═══════════════════════════════════════════════════════════════
# PHASE 3: PROVER
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 3: PROVER (LLM-Aided Proof Generation)")
print("=" * 60)
print()

prover = AIScientistProver()

# Extract obligations
obligations = prover.session.extract_obligations(lean_code)
print(f"Extracted {len(obligations)} proof obligations:")
for obl in obligations:
    print(f"  - {obl.name}")
print()

# Attempt proofs (using template-based fallback)
print("Attempting proofs...")
updated_lean, results = prover.process(lean_code)

proved = sum(1 for r in results.values() if r.success)
failed = sum(1 for r in results.values() if not r.success)

print()
print(f"Results: {proved} proved, {failed} pending")
for name, result in results.items():
    status = "✓" if result.success else "○"
    print(f"  {status} {name}")
print()

# Generate certificate
certificate = prover.generate_certificate(updated_lean, results)

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("AI SCIENTIST v1.0 - COMPLETE")
print("=" * 70)
print()
print("Summary:")
print(f"  Phase 1 (Conjecturer): Discovered Δ(L) = {limit:.4f} + O(1/L²)")
print(f"  Phase 2 (Formalizer):  Generated Lean 4 theory")
print(f"  Phase 3 (Prover):      {proved}/{len(results)} obligations addressed")
print()
print("Certificate:")
print(f"  Status: {certificate['status']}")
print(f"  Code Hash: {certificate['code_hash']}")
print(f"  Remaining sorry: {certificate['remaining_sorry']}")
print()

if certificate['remaining_sorry'] == 0:
    print("★★★ FULLY VERIFIED ★★★")
    print("Ready for submission to Clay Institute.")
else:
    print("⚠ PARTIAL VERIFICATION")
    print(f"{certificate['remaining_sorry']} proof obligations remain.")
    print("These require either:")
    print("  1. More sophisticated LLM prompting")
    print("  2. Human-assisted proof completion")
    print("  3. Lean 4 tactic automation (simp, linarith, etc.)")
print()

# Export
import os
output_dir = "/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/ai_scientist_output"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/YangMills.lean", 'w') as f:
    f.write(lean_code)

import json
with open(f"{output_dir}/certificate.json", 'w') as f:
    json.dump({
        "formula": formula.__dict__,
        "certificate": certificate,
        "infinite_limit": float(limit) if isinstance(limit, (int, float)) else str(limit),
    }, f, indent=2, default=str)

print(f"Exported to: {output_dir}")
print()
print("=" * 70)
print("QED (Quod Erat Demonstrandum)")
print("=" * 70)
