#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     WEAK COUPLING SCALING TEST                               ║
║                                                                              ║
║         Testing if mass gap survives the continuum limit (g → 0)            ║
║                         ★ CRITICAL VALIDATION ★                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

The Strong Coupling result Δ = (3/2)g² is mathematically correct but:
  - In continuum limit: a → 0 requires g → 0 (asymptotic freedom)
  - If Δ ~ g², then Δ → 0 as g → 0 — NO MASS GAP!

The REAL solution requires dimensional transmutation:
  Δ_physical ~ Λ_QCD ~ exp(-1/(2β₀g²))

This test explores the weak coupling regime (g < 0.1) to see if:
  1. The gap follows polynomial scaling (FAIL - no continuum mass gap)
  2. The gap follows exponential scaling (POSSIBLE - dimensional transmutation)
  3. Something else happens (TBD)

Author: TiganticLabz Yang-Mills Project
Date: 2026-01-15
"""

import sys
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import time
from datetime import datetime

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')

from yangmills.hamiltonian import SinglePlaquetteHamiltonian
from yangmills.gauss import SinglePlaquetteGauss


def compute_physical_gap(g: float, j_max: float = 1.5) -> dict:
    """
    Compute the physical (gauge-invariant) mass gap for coupling g.
    
    Returns gap in both lattice units and attempts to extract physical scale.
    """
    H_sys = SinglePlaquetteHamiltonian(j_max=j_max, g=g)
    H = H_sys.build_hamiltonian()
    gauss = SinglePlaquetteGauss(H_sys.hilbert)
    G2 = gauss.total_gauss_squared()
    
    # Full diagonalization for small systems
    H_dense = H.toarray() if sparse.issparse(H) else H
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    
    # Find physical states (G² = 0)
    physical_E = []
    for i in range(len(eigenvalues)):
        psi = eigenvectors[:, i]
        g2_val = np.abs(psi.conj() @ G2 @ psi)
        if g2_val < 1e-6:
            physical_E.append(eigenvalues[i])
    
    if len(physical_E) >= 2:
        E0 = physical_E[0]
        E1 = physical_E[1]
        gap = E1 - E0
    else:
        gap = np.nan
        E0 = physical_E[0] if len(physical_E) > 0 else np.nan
        E1 = np.nan
    
    return {
        'g': g,
        'E0': E0,
        'E1': E1,
        'gap_lattice': gap,
        'gap_over_g2': gap / (g**2) if gap > 0 and g > 0 else np.nan,
        'n_physical': len(physical_E),
        'hilbert_dim': H_sys.hilbert.total_dim ** 4
    }


def test_strong_coupling_regime():
    """Test 1: Verify strong coupling result Δ = (3/2)g²."""
    print("\n" + "=" * 70)
    print("TEST 1: STRONG COUPLING REGIME (g ≥ 1)")
    print("Expected: Δ/g² = 1.5 (constant)")
    print("=" * 70)
    
    couplings = [1.0, 1.5, 2.0, 3.0, 4.0]
    results = []
    
    print(f"\n{'g':>8} {'Δ_lattice':>12} {'Δ/g²':>10} {'Status':>12}")
    print("-" * 50)
    
    for g in couplings:
        r = compute_physical_gap(g, j_max=0.5)
        results.append(r)
        
        status = "✓ EXPECTED" if abs(r['gap_over_g2'] - 1.5) < 0.01 else "✗ UNEXPECTED"
        print(f"{g:>8.2f} {r['gap_lattice']:>12.6f} {r['gap_over_g2']:>10.4f} {status:>12}")
    
    # Verify polynomial scaling
    gaps = [r['gap_lattice'] for r in results]
    g_vals = [r['g'] for r in results]
    
    # Fit log(Δ) vs log(g) to get power law
    log_g = np.log(g_vals)
    log_gap = np.log(gaps)
    slope, intercept = np.polyfit(log_g, log_gap, 1)
    
    print(f"\nPower law fit: Δ ~ g^{slope:.4f}")
    print(f"Expected: Δ ~ g^2.0000")
    
    return results


def test_weak_coupling_regime():
    """Test 2: Explore weak coupling (g < 1) — the critical test."""
    print("\n" + "=" * 70)
    print("TEST 2: WEAK COUPLING REGIME (g < 1) — CRITICAL TEST")
    print("Question: Does Δ/g² remain constant, or does something else happen?")
    print("=" * 70)
    
    # Use j_max=0.5 to keep memory manageable
    # In weak coupling, this truncation may be insufficient, but let's see
    j_max = 0.5
    
    couplings = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
    results = []
    
    print(f"\n{'g':>8} {'Δ_lattice':>12} {'Δ/g²':>10} {'log(Δ)':>10} {'1/g²':>10}")
    print("-" * 60)
    
    for g in couplings:
        r = compute_physical_gap(g, j_max=j_max)
        results.append(r)
        
        log_gap = np.log(r['gap_lattice']) if r['gap_lattice'] > 0 else np.nan
        inv_g2 = 1.0 / (g**2)
        
        print(f"{g:>8.3f} {r['gap_lattice']:>12.6f} {r['gap_over_g2']:>10.4f} {log_gap:>10.4f} {inv_g2:>10.2f}")
    
    return results


def test_exponential_vs_polynomial():
    """Test 3: Fit data to both polynomial and exponential models."""
    print("\n" + "=" * 70)
    print("TEST 3: POLYNOMIAL vs EXPONENTIAL SCALING")
    print("Polynomial: Δ = A * g^n  (strong coupling)")
    print("Exponential: Δ = B * exp(-C/g²)  (dimensional transmutation)")
    print("=" * 70)
    
    # Collect data across full range with j_max=0.5 (manageable)
    all_couplings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    results = []
    
    for g in all_couplings:
        r = compute_physical_gap(g, j_max=0.5)
        if not np.isnan(r['gap_lattice']) and r['gap_lattice'] > 0:
            results.append(r)
    
    g_vals = np.array([r['g'] for r in results])
    gaps = np.array([r['gap_lattice'] for r in results])
    
    # Model 1: Polynomial Δ = A * g^n
    log_g = np.log(g_vals)
    log_gap = np.log(gaps)
    poly_coeffs = np.polyfit(log_g, log_gap, 1)
    n_poly = poly_coeffs[0]
    A_poly = np.exp(poly_coeffs[1])
    
    gap_poly_fit = A_poly * g_vals**n_poly
    residual_poly = np.sum((gaps - gap_poly_fit)**2)
    
    print(f"\nPolynomial fit: Δ = {A_poly:.4f} × g^{n_poly:.4f}")
    print(f"  Residual: {residual_poly:.6e}")
    
    # Model 2: Exponential Δ = B * exp(-C/g²)
    # Take log: log(Δ) = log(B) - C/g²
    # Linear regression: log(Δ) vs 1/g²
    inv_g2 = 1.0 / (g_vals**2)
    try:
        exp_coeffs = np.polyfit(inv_g2, log_gap, 1)
        C_exp = -exp_coeffs[0]
        B_exp = np.exp(exp_coeffs[1])
        
        gap_exp_fit = B_exp * np.exp(-C_exp / g_vals**2)
        residual_exp = np.sum((gaps - gap_exp_fit)**2)
        
        print(f"\nExponential fit: Δ = {B_exp:.4f} × exp(-{C_exp:.4f}/g²)")
        print(f"  Residual: {residual_exp:.6e}")
        
        # Compare models
        print("\n" + "-" * 50)
        if residual_poly < residual_exp:
            print("RESULT: Polynomial fit is BETTER")
            print("→ Gap follows strong coupling scaling Δ ~ g²")
            print("→ In continuum limit (g→0): Δ → 0")
            print("→ This is NOT the Millennium Prize solution!")
            winner = "POLYNOMIAL"
        else:
            print("RESULT: Exponential fit is BETTER")
            print("→ Possible dimensional transmutation")
            print("→ Need to verify with β-function")
            winner = "EXPONENTIAL"
            
    except Exception as e:
        print(f"\nExponential fit failed: {e}")
        winner = "POLYNOMIAL"
        residual_exp = np.inf
    
    return {
        'g_vals': g_vals,
        'gaps': gaps,
        'poly': {'A': A_poly, 'n': n_poly, 'residual': residual_poly},
        'exp': {'B': B_exp if 'B_exp' in dir() else None, 
                'C': C_exp if 'C_exp' in dir() else None, 
                'residual': residual_exp if 'residual_exp' in dir() else np.inf},
        'winner': winner
    }


def test_continuum_limit_problem():
    """Test 4: Demonstrate the continuum limit problem explicitly."""
    print("\n" + "=" * 70)
    print("TEST 4: THE CONTINUUM LIMIT PROBLEM")
    print("=" * 70)
    
    print("\nThe relationship between lattice spacing a and coupling g:")
    print("  a(g) = (1/Λ) × exp(1/(2β₀g²))  [asymptotic freedom]")
    print()
    print("For SU(2), β₀ = 11/3 × (2) / (16π²) ≈ 0.116")
    print()
    
    beta_0 = 11 * 2 / (3 * 16 * np.pi**2)  # SU(2)
    
    print(f"{'g':>8} {'a/Λ⁻¹':>15} {'Δ_lattice':>12} {'Δ_physical':>12}")
    print("-" * 55)
    
    for g in [1.0, 0.5, 0.3, 0.2, 0.1]:
        r = compute_physical_gap(g, j_max=0.5)
        
        # Lattice spacing in units of 1/Λ
        a_over_Lambda_inv = np.exp(1 / (2 * beta_0 * g**2))
        
        # Gap in lattice units
        gap_lattice = r['gap_lattice']
        
        # Gap in physical units: Δ_phys = Δ_lattice / a
        # If Δ_lattice ~ g² and a ~ exp(1/(2β₀g²)), then
        # Δ_phys ~ g² × exp(-1/(2β₀g²)) → 0 as g → 0
        if not np.isnan(gap_lattice):
            gap_physical = gap_lattice / a_over_Lambda_inv
        else:
            gap_physical = np.nan
        
        print(f"{g:>8.3f} {a_over_Lambda_inv:>15.2e} {gap_lattice:>12.6f} {gap_physical:>12.2e}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("  If Δ_lattice ~ g² (our strong coupling result)")
    print("  And a ~ exp(1/(2β₀g²)) (asymptotic freedom)")
    print("  Then Δ_physical ~ g² × exp(-1/(2β₀g²)) → 0 as g → 0")
    print()
    print("  THE MASS GAP VANISHES IN THE CONTINUUM LIMIT!")
    print("=" * 70)


def main():
    """Run all weak coupling tests."""
    
    print("╔" + "═" * 68 + "╗")
    print("║" + "WEAK COUPLING SCALING TEST".center(68) + "║")
    print("║" + "Testing if mass gap survives continuum limit".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Execution Time: {datetime.now().isoformat()}")
    print()
    print("CRITICAL QUESTION:")
    print("  Does Δ follow polynomial scaling (Δ ~ g²) → NO continuum mass gap")
    print("  Or exponential scaling (Δ ~ exp(-c/g²)) → POSSIBLE continuum mass gap")
    
    start_time = time.time()
    
    # Run tests
    strong_results = test_strong_coupling_regime()
    weak_results = test_weak_coupling_regime()
    scaling_results = test_exponential_vs_polynomial()
    test_continuum_limit_problem()
    
    # Final verdict
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "FINAL VERDICT".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    if scaling_results['winner'] == "POLYNOMIAL":
        print()
        print("  ╔═══════════════════════════════════════════════════════════════╗")
        print("  ║  STRONG COUPLING RESULT: Δ = (3/2)g² is CORRECT              ║")
        print("  ║                                                               ║")
        print("  ║  BUT: This is NOT the Millennium Prize solution!             ║")
        print("  ║                                                               ║")
        print("  ║  The gap vanishes in the continuum limit (g → 0).            ║")
        print("  ║                                                               ║")
        print("  ║  To solve the prize, we need:                                ║")
        print("  ║    1. Non-perturbative methods (not strong coupling)         ║")
        print("  ║    2. Evidence of dimensional transmutation                  ║")
        print("  ║    3. Δ_physical ~ Λ_QCD remaining finite as a → 0           ║")
        print("  ╚═══════════════════════════════════════════════════════════════╝")
    else:
        print()
        print("  Exponential scaling detected — further investigation needed.")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return scaling_results


if __name__ == "__main__":
    results = main()
