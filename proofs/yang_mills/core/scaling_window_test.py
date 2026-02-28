#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCALING WINDOW STRESS TEST                                ║
║                                                                              ║
║     Hunt for the Crossover: Strong Coupling → Asymptotic Freedom             ║
╚══════════════════════════════════════════════════════════════════════════════╝

HYPOTHESIS:
===========
At weak coupling (g → 0), the glueball "swells" and entanglement explodes.
If Δ/g² stays flat and S stays at ln(2), we're stuck in strong-coupling vacuum.
If Δ/g² curves up and S increases, we've found the SCALING WINDOW!

EXPERIMENT:
===========
1. Push to g = 0.1, 0.05, 0.01 (deep weak coupling)
2. Unlock bond dimension to 64, 128, 256
3. Monitor:
   - Does S increase beyond ln(2)?
   - Does the solver saturate the available ranks?
   - Does Δ/g² deviate from 1.5 (or 0.375)?

Author: TiganticLabz Yang-Mills Project
Date: 2026-01-16
"""

import numpy as np
import torch
import time
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')

from yangmills.tensor_network.dmrg import compute_gap_tensor_network


def run_scaling_window_test():
    """
    Stress test to find the crossover region.
    
    We're looking for:
    1. S > ln(2) → entanglement explosion
    2. Rank saturation → solver needs more resources
    3. Δ/g² deviation → departure from strong coupling
    """
    
    print("=" * 70)
    print("SCALING WINDOW STRESS TEST")
    print("Hunting for Crossover: Confined Flux → Asymptotic Freedom")
    print("=" * 70)
    
    print("""
    CRITICAL QUESTION:
    ==================
    At weak coupling (g → 0), the correlation length ξ grows.
    The glueball "swells" and should become highly entangled.
    
    Current observation: S ≈ ln(2) ≈ 0.69 (constant)
    This suggests: Product state / valence bond ground state
    
    If we increase bond dimension and decrease g:
    - S increases → We've entered the scaling window!
    - S stays flat → Solver stuck in strong-coupling minimum
    """)
    
    # Test matrix: coupling × bond dimension
    g_values = [1.0, 0.5, 0.2, 0.1, 0.05]
    chi_values = [32, 64, 128]
    
    results = []
    
    for chi in chi_values:
        print(f"\n{'='*60}")
        print(f"BOND DIMENSION χ = {chi}")
        print(f"{'='*60}")
        
        for g in g_values:
            print(f"\n--- g = {g}, χ = {chi} ---")
            
            try:
                start = time.time()
                result = compute_gap_tensor_network(
                    g=g,
                    j_max=1.0,
                    bond_dim=chi,
                    verbose=False
                )
                elapsed = time.time() - start
                
                result['chi_max'] = chi
                result['time'] = elapsed
                results.append(result)
                
                # Check for anomalies
                S = result['entropy']
                gap_ratio = result['gap_over_g2']
                chi_used = result['bond_dim_used']
                
                # Flag interesting behavior
                flags = []
                if S > 0.75:  # Above ln(2)
                    flags.append("S↑")
                if chi_used >= chi * 0.9:  # Rank saturation
                    flags.append("χ-SAT")
                if abs(gap_ratio - 1.5) > 0.1:  # Gap deviation
                    flags.append("Δ-DEV")
                
                flag_str = " ".join(flags) if flags else "normal"
                
                print(f"  Δ/g² = {gap_ratio:.4f}, S = {S:.4f}, χ_used = {chi_used}, flags: {flag_str}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'g': g, 'chi_max': chi, 'error': str(e)
                })
    
    # Summary table
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    
    print(f"\n{'g':>8} {'χ_max':>8} {'χ_used':>8} {'Δ/g²':>10} {'S':>8} {'S/ln2':>8} {'Status'}")
    print("-" * 70)
    
    ln2 = np.log(2)
    
    for r in results:
        if 'error' in r:
            print(f"{r['g']:>8.4f} {r['chi_max']:>8} {'ERROR':>8}")
            continue
            
        g = r['g']
        chi_max = r['chi_max']
        chi_used = r['bond_dim_used']
        gap_ratio = r['gap_over_g2']
        S = r['entropy']
        S_ratio = S / ln2
        
        # Determine status
        if chi_used >= chi_max * 0.9:
            status = "⚠ RANK SATURATED"
        elif S > 0.8:
            status = "🔥 ENTANGLEMENT UP"
        elif abs(gap_ratio - 1.5) > 0.2:
            status = "⚡ GAP DEVIATION"
        else:
            status = "✓ Strong coupling"
        
        print(f"{g:>8.4f} {chi_max:>8} {chi_used:>8} {gap_ratio:>10.4f} {S:>8.4f} {S_ratio:>8.2f} {status}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check if S increased with chi at weak coupling
    weak_coupling = [r for r in results if r.get('g', 1) <= 0.1 and 'error' not in r]
    
    if weak_coupling:
        S_values = [r['entropy'] for r in weak_coupling]
        chi_values_used = [r['bond_dim_used'] for r in weak_coupling]
        
        print(f"\nAt weak coupling (g ≤ 0.1):")
        print(f"  Entropy range: {min(S_values):.4f} - {max(S_values):.4f}")
        print(f"  χ used range: {min(chi_values_used)} - {max(chi_values_used)}")
        
        if max(S_values) > 0.75:
            print("\n  🔥 SCALING WINDOW DETECTED!")
            print("  Entanglement exceeds ln(2) - glueball is swelling!")
        elif max(chi_values_used) >= max(chi_values) * 0.9:
            print("\n  ⚠ RANK SATURATION!")
            print("  Solver needs higher bond dimension to capture physics.")
            print("  Try χ = 256 or 512.")
        else:
            print("\n  Ground state appears to be in strong-coupling regime.")
            print("  Either:")
            print("    1. Need even weaker coupling (g = 0.01)")
            print("    2. Need larger lattice (more plaquettes)")
            print("    3. Strong-coupling vacuum extends further than expected")
    
    return results


def deep_weak_coupling_test():
    """
    Push to extremely weak coupling with maximum resources.
    """
    
    print("\n" + "=" * 70)
    print("DEEP WEAK COUPLING TEST")
    print("=" * 70)
    
    print("""
    Going to g = 0.05 and g = 0.01 with χ = 128
    
    At these couplings:
    - Correlation length ξ ~ 1/g² → very large
    - Glueball mass M ~ Λ_QCD × exp(-8π²/(b₀ g²)) 
    - If we see S → large and χ saturates, we're in the scaling window!
    """)
    
    extreme_g = [0.05, 0.02, 0.01]
    chi = 128
    
    for g in extreme_g:
        print(f"\n--- EXTREME TEST: g = {g}, χ = {chi} ---")
        
        try:
            result = compute_gap_tensor_network(
                g=g,
                j_max=1.5,  # Higher j_max for weak coupling
                bond_dim=chi,
                verbose=True
            )
            
            print(f"\n  Critical metrics:")
            print(f"    Δ/g² = {result['gap_over_g2']:.4f} (expect 1.5 if strong coupling)")
            print(f"    S = {result['entropy']:.4f} (expect > ln(2) = 0.69 if scaling)")
            print(f"    χ_used = {result['bond_dim_used']} (expect {chi} if saturated)")
            
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    results = run_scaling_window_test()
    deep_weak_coupling_test()
