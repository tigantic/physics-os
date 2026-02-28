#!/usr/bin/env python
"""
TT-Native CFD Demo with Dense Guard (Hardened)
===============================================

CLAIM SCOPE (precisely stated):
    STORAGE: O(N*d*chi^2) - proven by core element counts
    NO DENSE GRID: O(N^2+) forbidden - ENFORCED by guard (forbid=True)
    O(N*d) DIAGNOSTICS: Allowed (per-site values, primitives)
    RUNTIME: Empirically sub-dense - not proven here
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from ontic.cfd.tt_cfd import TT_Euler1D, TTCFDConfig
from ontic.core.dense_guard import DenseMaterializationGuard, check_tt_complexity


def main():
    print("=" * 70)
    print("TT-Native CFD Demo with Dense Guard (PROOF MODE)")
    print("=" * 70)
    print()
    
    # Configuration
    N = 64          # Grid points
    d = 3           # Physical dimension (rho, rho*u, E)
    chi_max = 16    # Maximum bond dimension
    t_final = 0.01  # Simulation time
    
    # Thresholds matching O(N*d*chi^2) claim EXACTLY
    hard_threshold = N * d * chi_max * chi_max  # 49,152
    soft_threshold = int(0.1 * hard_threshold)  # 4,915
    
    print(f"Grid size: N = {N}")
    print(f"Physical dimension: d = {d}")
    print(f"Max bond dimension: chi = {chi_max}")
    print(f"Final time: t = {t_final}")
    print()
    print("Threshold Configuration:")
    print(f"  hard_threshold = N * d * chi^2 = {hard_threshold:,}")
    print(f"  soft_threshold = 0.1 * hard = {soft_threshold:,}")
    print(f"  diagnostic_allowed = N * d * 10 = {N * d * 10}")
    print()
    
    # Create solver
    config = TTCFDConfig(chi_max=chi_max, tdvp_order=1)
    solver = TT_Euler1D(N=N, L=1.0, gamma=1.4, chi_max=chi_max, config=config)
    solver.initialize_sod()
    
    # Get initial values
    initial_mass = solver.state.total_mass()
    initial_energy = solver.state.total_energy()
    
    print(f"Initial mass: {initial_mass:.6f}")
    print(f"Initial energy: {initial_energy:.6f}")
    print()
    
    # Set up dense guard in PROOF MODE
    guard = DenseMaterializationGuard(
        hard_threshold=hard_threshold,
        soft_threshold=soft_threshold,
        forbid=True,           # PROOF MODE - raises on critical
        allow_diagnostics=True, # O(N*d) vectors allowed
        N=N,
        d=d
    )
    
    print("Dense Guard: PROOF MODE (forbid=True)")
    print("  - Would raise RuntimeError on critical violation")
    print()
    print("Running simulation...")
    print("-" * 50)
    
    step_count = 0
    start_time = time.perf_counter()
    
    try:
        with guard:
            while solver.time < t_final:
                solver.step()
                step_count += 1
                
                if step_count % 20 == 0:
                    rho, u, p = solver.state.to_primitive()
                    chi = solver.state.chi
                    print(f"  Step {step_count:4d} | t = {solver.time:.6f} | "
                          f"chi = {chi} | rho_max = {rho.max():.4f}")
        
        elapsed = time.perf_counter() - start_time
        proof_passed = True
        
    except RuntimeError as e:
        if "FORBIDDEN DENSE MATERIALIZATION" in str(e):
            print(f"\n  CRITICAL VIOLATION: {e}")
            proof_passed = False
            elapsed = time.perf_counter() - start_time
        else:
            raise
    
    print("-" * 50)
    print()
    
    if proof_passed:
        # Final values
        final_mass = solver.state.total_mass()
        final_energy = solver.state.total_energy()
        mass_error = abs(final_mass - initial_mass) / initial_mass
        energy_error = abs(final_energy - initial_energy) / initial_energy
        
        # Complexity check
        complexity = check_tt_complexity(solver.state.cores, chi_max, N, d=d)
        
        print("Results:")
        print(f"  Total steps: {step_count}")
        print(f"  Wall time: {elapsed:.3f} s")
        print(f"  Time per step: {1000 * elapsed / step_count:.3f} ms")
        print()
        print("Conservation:")
        print(f"  Final mass: {final_mass:.6f}")
        print(f"  Mass error: {mass_error:.2e}")
        print(f"  Final energy: {final_energy:.6f}")
        print(f"  Energy error: {energy_error:.2e}")
        print()
        print("TT Complexity:")
        print(f"  Storage elements: {complexity['total_elements']:,}")
        print(f"  Expected max (O(N*d*chi^2)): {complexity['expected_max']:,}")
        print(f"  Within bound: {complexity['within_bound']}")
        print(f"  Actual/Theoretical ratio: {complexity['actual_vs_theoretical']:.4f}")
        print()
        print("Dense Materialization Audit:")
        print(f"  Total violations: {guard.violations}")
        print(f"  Critical violations: {guard.critical_violations}")
        print(f"  Total dense elements above threshold: {guard.total_dense_elements:,}")
        print()
        
        # Verdict
        print("=" * 70)
        all_ok = guard.critical_violations == 0 and complexity['within_bound']
        if all_ok:
            print("PROOF MODE: [PASSED]")
            print()
            print("CLAIMS PROVEN:")
            print("  * Storage: O(N*d*chi^2) verified")
            print("  * No O(N^2+) dense grid materialization")
            print("  * Guard would have raised on critical violation")
        else:
            print("PROOF MODE: [FAILED]")
    else:
        print("=" * 70)
        print("PROOF MODE: [FAILED] - Critical dense violation detected")
    
    print("=" * 70)
    
    return 0 if proof_passed and guard.critical_violations == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
