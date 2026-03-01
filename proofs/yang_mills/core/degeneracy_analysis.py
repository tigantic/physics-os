"""
Investigation of Ground State Degeneracy in Multi-Plaquette Yang-Mills

Key finding: 2×2 OBC has gap = 0 (degenerate ground states)
This needs investigation:
1. What is the degeneracy structure?
2. Is it due to global gauge symmetry?
3. Does it persist with higher j_max?
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict
import sys

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')

from yangmills.efficient_subspace import (
    EfficientMultiPlaquetteHamiltonian,
    EfficientPhysicalSubspace,
    enumerate_link_states
)


def analyze_spectrum(Lx: int, Ly: int, g: float, j_max: float, pbc: bool):
    """Analyze full spectrum to understand degeneracy structure."""
    print(f"\n{'='*60}")
    print(f"SPECTRUM ANALYSIS: {Lx}×{Ly} {'PBC' if pbc else 'OBC'}")
    print(f"{'='*60}")
    
    ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
    
    if ham.n_physical < 2:
        print("Not enough physical states")
        return None
    
    # Build and diagonalize
    H = ham.build_hamiltonian_dense()
    eigenvalues = np.linalg.eigvalsh(H)
    
    # Count degeneracies
    unique_E = []
    degeneracy = []
    tol = 1e-8
    
    i = 0
    while i < len(eigenvalues):
        E = eigenvalues[i]
        count = 1
        while i + count < len(eigenvalues) and abs(eigenvalues[i + count] - E) < tol:
            count += 1
        unique_E.append(E)
        degeneracy.append(count)
        i += count
    
    print(f"\nSpectrum (first 10 levels):")
    print(f"{'Level':<8} {'Energy':<15} {'Degeneracy':<12} {'E/g²':<12}")
    print("-" * 50)
    
    for i in range(min(10, len(unique_E))):
        print(f"{i:<8} {unique_E[i]:<15.6f} {degeneracy[i]:<12} {unique_E[i]/g**2:<12.6f}")
    
    # Gap structure
    if len(unique_E) >= 2:
        gap = unique_E[1] - unique_E[0]
        print(f"\nGround state energy: {unique_E[0]:.6f}")
        print(f"Ground state degeneracy: {degeneracy[0]}")
        print(f"First excited energy: {unique_E[1]:.6f}")
        print(f"First excited degeneracy: {degeneracy[1]}")
        print(f"Gap: {gap:.6f}")
        print(f"Gap/g²: {gap/g**2:.6f}")
    
    return {
        'eigenvalues': unique_E,
        'degeneracies': degeneracy,
        'n_physical': ham.n_physical
    }


def analyze_ground_state_structure(Lx: int, Ly: int, g: float, j_max: float, pbc: bool):
    """Analyze the structure of degenerate ground states."""
    print(f"\n{'='*60}")
    print(f"GROUND STATE STRUCTURE: {Lx}×{Ly} {'PBC' if pbc else 'OBC'}")
    print(f"{'='*60}")
    
    ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
    H = ham.build_hamiltonian_dense()
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Find ground state manifold
    E0 = eigenvalues[0]
    gs_indices = np.where(np.abs(eigenvalues - E0) < 1e-8)[0]
    gs_degeneracy = len(gs_indices)
    
    print(f"Ground state degeneracy: {gs_degeneracy}")
    
    # Analyze what physical states contribute
    link_states = ham.link_states
    
    for gs_idx in range(min(3, gs_degeneracy)):
        psi = eigenvectors[:, gs_indices[gs_idx]]
        
        print(f"\nGround state #{gs_idx}:")
        
        # Find dominant components
        probs = np.abs(psi)**2
        top_indices = np.argsort(probs)[::-1][:5]
        
        print(f"  Top 5 components:")
        for idx in top_indices:
            if probs[idx] > 0.01:
                state_int = ham.physical_states[idx]
                state_tuple = ham.subspace.state_to_tuple(state_int)
                
                js = [link_states[s].j for s in state_tuple]
                ms = [link_states[s].m for s in state_tuple]
                
                print(f"    prob={probs[idx]:.4f}: j={js}, m={ms}")


def compare_truncations():
    """Compare gap structure with different truncations."""
    print("\n" + "=" * 70)
    print("TRUNCATION COMPARISON")
    print("=" * 70)
    
    g = 1.0
    Lx, Ly = 1, 1
    pbc = False
    
    print(f"\nConfig: {Lx}×{Ly} OBC, g = {g}")
    print(f"\n{'j_max':<8} {'n_phys':<10} {'Δ/g²':<12} {'Expected':<12} {'Match':<8}")
    print("-" * 55)
    
    for j_max in [0.5, 1.0, 1.5]:
        ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
        
        if ham.n_physical >= 2:
            H = ham.build_hamiltonian_dense()
            eigenvalues = np.linalg.eigvalsh(H)
            gap = eigenvalues[1] - eigenvalues[0]
            gap_over_g2 = gap / g**2
            
            expected = 1.5
            match = '✓' if abs(gap_over_g2 - expected) < 0.01 else '✗'
            
            print(f"{j_max:<8.1f} {ham.n_physical:<10} {gap_over_g2:<12.6f} {expected:<12.1f} {match:<8}")


def investigate_degeneracy_origin():
    """
    Investigate whether degeneracy comes from:
    1. Global gauge transformations
    2. Topological sectors (for PBC)
    3. Physical excited states at same energy
    """
    print("\n" + "=" * 70)
    print("DEGENERACY ORIGIN INVESTIGATION")
    print("=" * 70)
    
    g = 1.0
    j_max = 0.5
    
    # For 2×2 OBC, we found degeneracy. Let's understand why.
    ham = EfficientMultiPlaquetteHamiltonian(2, 2, g, j_max, pbc=False)
    
    H = ham.build_hamiltonian_dense()
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    E0 = eigenvalues[0]
    gs_mask = np.abs(eigenvalues - E0) < 1e-8
    gs_count = np.sum(gs_mask)
    
    print(f"\n2×2 OBC Analysis:")
    print(f"  Physical states: {ham.n_physical}")
    print(f"  Ground state degeneracy: {gs_count}")
    
    # Look at the j-values in ground states
    print(f"\n  Ground state compositions (j-values on each link):")
    
    gs_indices = np.where(gs_mask)[0]
    
    # Collect all (j) configurations that appear in ground states
    j_configs = set()
    
    for gs_idx in gs_indices[:5]:
        psi = eigenvectors[:, gs_idx]
        
        # Find dominant component
        dominant_idx = np.argmax(np.abs(psi)**2)
        state_int = ham.physical_states[dominant_idx]
        state_tuple = ham.subspace.state_to_tuple(state_int)
        
        js = tuple(ham.link_states[s].j for s in state_tuple)
        j_configs.add(js)
    
    print(f"  Distinct j-configurations: {len(j_configs)}")
    for config in list(j_configs)[:5]:
        print(f"    {config}")
    
    # Key insight: Ground state should have all j=0 (lowest E²)
    # Degeneracy comes from different m-configurations that satisfy Gauss law
    
    print(f"\n  Analysis: With j=0 on all links:")
    print(f"    E² = 0 for each link → Total E² = 0")
    print(f"    Any m-configuration satisfies j=0 → m=0 uniquely")
    print(f"    So ground state should be unique!")
    
    # Check actual ground state
    psi0 = eigenvectors[:, 0]
    dominant_idx = np.argmax(np.abs(psi0)**2)
    state_int = ham.physical_states[dominant_idx]
    state_tuple = ham.subspace.state_to_tuple(state_int)
    
    js = [ham.link_states[s].j for s in state_tuple]
    ms = [ham.link_states[s].m for s in state_tuple]
    
    print(f"\n  Actual dominant ground state component:")
    print(f"    j-values: {js}")
    print(f"    m-values: {ms}")
    
    # If we see j≠0 in ground state, it means Gauss constraint forces non-zero j
    all_zero = all(j == 0 for j in js)
    print(f"    All j=0: {all_zero}")
    
    if not all_zero:
        print(f"\n  INSIGHT: Ground state forced to have j≠0 by Gauss law!")
        print(f"           This creates degeneracy from m-quantum numbers.")


def main():
    g = 1.0
    j_max = 0.5
    
    # Analyze spectrum for different lattice sizes
    analyze_spectrum(1, 1, g, j_max, pbc=False)
    analyze_spectrum(2, 1, g, j_max, pbc=False)
    analyze_spectrum(2, 2, g, j_max, pbc=False)
    
    # Ground state structure
    analyze_ground_state_structure(2, 2, g, j_max, pbc=False)
    
    # Compare truncations
    compare_truncations()
    
    # Investigate degeneracy
    investigate_degeneracy_origin()


if __name__ == "__main__":
    main()
