#!/usr/bin/env python3
"""
Phase III Verification Test Suite

Verify all key findings from multi-plaquette Yang-Mills extension.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')


def test_single_plaquette_reference():
    """Test 1: Verify reference single plaquette gives Δ/g² = 1.5"""
    from yangmills.hamiltonian import SinglePlaquetteHamiltonian
    from yangmills.gauss import SinglePlaquetteGauss
    from scipy import sparse
    
    g = 1.0
    j_max = 0.5
    
    H_sys = SinglePlaquetteHamiltonian(j_max=j_max, g=g)
    H = H_sys.build_hamiltonian()
    gauss = SinglePlaquetteGauss(H_sys.hilbert)
    G2 = gauss.total_gauss_squared()
    
    H_dense = H.toarray() if sparse.issparse(H) else H
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    
    physical_E = []
    for i in range(len(eigenvalues)):
        psi = eigenvectors[:, i]
        g2_val = np.abs(psi.conj() @ G2 @ psi)
        if g2_val < 1e-6:
            physical_E.append(eigenvalues[i])
    
    gap = physical_E[1] - physical_E[0]
    gap_over_g2 = gap / g**2
    
    expected = 1.5
    passed = abs(gap_over_g2 - expected) < 0.01
    
    print(f"Test 1: Single Plaquette Reference")
    print(f"  Δ/g² = {gap_over_g2:.6f}")
    print(f"  Expected: {expected}")
    print(f"  Status: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return passed


def test_multi_plaquette_1x1_match():
    """Test 2: Verify 1×1 OBC multi-plaquette matches single plaquette"""
    from yangmills.efficient_subspace import EfficientMultiPlaquetteHamiltonian
    
    g = 1.0
    j_max = 0.5
    
    ham = EfficientMultiPlaquetteHamiltonian(1, 1, g, j_max, pbc=False)
    
    if ham.n_physical >= 2:
        H = ham.build_hamiltonian_dense()
        eigenvalues = np.linalg.eigvalsh(H)
        gap = eigenvalues[1] - eigenvalues[0]
        gap_over_g2 = gap / g**2
    else:
        gap_over_g2 = float('nan')
    
    expected = 1.5
    passed = abs(gap_over_g2 - expected) < 0.01
    
    print(f"\nTest 2: Multi-Plaquette 1×1 OBC = Single Plaquette")
    print(f"  Δ/g² = {gap_over_g2:.6f}")
    print(f"  Expected: {expected}")
    print(f"  Status: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return passed


def test_gap_stabilization():
    """Test 3: Verify gap stabilizes at 0.375 for L > 1"""
    from yangmills.efficient_subspace import EfficientMultiPlaquetteHamiltonian
    
    g = 1.0
    j_max = 0.5
    
    results = []
    for L in [2, 3]:
        ham = EfficientMultiPlaquetteHamiltonian(L, 1, g, j_max, pbc=False)
        
        if ham.n_physical >= 2:
            H = ham.build_hamiltonian_dense()
            eigenvalues = np.linalg.eigvalsh(H)
            gap = eigenvalues[1] - eigenvalues[0]
            gap_over_g2 = gap / g**2
            results.append(gap_over_g2)
    
    # Check all are close to 0.375
    expected = 0.375
    all_close = all(abs(r - expected) < 0.01 for r in results)
    
    print(f"\nTest 3: Gap Stabilization for L > 1")
    print(f"  2×1 OBC: Δ/g² = {results[0]:.6f}")
    print(f"  3×1 OBC: Δ/g² = {results[1]:.6f}")
    print(f"  Expected: {expected}")
    print(f"  Status: {'PASSED ✓' if all_close else 'FAILED ✗'}")
    
    return all_close


def test_coupling_independence():
    """Test 4: Verify Δ/g² is constant for different g values"""
    from yangmills.efficient_subspace import EfficientMultiPlaquetteHamiltonian
    
    j_max = 0.5
    g_values = [0.5, 1.0, 1.5, 2.0]
    
    # Test on 2×1 OBC
    results = []
    for g in g_values:
        ham = EfficientMultiPlaquetteHamiltonian(2, 1, g, j_max, pbc=False)
        
        if ham.n_physical >= 2:
            H = ham.build_hamiltonian_dense()
            eigenvalues = np.linalg.eigvalsh(H)
            gap = eigenvalues[1] - eigenvalues[0]
            gap_over_g2 = gap / g**2
            results.append(gap_over_g2)
    
    # Check all are close to each other
    mean_val = np.mean(results)
    all_consistent = all(abs(r - mean_val) < 0.01 for r in results)
    
    print(f"\nTest 4: Coupling Independence (2×1 OBC)")
    for i, g in enumerate(g_values):
        print(f"  g={g:.1f}: Δ/g² = {results[i]:.6f}")
    print(f"  Status: {'PASSED ✓' if all_consistent else 'FAILED ✗'}")
    
    return all_consistent


def test_ground_state_unique():
    """Test 5: Verify ground state is unique (non-degenerate)"""
    from yangmills.efficient_subspace import EfficientMultiPlaquetteHamiltonian
    
    g = 1.0
    j_max = 0.5
    
    configs = [(1, 1, False), (2, 1, False), (3, 1, False)]
    
    print(f"\nTest 5: Ground State Uniqueness")
    all_unique = True
    
    for Lx, Ly, pbc in configs:
        ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
        
        if ham.n_physical >= 2:
            H = ham.build_hamiltonian_dense()
            eigenvalues = np.linalg.eigvalsh(H)
            
            E0 = eigenvalues[0]
            degeneracy = np.sum(np.abs(eigenvalues - E0) < 1e-8)
            
            unique = (degeneracy == 1)
            all_unique = all_unique and unique
            
            name = f"{Lx}×{Ly} {'PBC' if pbc else 'OBC'}"
            status = "unique ✓" if unique else f"degenerate ({degeneracy}) ✗"
            print(f"  {name}: {status}")
    
    print(f"  Status: {'PASSED ✓' if all_unique else 'FAILED ✗'}")
    
    return all_unique


def run_all_tests():
    """Run all verification tests"""
    print("=" * 70)
    print("PHASE III VERIFICATION TEST SUITE")
    print("=" * 70)
    
    results = []
    
    results.append(("Single plaquette reference", test_single_plaquette_reference()))
    results.append(("Multi-plaquette 1×1 match", test_multi_plaquette_1x1_match()))
    results.append(("Gap stabilization", test_gap_stabilization()))
    results.append(("Coupling independence", test_coupling_independence()))
    results.append(("Ground state uniqueness", test_ground_state_unique()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASSED ✓" if result else "FAILED ✗"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Phase III verified!")
    else:
        print(f"\n✗ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
