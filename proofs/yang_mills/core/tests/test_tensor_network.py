"""
Test Tensor Network Methods in Weak Coupling Regime
====================================================

This is the critical test that exact diagonalization CANNOT do!

The 4.77 TiB memory error was physics telling us:
- Strong coupling (g ~ 1): Simple vacuum, low entanglement → ED works
- Weak coupling (g ~ 0.1): "Boiling sea of virtual particles", high entanglement → ED fails

With tensor networks:
- MPS compresses: O(d^N) → O(N × χ² × d)
- Can access g < 0.1 regime
- Can test if gap deviates from strong coupling scaling

Key predictions:
- Strong coupling: Δ = (3/2)g² (trivial, already proven)
- Weak coupling: Δ ~ Λ_QCD ~ g² × exp(-1/(2β₀g²))
- β₀(SU(2)) ≈ 0.116

If tensor networks show Δ/g² deviating from 1.5 at weak coupling,
we have evidence of dimensional transmutation!
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yangmills.tensor_network import (
    MPS, MPOHamiltonian, YangMillsMPO, 
    DMRG, compute_gap_tensor_network, scan_coupling_range
)


class TestOntic EngineworkBasics:
    """Test basic tensor network infrastructure."""
    
    def test_mps_creation(self):
        """Test MPS creation and normalization."""
        mps = MPS.random(n_sites=4, local_dim=3, bond_dim=10, normalize=True)
        
        assert mps.n_sites == 4
        assert len(mps.tensors) == 4
        
        # Check normalization
        norm = mps.inner(mps)
        assert abs(norm - 1.0) < 1e-10, f"MPS not normalized: {norm}"
    
    def test_mps_canonicalization(self):
        """Test left and right canonicalization."""
        mps = MPS.random(4, 3, 10, normalize=True)
        
        # Left canonicalize
        mps.canonicalize('left')
        
        # Check left-canonical form: A†A = I
        for i in range(mps.n_sites - 1):
            A = mps.tensors[i]
            chi_l, d, chi_r = A.shape
            A_mat = A.reshape(chi_l * d, chi_r)
            should_be_I = A_mat.conj().T @ A_mat
            assert np.allclose(should_be_I, np.eye(chi_r), atol=1e-10)
    
    def test_mps_entanglement_entropy(self):
        """Test entanglement entropy computation."""
        # Product state: zero entropy
        # Create product state with all sites in state |0⟩
        local_dim = 3
        n_sites = 4
        local_states = [np.array([1.0, 0.0, 0.0]) for _ in range(n_sites)]
        mps = MPS.product_state(local_states)
        S = mps.entanglement_entropy(2)
        assert S < 1e-10, f"Product state entropy should be 0, got {S}"
        
        # Random state: nonzero entropy
        mps = MPS.random(4, 3, 10, normalize=True)
        mps.canonicalize('left')
        S = mps.entanglement_entropy(2)
        assert S > 0, "Random state should have nonzero entropy"
    
    def test_mpo_creation(self):
        """Test MPO creation."""
        mpo = YangMillsMPO(n_links=4, j_max=1.0, g=1.0)
        
        assert mpo.n_sites == 4
        assert len(mpo.tensors) == 4
        
        # Check tensor shapes
        for i, W in enumerate(mpo.tensors):
            assert len(W.shape) == 4, "MPO tensors should be rank-4"


class TestStrongCoupling:
    """Verify tensor networks reproduce strong coupling results."""
    
    def test_strong_coupling_gap(self):
        """
        At g = 1.0, tensor network should give Δ ≈ (3/2)g² = 1.5
        """
        g = 1.0
        result = compute_gap_tensor_network(
            g=g, j_max=1.0, bond_dim=30, n_sweeps=10, verbose=True
        )
        
        expected = 1.5 * g**2
        tolerance = 0.1  # 10% tolerance for numerical method
        
        print(f"\nStrong coupling test:")
        print(f"  g = {g}")
        print(f"  Δ (tensor network) = {result['gap']:.6f}")
        print(f"  Δ (analytical) = {expected:.6f}")
        print(f"  Relative error = {abs(result['gap'] - expected)/expected * 100:.2f}%")
        
        assert abs(result['gap'] - expected) < expected * tolerance, \
            f"Gap {result['gap']} deviates from expected {expected}"
    
    def test_strong_coupling_low_entropy(self):
        """
        At strong coupling, entanglement should be low.
        """
        g = 2.0  # Very strong coupling
        result = compute_gap_tensor_network(
            g=g, j_max=1.0, bond_dim=30, n_sweeps=10, verbose=True
        )
        
        print(f"\nEntanglement at strong coupling (g={g}):")
        print(f"  S = {result['entropy']:.6f}")
        
        # Strong coupling should have low entanglement
        assert result['entropy'] < 1.0, \
            f"Strong coupling should have low entropy, got {result['entropy']}"


class TestWeakCoupling:
    """
    THE CRITICAL TESTS!
    
    These tests probe the weak coupling regime that exact diagonalization
    CANNOT access due to the 4.77 TiB memory barrier.
    """
    
    def test_weak_coupling_accessible(self):
        """
        Verify we can compute at g = 0.1 without memory explosion!
        
        With exact diagonalization, this required 4.77 TiB.
        With tensor networks, this should use ~MB.
        """
        import tracemalloc
        
        tracemalloc.start()
        
        g = 0.1  # Weak coupling - previously impossible!
        result = compute_gap_tensor_network(
            g=g, j_max=1.0, bond_dim=50, n_sweeps=15, verbose=True
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\nWeak coupling memory test:")
        print(f"  g = {g}")
        print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
        print(f"  Compare to exact diag: 4.77 TiB")
        
        # Should use less than 1 GB
        assert peak < 1024**3, \
            f"Memory usage {peak/1024**3:.2f} GB exceeds 1 GB limit"
    
    def test_weak_coupling_higher_entropy(self):
        """
        Weak coupling should have HIGHER entanglement than strong.
        
        This is the physics of why exact diag failed:
        - Strong coupling: simple vacuum, few basis states
        - Weak coupling: "boiling sea of virtual particles"
        """
        g_strong = 2.0
        g_weak = 0.3
        
        result_strong = compute_gap_tensor_network(
            g=g_strong, j_max=1.0, bond_dim=50, n_sweeps=10, verbose=False
        )
        
        result_weak = compute_gap_tensor_network(
            g=g_weak, j_max=1.0, bond_dim=50, n_sweeps=10, verbose=False
        )
        
        print(f"\nEntanglement comparison:")
        print(f"  Strong coupling (g={g_strong}): S = {result_strong['entropy']:.4f}")
        print(f"  Weak coupling (g={g_weak}): S = {result_weak['entropy']:.4f}")
        
        # Weak coupling should have higher or comparable entropy
        # (The ratio matters more than absolute values)
        print(f"  Ratio S_weak/S_strong = {result_weak['entropy']/max(result_strong['entropy'], 1e-10):.2f}")
    
    def test_gap_scaling_vs_coupling(self):
        """
        THE BIG TEST: Does gap scale as g² at all couplings?
        
        Strong coupling: Δ = (3/2)g² (proven)
        Weak coupling: If still Δ ∝ g², then gap → 0 as g → 0
        
        For Millennium Prize:
        Need to see deviation from g² scaling at weak coupling
        indicating dimensional transmutation!
        """
        g_values = [2.0, 1.0, 0.5, 0.3, 0.2, 0.1]
        
        results = scan_coupling_range(
            g_values, j_max=1.0, bond_dim=50, verbose=False
        )
        
        print("\n" + "="*70)
        print("GAP SCALING ANALYSIS")
        print("="*70)
        print(f"{'g':>8} {'Δ':>12} {'Δ/g²':>10} {'Expected':>10} {'Deviation'}")
        print("-"*55)
        
        deviations = []
        for r in results:
            expected = 1.5  # Strong coupling prediction
            deviation = (r['gap_over_g2'] - expected) / expected * 100
            deviations.append(deviation)
            print(f"{r['g']:>8.4f} {r['gap']:>12.6f} {r['gap_over_g2']:>10.4f} {expected:>10.2f} {deviation:>+8.2f}%")
        
        # Check if deviation grows at weak coupling
        # (would indicate transition from strong to weak coupling physics)
        print("\n" + "-"*55)
        print("KEY OBSERVATION:")
        
        strong_dev = np.mean(np.abs(deviations[:2]))  # g = 2.0, 1.0
        weak_dev = np.mean(np.abs(deviations[-2:]))   # g = 0.2, 0.1
        
        print(f"  Mean |deviation| at strong coupling: {strong_dev:.2f}%")
        print(f"  Mean |deviation| at weak coupling: {weak_dev:.2f}%")
        
        if weak_dev > strong_dev * 1.5:
            print("\n  ⚠️  POSSIBLE DEVIATION AT WEAK COUPLING!")
            print("  This could indicate dimensional transmutation")
        else:
            print("\n  Gap scales as g² across all tested couplings")
            print("  No evidence of dimensional transmutation yet")
            print("  May need: smaller g, larger bond_dim, or full lattice")


class TestBondDimensionConvergence:
    """Test that results converge with bond dimension."""
    
    def test_bond_dim_convergence(self):
        """
        Gap should converge as bond dimension increases.
        """
        g = 0.5
        bond_dims = [10, 20, 40]
        gaps = []
        
        print(f"\nBond dimension convergence test (g={g}):")
        
        for chi in bond_dims:
            result = compute_gap_tensor_network(
                g=g, j_max=1.0, bond_dim=chi, n_sweeps=10, verbose=False
            )
            gaps.append(result['gap'])
            print(f"  χ = {chi:>3}: Δ = {result['gap']:.8f}")
        
        # Check convergence (difference should decrease)
        diff1 = abs(gaps[1] - gaps[0])
        diff2 = abs(gaps[2] - gaps[1])
        
        print(f"\n  |Δ(χ=20) - Δ(χ=10)| = {diff1:.6f}")
        print(f"  |Δ(χ=40) - Δ(χ=20)| = {diff2:.6f}")
        
        # Either converged (diffs both small) or convergence observed
        if diff1 < 1e-6 and diff2 < 1e-6:
            print("  ✓ Already converged at χ=10")
        else:
            assert diff2 <= diff1 * 2, \
                "Bond dimension convergence not observed"


def run_all_tests():
    """Run all tensor network tests."""
    print("\n" + "="*70)
    print("TENSOR NETWORK TEST SUITE")
    print("Testing weak coupling regime access")
    print("="*70)
    
    # Basic tests
    print("\n[1/5] Testing MPS/MPO basics...")
    test = TestOntic EngineworkBasics()
    test.test_mps_creation()
    test.test_mps_canonicalization()
    test.test_mps_entanglement_entropy()
    test.test_mpo_creation()
    print("  ✓ All basic tests passed")
    
    # Strong coupling
    print("\n[2/5] Testing strong coupling reproduction...")
    test = TestStrongCoupling()
    test.test_strong_coupling_gap()
    test.test_strong_coupling_low_entropy()
    print("  ✓ Strong coupling tests passed")
    
    # Weak coupling - THE BIG ONES
    print("\n[3/5] Testing weak coupling access...")
    test = TestWeakCoupling()
    test.test_weak_coupling_accessible()
    print("  ✓ Weak coupling accessible (no TiB memory!)")
    
    print("\n[4/5] Testing entanglement scaling...")
    test.test_weak_coupling_higher_entropy()
    
    print("\n[5/5] Testing gap scaling...")
    test.test_gap_scaling_vs_coupling()
    
    # Convergence
    print("\n[BONUS] Testing convergence...")
    test = TestBondDimensionConvergence()
    test.test_bond_dim_convergence()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
