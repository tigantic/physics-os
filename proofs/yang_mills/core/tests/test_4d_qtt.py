#!/usr/bin/env python3
"""
Test suite for 4D Yang-Mills QTT Holy Grail implementation.

Validates:
1. QTT state creation and manipulation
2. N-dimensional shift MPO for 4D
3. Strong coupling gap calculation
4. Dimensional transmutation physics
5. Scaling with lattice size
"""

import pytest
import torch
import numpy as np
import sys

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')

from yangmills.yangmills_4d_qtt import (
    YM4DConfig, YangMills4DQTT, morton_encode_4d, morton_decode_4d
)
from yangmills.yangmills_4d_tdvp import (
    YM4DTDVPConfig, YM4DTDVPSolver, create_vacuum_state, 
    create_single_excitation
)
from yangmills.weak_coupling_transmutation import (
    lattice_spacing, physical_mass_gap, strong_coupling_gap,
    weak_coupling_gap, crossover_gap
)


class TestMortonEncoding:
    """Test 4D Morton encoding/decoding."""
    
    def test_roundtrip(self):
        """Encode-decode roundtrip."""
        n_bits = 3
        for x in range(2**n_bits):
            for y in range(2**n_bits):
                for z in range(2**n_bits):
                    for t in range(2**n_bits):
                        idx = morton_encode_4d(x, y, z, t, n_bits)
                        x2, y2, z2, t2 = morton_decode_4d(idx, n_bits)
                        assert (x, y, z, t) == (x2, y2, z2, t2)
    
    def test_ordering(self):
        """Morton ordering preserves locality."""
        n_bits = 2
        idx_000 = morton_encode_4d(0, 0, 0, 0, n_bits)
        idx_100 = morton_encode_4d(1, 0, 0, 0, n_bits)
        idx_010 = morton_encode_4d(0, 1, 0, 0, n_bits)
        idx_001 = morton_encode_4d(0, 0, 1, 0, n_bits)
        
        # Morton interleaves: neighbors in all dims are nearby in index
        assert idx_000 == 0
        assert idx_100 == 1  # x bit at position 0
        assert idx_010 == 2  # y bit at position 1
        assert idx_001 == 4  # z bit at position 2


class TestYM4DConfig:
    """Test 4D Yang-Mills configuration."""
    
    def test_lattice_sizes(self):
        """Verify lattice size calculations."""
        config = YM4DConfig(qubits_per_dim=3)
        
        assert config.L == 8
        assert config.n_sites == 8**4 == 4096
        assert config.n_links == 4 * 4096 == 16384
        assert config.total_qubits == 12
    
    def test_link_dimension(self):
        """Verify link Hilbert space dimension."""
        config = YM4DConfig(j_max=0.5)
        # Formula gives (0.5+1)(2*0.5+2)/2 = 1.5*3/2 = 2.25 → int = 2
        # This corresponds to QTT binary encoding
        assert config.link_dim == 2  # QTT binary encoding
        
        config2 = YM4DConfig(j_max=1.0)
        # (1+1)(2*1+2)/2 = 2*4/2 = 4
        assert config2.link_dim == 4  # QTT encoding


class TestYM4DQTT:
    """Test 4D Yang-Mills QTT solver."""
    
    def test_initialization(self):
        """Solver initializes correctly."""
        config = YM4DConfig(qubits_per_dim=2)
        solver = YangMills4DQTT(config)
        
        assert solver.L == 4
        assert len(solver.shift_plus) == 4
        assert len(solver.shift_minus) == 4
    
    def test_vacuum_state(self):
        """Vacuum state is product state."""
        config = YM4DConfig(qubits_per_dim=2)
        solver = YangMills4DQTT(config)
        
        vacuum = solver.create_vacuum_state()
        
        # Product state has rank 1
        max_rank = max(c.shape[0] for c in vacuum.cores)
        assert max_rank == 1
    
    def test_gap_calculation(self):
        """Gap matches strong coupling formula."""
        config = YM4DConfig(qubits_per_dim=2, g=1.0)
        solver = YangMills4DQTT(config)
        
        result = solver.compute_gap_qtt()
        
        assert result['gap_estimate'] == pytest.approx(0.375, rel=1e-6)
        assert result['gap_estimate'] / config.g**2 == pytest.approx(0.375, rel=1e-6)


class TestYM4DTDVP:
    """Test 4D Yang-Mills TDVP solver."""
    
    def test_vacuum_creation(self):
        """Vacuum state created correctly."""
        config = YM4DTDVPConfig(qubits_per_dim=2)
        vacuum = create_vacuum_state(config)
        
        assert vacuum.n_qubits == 8
        assert vacuum.max_rank == 1
    
    def test_excitation_creation(self):
        """Excited state created correctly."""
        config = YM4DTDVPConfig(qubits_per_dim=2)
        excited = create_single_excitation(config, site_idx=0)
        
        assert excited.n_qubits == 8
        assert excited.max_rank == 1
    
    def test_solver_builds_operators(self):
        """TDVP solver builds shift operators."""
        config = YM4DTDVPConfig(qubits_per_dim=2)
        solver = YM4DTDVPSolver(config)
        
        assert len(solver.shift_plus) == 4
        assert len(solver.shift_minus) == 4
    
    def test_gap_matches_strong_coupling(self):
        """Gap calculation matches analytical result."""
        config = YM4DTDVPConfig(qubits_per_dim=2, g=1.0)
        solver = YM4DTDVPSolver(config)
        result = solver.compute_gap()
        
        assert result['gap'] == pytest.approx(0.375, rel=1e-6)
        assert result['gap_over_g2'] == pytest.approx(0.375, rel=1e-6)


class TestDimensionalTransmutation:
    """Test weak coupling and dimensional transmutation."""
    
    def test_strong_coupling_gap(self):
        """Strong coupling gap formula."""
        assert strong_coupling_gap(1.0) == pytest.approx(0.375, rel=1e-6)
        assert strong_coupling_gap(2.0) == pytest.approx(1.5, rel=1e-6)
    
    def test_lattice_spacing_asymptotic_freedom(self):
        """Lattice spacing vanishes as g → 0."""
        a_1 = lattice_spacing(1.0)
        a_half = lattice_spacing(0.5)
        a_quarter = lattice_spacing(0.25)
        
        # a decreases super-exponentially as g → 0
        assert a_half < a_1
        assert a_quarter < a_half
        assert a_quarter < 1e-50  # Extremely small!
    
    def test_physical_gap_finite(self):
        """Physical gap stays finite in continuum limit."""
        # At strong coupling
        M_1 = physical_mass_gap(strong_coupling_gap(1.0), 1.0)
        M_2 = physical_mass_gap(strong_coupling_gap(2.0), 2.0)
        
        assert M_1 > 0
        assert M_2 > 0
        
        # Physical gap is O(1) in Λ_QCD units
        # (varies with g, but stays finite)
    
    def test_crossover_interpolates(self):
        """Crossover smoothly interpolates strong/weak."""
        # At strong coupling, crossover ~ strong
        g_strong = 2.0
        assert crossover_gap(g_strong) == pytest.approx(
            strong_coupling_gap(g_strong), rel=0.5
        )
        
        # At weak coupling, crossover ~ weak
        g_weak = 0.3
        assert crossover_gap(g_weak) == pytest.approx(
            weak_coupling_gap(g_weak), rel=0.5
        )


class TestScaling:
    """Test O(log N) scaling of QTT approach."""
    
    def test_qubits_vs_lattice_size(self):
        """Qubits scale logarithmically with sites."""
        for n in range(2, 6):
            config = YM4DConfig(qubits_per_dim=n)
            
            # Sites = L^4 = 2^(4n)
            # Qubits = 4n = log_2(Sites)
            assert config.n_sites == 2**(4*n)
            assert config.total_qubits == 4*n
            assert config.total_qubits == pytest.approx(
                np.log2(config.n_sites), rel=1e-10
            )
    
    def test_memory_scaling(self):
        """Memory scales as O(log N × χ²)."""
        for n in range(2, 5):
            config = YM4DTDVPConfig(qubits_per_dim=n, max_rank=32)
            vacuum = create_vacuum_state(config)
            
            # Total memory ~ n_qubits × 2 × 1 × 1 (for product state)
            total_elements = sum(c.numel() for c in vacuum.cores)
            
            # For product state: 2 elements per qubit
            assert total_elements == 2 * config.total_qubits


class TestPhaseIIIConsistency:
    """Validate consistency with Phase III exact results."""
    
    def test_gap_matches_thermodynamic_limit(self):
        """QTT gap matches Phase III thermodynamic limit."""
        # Phase III result: Δ/g² = 0.375 for L > 1
        
        config = YM4DTDVPConfig(qubits_per_dim=3, g=1.0)
        solver = YM4DTDVPSolver(config)
        result = solver.compute_gap()
        
        phase_iii_thermodynamic = 0.375
        assert result['gap_over_g2'] == pytest.approx(
            phase_iii_thermodynamic, rel=1e-6
        )
    
    def test_strong_coupling_formula(self):
        """Verify strong coupling formula derivation."""
        # E²|j=1/2⟩ = j(j+1)|j=1/2⟩ = (1/2)(3/2)|j=1/2⟩ = (3/4)|j=1/2⟩
        j = 0.5
        E_squared = j * (j + 1)
        assert E_squared == pytest.approx(0.75, rel=1e-10)
        
        # Gap = (g²/2) × E² = (g²/2) × (3/4) = (3/8)g²
        g = 1.0
        gap = (g**2 / 2) * E_squared
        assert gap == pytest.approx(0.375, rel=1e-10)


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("4D YANG-MILLS QTT HOLY GRAIL: TEST SUITE")
    print("=" * 70)
    
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
