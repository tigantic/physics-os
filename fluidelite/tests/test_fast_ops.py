"""
Tests for fluidelite.core.fast_ops

Constitutional Compliance:
    - Article II.2.2: 80% test coverage required
"""

import pytest
import torch
from fluidelite.core.fast_ops import vectorized_mpo_apply, vectorized_mps_add
from fluidelite.core.mps import MPS


class TestVectorizedMPOApply:
    """Test vectorized_mpo_apply function."""
    
    def test_basic_apply(self):
        """Test basic MPO application."""
        L, chi, d, D = 8, 4, 2, 3
        
        # Create stacked MPS tensors
        mps_cores = torch.randn(L, chi, d, chi, dtype=torch.float64)
        
        # Create stacked MPO tensors (L, D, d_out, d_in, D)
        mpo_cores = torch.randn(L, D, d, d, D, dtype=torch.float64)
        
        result = vectorized_mpo_apply(mps_cores, mpo_cores)
        
        assert result.shape == (L, chi * D, d, chi * D)
    
    def test_output_dtype(self):
        """Test output dtype matches input."""
        L, chi, d, D = 6, 2, 2, 2
        
        mps_cores = torch.randn(L, chi, d, chi, dtype=torch.float64)
        mpo_cores = torch.randn(L, D, d, d, D, dtype=torch.float64)
        
        result = vectorized_mpo_apply(mps_cores, mpo_cores)
        
        assert result.dtype == torch.float64
    
    def test_identity_mpo(self):
        """Test that identity MPO preserves MPS (up to reshape)."""
        L, chi, d = 4, 2, 2
        
        mps_cores = torch.randn(L, chi, d, chi, dtype=torch.float64)
        
        # Create identity MPO (D=1)
        mpo_cores = torch.zeros(L, 1, d, d, 1, dtype=torch.float64)
        for i in range(L):
            for j in range(d):
                mpo_cores[i, 0, j, j, 0] = 1.0
        
        result = vectorized_mpo_apply(mps_cores, mpo_cores)
        
        # Result should have same data but reshaped (chi*1=chi)
        assert result.shape == (L, chi, d, chi)


class TestVectorizedMPSAdd:
    """Test vectorized_mps_add function."""
    
    def test_basic_add(self):
        """Test basic MPS addition."""
        L, d = 8, 2
        chi_a, chi_b = 4, 6
        
        mps_a = torch.randn(L, chi_a, d, chi_a, dtype=torch.float64)
        mps_b = torch.randn(L, chi_b, d, chi_b, dtype=torch.float64)
        
        result = vectorized_mps_add(mps_a, mps_b)
        
        # Direct sum: bond dimensions add
        expected_chi = chi_a + chi_b
        assert result.shape == (L, expected_chi, d, expected_chi)
    
    def test_add_same_dimension(self):
        """Test adding MPS with same bond dimension."""
        L, chi, d = 6, 4, 2
        
        mps_a = torch.randn(L, chi, d, chi, dtype=torch.float64)
        mps_b = torch.randn(L, chi, d, chi, dtype=torch.float64)
        
        result = vectorized_mps_add(mps_a, mps_b)
        
        assert result.shape == (L, 2*chi, d, 2*chi)
    
    def test_add_block_diagonal(self):
        """Test that result is block diagonal."""
        L, d = 4, 2
        chi_a, chi_b = 2, 3
        
        mps_a = torch.ones(L, chi_a, d, chi_a, dtype=torch.float64)
        mps_b = 2 * torch.ones(L, chi_b, d, chi_b, dtype=torch.float64)
        
        result = vectorized_mps_add(mps_a, mps_b)
        
        # Check block structure for interior sites
        for i in range(1, L-1):
            # Top-left block (chi_a x chi_a) should be from mps_a
            assert torch.allclose(result[i, :chi_a, :, :chi_a], mps_a[i])
            
            # Bottom-right block (chi_b x chi_b) should be from mps_b
            assert torch.allclose(result[i, chi_a:, :, chi_a:], mps_b[i])
            
            # Off-diagonal blocks should be zero
            assert torch.allclose(result[i, :chi_a, :, chi_a:], 
                                  torch.zeros(chi_a, d, chi_b, dtype=torch.float64))
    
    def test_dtype_preservation(self):
        """Test output dtype matches input."""
        L, chi, d = 4, 2, 2
        
        mps_a = torch.randn(L, chi, d, chi, dtype=torch.float64)
        mps_b = torch.randn(L, chi, d, chi, dtype=torch.float64)
        
        result = vectorized_mps_add(mps_a, mps_b)
        
        assert result.dtype == torch.float64
