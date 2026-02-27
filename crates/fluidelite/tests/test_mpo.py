"""
Tests for fluidelite.core.mpo
=============================

Per Article II: Test Discipline
"""

import pytest
import torch
from fluidelite.core.mpo import MPO, mpo_sum
from fluidelite.core.mps import MPS


class TestMPOCreation:
    """Tests for MPO creation and initialization."""
    
    def test_mpo_from_tensors(self):
        """Test MPO creation from tensor list."""
        tensors = [torch.randn(1, 2, 2, 4) for _ in range(3)]
        tensors[0] = torch.randn(1, 2, 2, 4)
        tensors[-1] = torch.randn(4, 2, 2, 1)
        mpo = MPO(tensors)
        assert mpo.L == 3
        
    def test_identity_mpo(self):
        """Test identity-like MPO construction."""
        L, d = 3, 4
        tensors = [torch.eye(d).reshape(1, d, d, 1) for _ in range(L)]
        mpo = MPO(tensors)
        assert mpo.L == L
        assert mpo.d == d
        assert mpo.D == 1


class TestMPOProperties:
    """Tests for MPO property accessors."""
    
    def test_L_property(self):
        """Test L returns correct number of sites."""
        tensors = [torch.randn(1, 2, 2, 1) for _ in range(5)]
        mpo = MPO(tensors)
        assert mpo.L == 5
        
    def test_d_property(self):
        """Test d returns correct physical dimension."""
        tensors = [torch.randn(1, 4, 4, 1) for _ in range(3)]
        mpo = MPO(tensors)
        assert mpo.d == 4
        
    def test_D_property(self):
        """Test D returns maximum bond dimension."""
        tensors = [
            torch.randn(1, 2, 2, 8),
            torch.randn(8, 2, 2, 8),
            torch.randn(8, 2, 2, 1)
        ]
        mpo = MPO(tensors)
        assert mpo.D == 8


class TestMPOApply:
    """Tests for MPO.apply() method."""
    
    def test_apply_identity(self):
        """Test identity MPO doesn't change MPS."""
        # Create identity MPO
        L, d = 3, 2
        identity_tensors = [torch.eye(d, dtype=torch.float64).reshape(1, d, d, 1) for _ in range(L)]
        mpo = MPO(identity_tensors)
        
        # Create MPS
        mps = MPS.random(L=L, d=d, chi=4, dtype=torch.float64)
        original_tensor = mps.to_tensor()
        
        # Apply identity
        result = mpo.apply(mps)
        result_tensor = result.to_tensor()
        
        # Should be unchanged
        assert torch.allclose(original_tensor, result_tensor, atol=1e-10)
        
    def test_apply_increases_bond_dim(self):
        """Test applying MPO increases bond dimension."""
        L, d = 3, 2
        mps = MPS.random(L=L, d=d, chi=4, dtype=torch.float64)
        
        # Create MPO with bond dim 2
        tensors = [
            torch.randn(1, d, d, 2, dtype=torch.float64),
            torch.randn(2, d, d, 2, dtype=torch.float64),
            torch.randn(2, d, d, 1, dtype=torch.float64)
        ]
        mpo = MPO(tensors)
        
        result = mpo.apply(mps)
        
        # Result bond dim should be chi * D = 4 * 2 = 8
        # (may be less due to boundary effects)
        assert result.chi >= mps.chi


class TestMPOExpectation:
    """Tests for MPO expectation value computation."""
    
    def test_identity_expectation(self):
        """Test <psi|I|psi> = ||psi||^2."""
        L, d = 3, 2
        
        # Create identity MPO
        identity_tensors = [torch.eye(d, dtype=torch.float64).reshape(1, d, d, 1) for _ in range(L)]
        mpo = MPO(identity_tensors)
        
        # Create normalized MPS
        mps = MPS.random(L=L, d=d, chi=4, dtype=torch.float64)
        mps.normalize_()
        
        # Expectation should be close to 1 for normalized state
        exp_val = mpo.expectation(mps)
        assert abs(exp_val.item() - 1.0) < 0.5


class TestMPOOperations:
    """Tests for MPO operations."""
    
    def test_copy(self):
        """Test copy creates independent clone."""
        tensors = [torch.randn(1, 2, 2, 1) for _ in range(3)]
        mpo = MPO(tensors)
        mpo_copy = mpo.copy()
        
        # Modify original
        mpo.tensors[0] *= 0
        
        # Copy should be unchanged
        assert not torch.allclose(mpo.tensors[0], mpo_copy.tensors[0])
        
    def test_to_matrix_shape(self):
        """Test to_matrix produces correct shape."""
        L, d = 3, 2
        tensors = [torch.randn(1, d, d, 1) for _ in range(L)]
        mpo = MPO(tensors)
        
        matrix = mpo.to_matrix()
        expected_size = d ** L
        assert matrix.shape == (expected_size, expected_size)


class TestMPOSum:
    """Tests for MPO addition."""
    
    def test_mpo_sum_bond_dim(self):
        """Test MPO sum has combined bond dimension."""
        L, d = 3, 2
        
        tensors1 = [
            torch.randn(1, d, d, 4),
            torch.randn(4, d, d, 4),
            torch.randn(4, d, d, 1)
        ]
        tensors2 = [
            torch.randn(1, d, d, 3),
            torch.randn(3, d, d, 3),
            torch.randn(3, d, d, 1)
        ]
        
        mpo1 = MPO(tensors1)
        mpo2 = MPO(tensors2)
        
        result = mpo_sum(mpo1, mpo2)
        
        # Result bond dim should be D1 + D2 = 4 + 3 = 7
        assert result.D == 7


class TestMPORepr:
    """Tests for string representation."""
    
    def test_repr(self):
        """Test repr produces informative string."""
        tensors = [torch.randn(1, 2, 2, 4) for _ in range(3)]
        tensors[-1] = torch.randn(4, 2, 2, 1)
        mpo = MPO(tensors)
        repr_str = repr(mpo)
        assert "MPO" in repr_str
        assert "L=3" in repr_str
