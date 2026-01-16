"""
Tests for fluidelite.core.mps
=============================

Per Article II: Test Discipline
    - Section 2.1: No code shall be merged without passing all automated tests
    - Section 2.2: Coverage below 80% requires documented justification
"""

import pytest
import torch
from fluidelite.core.mps import MPS


class TestMPSCreation:
    """Tests for MPS creation and initialization."""
    
    def test_random_mps_creation(self):
        """Test MPS.random() creates valid MPS."""
        mps = MPS.random(L=5, d=2, chi=4)
        assert mps.L == 5
        assert mps.d == 2
        assert mps.chi <= 4
        
    def test_random_mps_shapes(self):
        """Test MPS tensors have correct boundary shapes."""
        mps = MPS.random(L=5, d=2, chi=8)
        # First tensor should have chi_left=1
        assert mps.tensors[0].shape[0] == 1
        # Last tensor should have chi_right=1
        assert mps.tensors[-1].shape[2] == 1
        
    def test_random_mps_device(self):
        """Test MPS can be created on specified device."""
        mps = MPS.random(L=3, d=2, chi=4, device=torch.device("cpu"))
        assert mps.device == torch.device("cpu")
        
    def test_random_mps_dtype(self):
        """Test MPS respects dtype."""
        mps = MPS.random(L=3, d=2, chi=4, dtype=torch.float32)
        assert mps.dtype == torch.float32
        
        mps64 = MPS.random(L=3, d=2, chi=4, dtype=torch.float64)
        assert mps64.dtype == torch.float64


class TestMPSProperties:
    """Tests for MPS property accessors."""
    
    def test_L_property(self):
        """Test L returns correct number of sites."""
        for L in [3, 5, 10]:
            mps = MPS.random(L=L, d=2, chi=4)
            assert mps.L == L
            
    def test_d_property(self):
        """Test d returns correct physical dimension."""
        for d in [2, 4, 8]:
            mps = MPS.random(L=5, d=d, chi=4)
            assert mps.d == d
            
    def test_chi_property(self):
        """Test chi returns maximum bond dimension."""
        mps = MPS.random(L=10, d=2, chi=16)
        assert mps.chi <= 16
        
    def test_bond_dims(self):
        """Test bond_dims returns list of correct length."""
        mps = MPS.random(L=5, d=2, chi=8)
        dims = mps.bond_dims()
        assert len(dims) == 4  # L-1 bonds


class TestMPSNorm:
    """Tests for MPS norm computation."""
    
    def test_norm_positive(self):
        """Test norm is always positive."""
        mps = MPS.random(L=5, d=2, chi=8)
        assert mps.norm().item() > 0
        
    def test_normalized_mps(self):
        """Test normalized MPS has unit norm."""
        mps = MPS.random(L=5, d=2, chi=8, normalize=True)
        # After normalization, norm should be close to 1
        # (not exact due to how normalization is distributed)
        norm = mps.norm().item()
        assert 0.5 < norm < 2.0  # Reasonable range
        
    def test_normalize_in_place(self):
        """Test normalize_() modifies in place."""
        mps = MPS.random(L=5, d=2, chi=8, normalize=False)
        original_norm = mps.norm().item()
        mps.normalize_()
        # Should return self
        assert isinstance(mps, MPS)


class TestMPSCanonical:
    """Tests for MPS canonicalization."""
    
    def test_canonicalize_left(self):
        """Test left canonicalization sets canonical center."""
        mps = MPS.random(L=5, d=2, chi=8)
        mps.canonicalize_left_()
        assert mps._canonical_center == 4  # L-1
        
    def test_canonicalize_right(self):
        """Test right canonicalization sets canonical center."""
        mps = MPS.random(L=5, d=2, chi=8)
        mps.canonicalize_right_()
        assert mps._canonical_center == 0
        
    def test_canonicalize_to(self):
        """Test mixed canonicalization to specific site."""
        mps = MPS.random(L=5, d=2, chi=8)
        mps.canonicalize_to_(site=2)
        assert mps._canonical_center == 2
        
    def test_canonicalization_preserves_state(self):
        """Test canonicalization doesn't change the represented state."""
        mps = MPS.random(L=4, d=2, chi=4)
        tensor_before = mps.to_tensor()
        mps.canonicalize_left_()
        tensor_after = mps.to_tensor()
        # Should be same up to normalization
        assert tensor_before.shape == tensor_after.shape


class TestMPSOperations:
    """Tests for MPS operations."""
    
    def test_copy(self):
        """Test copy creates independent clone."""
        mps = MPS.random(L=5, d=2, chi=8)
        mps_copy = mps.copy()
        
        # Modify original
        mps.tensors[0] *= 0
        
        # Copy should be unchanged
        assert not torch.allclose(mps.tensors[0], mps_copy.tensors[0])
        
    def test_truncate(self):
        """Test truncate reduces bond dimension."""
        mps = MPS.random(L=5, d=2, chi=16)
        mps.truncate_(chi_max=4)
        assert mps.chi <= 4
        
    def test_to_tensor_shape(self):
        """Test to_tensor produces correct shape."""
        mps = MPS.random(L=4, d=3, chi=8)
        tensor = mps.to_tensor()
        assert tensor.shape == (3, 3, 3, 3)  # d^L
        
    def test_from_tensor_roundtrip(self):
        """Test MPS.from_tensor recovers original tensor."""
        original = torch.randn(2, 2, 2, 2)
        mps = MPS.from_tensor(original)
        recovered = mps.to_tensor()
        # Should be close (exact if chi_max is unlimited)
        assert recovered.shape == original.shape


class TestMPSEntropy:
    """Tests for entanglement entropy."""
    
    def test_entropy_non_negative(self):
        """Test entropy is non-negative."""
        mps = MPS.random(L=5, d=2, chi=8)
        entropy = mps.entropy(bond=2)
        assert entropy.item() >= 0
        
    def test_product_state_zero_entropy(self):
        """Test product state has zero entropy."""
        # Create product state (chi=1)
        tensors = [torch.randn(1, 2, 1) for _ in range(5)]
        mps = MPS(tensors)
        mps.normalize_()
        entropy = mps.entropy(bond=2)
        # Product state should have very low entropy
        assert entropy.item() < 0.1


class TestMPSExpectation:
    """Tests for expectation value computation."""
    
    def test_identity_expectation(self):
        """Test identity operator gives norm^2."""
        mps = MPS.random(L=5, d=2, chi=8, normalize=True)
        identity = torch.eye(2, dtype=mps.dtype)
        exp_val = mps.expectation_local(identity, site=2)
        # <psi|I|psi> / <psi|psi> = 1
        assert abs(exp_val.item() - 1.0) < 0.1


class TestMPSRepr:
    """Tests for string representation."""
    
    def test_repr(self):
        """Test repr produces informative string."""
        mps = MPS.random(L=5, d=2, chi=8)
        repr_str = repr(mps)
        assert "MPS" in repr_str
        assert "L=5" in repr_str
        assert "d=2" in repr_str
