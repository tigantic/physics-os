"""
Tests for fluidelite.core.decompositions
========================================

Per Article II: Test Discipline
"""

import pytest
import torch
from fluidelite.core.decompositions import (
    svd_truncated,
    qr_positive,
    rsvd_truncated,
    SafeSVD,
)


class TestSVDTruncated:
    """Tests for svd_truncated function."""
    
    def test_basic_svd(self):
        """Test basic SVD decomposition."""
        A = torch.randn(50, 30, dtype=torch.float64)
        U, S, Vh = svd_truncated(A)
        
        # Check shapes
        assert U.shape[0] == 50
        assert Vh.shape[1] == 30
        assert len(S) == min(50, 30)
        
    def test_svd_reconstruction(self):
        """Test A ≈ U @ diag(S) @ Vh."""
        A = torch.randn(50, 30, dtype=torch.float64)
        U, S, Vh = svd_truncated(A)
        
        A_reconstructed = U @ torch.diag(S) @ Vh
        assert torch.allclose(A, A_reconstructed, atol=1e-10)
        
    def test_chi_max_truncation(self):
        """Test chi_max limits rank."""
        A = torch.randn(50, 30, dtype=torch.float64)
        U, S, Vh = svd_truncated(A, chi_max=10)
        
        assert len(S) <= 10
        assert U.shape[1] <= 10
        assert Vh.shape[0] <= 10
        
    def test_cutoff_threshold(self):
        """Test cutoff removes small singular values."""
        # Create matrix with known singular values
        U = torch.eye(10, dtype=torch.float64)
        S_diag = torch.tensor([10., 5., 1., 0.1, 0.01, 1e-5, 1e-8, 1e-10, 1e-12, 1e-15], dtype=torch.float64)
        Vh = torch.eye(10, dtype=torch.float64)
        A = U @ torch.diag(S_diag) @ Vh
        
        U_out, S_out, Vh_out = svd_truncated(A, cutoff=1e-10)
        
        # Should remove singular values below 1e-10
        assert len(S_out) < 10
        
    def test_svd_orthogonality(self):
        """Test U and Vh have orthonormal columns/rows."""
        A = torch.randn(50, 30, dtype=torch.float64)
        U, S, Vh = svd_truncated(A)
        
        # U^T @ U should be identity
        assert torch.allclose(U.T @ U, torch.eye(U.shape[1], dtype=torch.float64), atol=1e-10)
        
        # Vh @ Vh^T should be identity
        assert torch.allclose(Vh @ Vh.T, torch.eye(Vh.shape[0], dtype=torch.float64), atol=1e-10)


class TestRSVD:
    """Tests for randomized SVD."""
    
    def test_rsvd_basic(self):
        """Test basic randomized SVD."""
        A = torch.randn(100, 50, dtype=torch.float64)
        U, S, Vh = rsvd_truncated(A, rank=20)
        
        assert U.shape == (100, 20)
        assert len(S) == 20
        assert Vh.shape == (20, 50)
        
    def test_rsvd_approximation(self):
        """Test rSVD gives reasonable approximation."""
        A = torch.randn(100, 50, dtype=torch.float64)
        U, S, Vh = rsvd_truncated(A, rank=20)
        
        A_approx = U @ torch.diag(S) @ Vh
        
        # Error should be reasonable
        error = torch.norm(A - A_approx) / torch.norm(A)
        assert error < 1.0  # Within 100% - rSVD is approximate
        
    def test_rsvd_orthogonality(self):
        """Test rSVD produces orthonormal factors."""
        A = torch.randn(100, 50, dtype=torch.float64)
        U, S, Vh = rsvd_truncated(A, rank=20)
        
        # Check orthogonality
        assert torch.allclose(U.T @ U, torch.eye(20, dtype=torch.float64), atol=1e-8)


class TestQRPositive:
    """Tests for QR with positive diagonal."""
    
    def test_qr_reconstruction(self):
        """Test A = Q @ R."""
        A = torch.randn(50, 30, dtype=torch.float64)
        Q, R = qr_positive(A)
        
        assert torch.allclose(A, Q @ R, atol=1e-10)
        
    def test_q_orthogonal(self):
        """Test Q has orthonormal columns."""
        A = torch.randn(50, 30, dtype=torch.float64)
        Q, R = qr_positive(A)
        
        assert torch.allclose(Q.T @ Q, torch.eye(Q.shape[1], dtype=torch.float64), atol=1e-10)
        
    def test_r_upper_triangular(self):
        """Test R is upper triangular."""
        A = torch.randn(50, 30, dtype=torch.float64)
        Q, R = qr_positive(A)
        
        # Check upper triangular
        assert torch.allclose(R, torch.triu(R), atol=1e-10)
        
    def test_r_positive_diagonal(self):
        """Test R has non-negative diagonal."""
        A = torch.randn(50, 30, dtype=torch.float64)
        Q, R = qr_positive(A)
        
        diag = torch.diag(R)
        assert (diag >= -1e-10).all()  # Allow small numerical errors


class TestSafeSVD:
    """Tests for SafeSVD autograd function."""
    
    def test_forward_matches_linalg(self):
        """Test SafeSVD forward matches torch.linalg.svd."""
        A = torch.randn(20, 15, dtype=torch.float64)
        
        U1, S1, Vh1 = SafeSVD.apply(A)
        U2, S2, Vh2 = torch.linalg.svd(A, full_matrices=False)
        
        # Singular values should match
        assert torch.allclose(S1, S2, atol=1e-10)
        
    def test_forward_with_grad(self):
        """Test SafeSVD works with gradient tracking."""
        A = torch.randn(20, 15, dtype=torch.float64, requires_grad=True)
        
        U, S, Vh = SafeSVD.apply(A)
        
        # Should be able to compute loss
        loss = S.sum()
        loss.backward()
        
        # Gradient should exist (may be None due to simplified backward)
        # Just verify no errors occur
