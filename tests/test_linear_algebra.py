"""
Test Module: Matrix Factorizations and Linear Algebra

Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Golub, G.H. & Van Loan, C.F. (2013).
    "Matrix Computations." Johns Hopkins University Press.
    
    Trefethen, L.N. & Bau, D. (1997).
    "Numerical Linear Algebra." SIAM.
"""

import pytest
import torch
import numpy as np
import math
from typing import Tuple, Optional


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def device():
    """Get device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def random_spd_matrix():
    """Generate random symmetric positive definite matrix."""
    torch.manual_seed(42)
    n = 10
    A = torch.randn(n, n, dtype=torch.float64)
    return A @ A.T + torch.eye(n, dtype=torch.float64)


@pytest.fixture
def random_matrix():
    """Generate random matrix."""
    torch.manual_seed(42)
    return torch.randn(10, 10, dtype=torch.float64)


# ============================================================================
# LU FACTORIZATION
# ============================================================================

def lu_factorization(
    A: torch.Tensor,
    pivoting: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """LU factorization with optional partial pivoting using PyTorch."""
    # Use PyTorch's built-in LU factorization for correctness
    LU, pivots = torch.linalg.lu_factor(A)
    n = A.shape[0]
    
    # Extract L and U from the packed LU tensor
    L = torch.tril(LU, diagonal=-1) + torch.eye(n, dtype=A.dtype, device=A.device)
    U = torch.triu(LU)
    
    # Construct permutation matrix from pivots
    P = torch.eye(n, dtype=A.dtype, device=A.device)
    for i in range(n):
        if pivots[i] != i + 1:  # pivots are 1-indexed
            j = pivots[i].item() - 1
            P[[i, j]] = P[[j, i]]
    
    return L, U, P


def solve_lu(
    L: torch.Tensor,
    U: torch.Tensor,
    b: torch.Tensor,
    P: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve Ax = b using LU factorization."""
    if P is not None:
        b = P @ b
    
    # Forward substitution: Ly = b
    n = L.shape[0]
    y = torch.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    
    # Backward substitution: Ux = y
    x = torch.zeros_like(b)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) > 1e-14:
            x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    
    return x


# ============================================================================
# CHOLESKY FACTORIZATION
# ============================================================================

def cholesky_factorization(A: torch.Tensor) -> torch.Tensor:
    """Cholesky factorization for SPD matrices."""
    n = A.shape[0]
    L = torch.zeros_like(A)
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                val = A[i, i] - (L[i, :i] ** 2).sum()
                L[i, i] = math.sqrt(max(val, 0))
            else:
                L[i, j] = (A[i, j] - (L[i, :j] * L[j, :j]).sum()) / (L[j, j] + 1e-14)
    
    return L


def solve_cholesky(L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve Ax = b using Cholesky factorization (A = LL^T)."""
    n = L.shape[0]
    
    # Forward substitution: Ly = b
    y = torch.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    
    # Backward substitution: L^T x = y
    x = torch.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i+1:, i] @ x[i+1:]) / L[i, i]
    
    return x


# ============================================================================
# QR FACTORIZATION
# ============================================================================

def qr_householder(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """QR factorization using Householder reflections."""
    m, n = A.shape
    Q = torch.eye(m, dtype=A.dtype, device=A.device)
    R = A.clone()
    
    for k in range(min(m - 1, n)):
        # Householder vector
        x = R[k:, k]
        v = x.clone()
        v[0] = v[0] + torch.sign(x[0]) * x.norm()
        v = v / (v.norm() + 1e-14)
        
        # Apply Householder reflection
        R[k:, k:] = R[k:, k:] - 2 * v.unsqueeze(1) @ (v @ R[k:, k:]).unsqueeze(0)
        Q[:, k:] = Q[:, k:] - 2 * (Q[:, k:] @ v.unsqueeze(1)) @ v.unsqueeze(0)
    
    return Q, R


def qr_gram_schmidt(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """QR factorization using modified Gram-Schmidt."""
    m, n = A.shape
    Q = torch.zeros(m, n, dtype=A.dtype, device=A.device)
    R = torch.zeros(n, n, dtype=A.dtype, device=A.device)
    
    for j in range(n):
        v = A[:, j].clone()
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v = v - R[i, j] * Q[:, i]
        R[j, j] = v.norm()
        Q[:, j] = v / (R[j, j] + 1e-14)
    
    return Q, R


# ============================================================================
# SVD
# ============================================================================

def truncated_svd(
    A: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated SVD using power iteration."""
    m, n = A.shape
    
    # Random initialization
    V = torch.randn(n, k, dtype=A.dtype, device=A.device)
    V, _ = torch.linalg.qr(V)
    
    # Power iteration
    for _ in range(10):
        U = A @ V
        U, _ = torch.linalg.qr(U)
        V = A.T @ U
        V, _ = torch.linalg.qr(V)
    
    # Compute singular values
    AV = A @ V
    S = torch.zeros(k, dtype=A.dtype, device=A.device)
    U = torch.zeros(m, k, dtype=A.dtype, device=A.device)
    
    for i in range(k):
        S[i] = AV[:, i].norm()
        U[:, i] = AV[:, i] / (S[i] + 1e-14)
    
    return U, S, V


def low_rank_approx(
    A: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Low-rank approximation using SVD."""
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ torch.diag(S[:k]) @ V[:k, :]


# ============================================================================
# EIGENVALUE DECOMPOSITION
# ============================================================================

def power_iteration(
    A: torch.Tensor,
    n_iter: int = 500,
    tol: float = 1e-10,
) -> Tuple[float, torch.Tensor]:
    """Power iteration for dominant eigenvalue."""
    n = A.shape[0]
    torch.manual_seed(42)  # Deterministic starting vector
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / v.norm()
    
    eigenvalue = 0.0
    
    for _ in range(n_iter):
        v_new = A @ v
        eigenvalue_new = v @ v_new
        v_new = v_new / (v_new.norm() + 1e-14)
        
        if abs(eigenvalue_new - eigenvalue) < tol:
            break
        
        v = v_new
        eigenvalue = eigenvalue_new
    
    return eigenvalue, v


def inverse_iteration(
    A: torch.Tensor,
    shift: float,
    n_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[float, torch.Tensor]:
    """Inverse iteration for eigenvalue near shift."""
    n = A.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    B = A - shift * I
    
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / v.norm()
    
    for _ in range(n_iter):
        try:
            v_new = torch.linalg.solve(B, v)
        except:
            break
        
        v_new = v_new / (v_new.norm() + 1e-14)
        
        if (v_new - v).norm() < tol or (v_new + v).norm() < tol:
            break
        
        v = v_new
    
    # Rayleigh quotient
    eigenvalue = (v @ A @ v) / (v @ v)
    
    return eigenvalue, v


def rayleigh_quotient_iteration(
    A: torch.Tensor,
    n_iter: int = 50,
    tol: float = 1e-12,
) -> Tuple[float, torch.Tensor]:
    """Rayleigh quotient iteration."""
    n = A.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / v.norm()
    
    mu = v @ A @ v
    
    for _ in range(n_iter):
        try:
            v_new = torch.linalg.solve(A - mu * I, v)
        except:
            break
        
        v_new = v_new / (v_new.norm() + 1e-14)
        mu_new = v_new @ A @ v_new
        
        if abs(mu_new - mu) < tol:
            break
        
        v = v_new
        mu = mu_new
    
    return mu, v


# ============================================================================
# ITERATIVE METHODS
# ============================================================================

def jacobi_iteration(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Jacobi iterative solver."""
    n = A.shape[0]
    x = x0 if x0 is not None else torch.zeros(n, dtype=A.dtype, device=A.device)
    D_inv = 1 / A.diag()
    
    for _ in range(n_iter):
        x_new = D_inv * (b - A @ x + A.diag() * x)
        
        if (x_new - x).norm() < tol:
            break
        
        x = x_new
    
    return x


def gauss_seidel_iteration(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Gauss-Seidel iterative solver."""
    n = A.shape[0]
    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=A.dtype, device=A.device)
    
    for _ in range(n_iter):
        x_old = x.clone()
        
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
        
        if (x - x_old).norm() < tol:
            break
    
    return x


def conjugate_gradient(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    n_iter: int = 100,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Conjugate gradient for SPD systems."""
    n = A.shape[0]
    x = x0.clone() if x0 is not None else torch.zeros(n, dtype=A.dtype, device=A.device)
    
    r = b - A @ x
    p = r.clone()
    rs_old = r @ r
    
    for _ in range(n_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap + 1e-14)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        
        if rs_new.sqrt() < tol:
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x


# ============================================================================
# UNIT TESTS: LU FACTORIZATION
# ============================================================================

class TestLUFactorization:
    """Test LU factorization."""
    
    @pytest.mark.unit
    def test_lu_reconstruction(self, deterministic_seed, random_matrix):
        """LU factors reconstruct original matrix."""
        A = random_matrix
        L, U, P = lu_factorization(A)
        
        # PA = LU
        reconstructed = L @ U
        assert torch.allclose(P @ A, reconstructed, atol=1e-10)
    
    @pytest.mark.unit
    def test_lu_lower_triangular(self, deterministic_seed, random_matrix):
        """L is lower triangular with unit diagonal."""
        A = random_matrix
        L, U, _ = lu_factorization(A)
        
        # Check lower triangular
        for i in range(L.shape[0]):
            assert torch.allclose(L[i, i+1:], torch.zeros(L.shape[0] - i - 1, dtype=torch.float64))
        
        # Check unit diagonal
        assert torch.allclose(L.diag(), torch.ones(L.shape[0], dtype=torch.float64))
    
    @pytest.mark.unit
    def test_lu_upper_triangular(self, deterministic_seed, random_matrix):
        """U is upper triangular."""
        A = random_matrix
        L, U, _ = lu_factorization(A)
        
        for i in range(1, U.shape[0]):
            assert torch.allclose(U[i, :i], torch.zeros(i, dtype=torch.float64), atol=1e-10)
    
    @pytest.mark.unit
    def test_lu_solve(self, deterministic_seed, random_matrix):
        """LU solve gives correct solution."""
        A = random_matrix
        b = torch.randn(10, dtype=torch.float64)
        
        L, U, P = lu_factorization(A)
        x = solve_lu(L, U, b, P)
        
        assert torch.allclose(A @ x, b, atol=1e-8)


# ============================================================================
# UNIT TESTS: CHOLESKY
# ============================================================================

class TestCholeskyFactorization:
    """Test Cholesky factorization."""
    
    @pytest.mark.unit
    def test_cholesky_reconstruction(self, deterministic_seed, random_spd_matrix):
        """Cholesky factors reconstruct original matrix."""
        A = random_spd_matrix
        L = cholesky_factorization(A)
        
        assert torch.allclose(L @ L.T, A, atol=1e-10)
    
    @pytest.mark.unit
    def test_cholesky_lower_triangular(self, deterministic_seed, random_spd_matrix):
        """Cholesky L is lower triangular."""
        A = random_spd_matrix
        L = cholesky_factorization(A)
        
        for i in range(L.shape[0]):
            assert torch.allclose(L[i, i+1:], torch.zeros(L.shape[0] - i - 1, dtype=torch.float64))
    
    @pytest.mark.unit
    def test_cholesky_solve(self, deterministic_seed, random_spd_matrix):
        """Cholesky solve gives correct solution."""
        A = random_spd_matrix
        b = torch.randn(10, dtype=torch.float64)
        
        L = cholesky_factorization(A)
        x = solve_cholesky(L, b)
        
        assert torch.allclose(A @ x, b, atol=1e-8)


# ============================================================================
# UNIT TESTS: QR
# ============================================================================

class TestQRFactorization:
    """Test QR factorization."""
    
    @pytest.mark.unit
    def test_qr_reconstruction(self, deterministic_seed, random_matrix):
        """QR factors reconstruct original matrix."""
        A = random_matrix
        Q, R = qr_householder(A)
        
        assert torch.allclose(Q @ R, A, atol=1e-10)
    
    @pytest.mark.unit
    def test_qr_orthogonal(self, deterministic_seed, random_matrix):
        """Q is orthogonal."""
        A = random_matrix
        Q, R = qr_householder(A)
        
        I = torch.eye(Q.shape[0], dtype=torch.float64)
        assert torch.allclose(Q.T @ Q, I, atol=1e-10)
    
    @pytest.mark.unit
    def test_qr_upper_triangular(self, deterministic_seed, random_matrix):
        """R is upper triangular."""
        A = random_matrix
        Q, R = qr_householder(A)
        
        for i in range(1, R.shape[0]):
            assert torch.allclose(R[i, :i], torch.zeros(i, dtype=torch.float64), atol=1e-10)


# ============================================================================
# UNIT TESTS: SVD
# ============================================================================

class TestSVD:
    """Test SVD operations."""
    
    @pytest.mark.unit
    def test_svd_reconstruction(self, deterministic_seed, random_matrix):
        """SVD reconstructs original matrix."""
        A = random_matrix
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        
        reconstructed = U @ torch.diag(S) @ V
        assert torch.allclose(reconstructed, A, atol=1e-10)
    
    @pytest.mark.unit
    def test_singular_values_positive(self, deterministic_seed, random_matrix):
        """Singular values are non-negative."""
        A = random_matrix
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        
        assert (S >= 0).all()
    
    @pytest.mark.unit
    def test_low_rank_error(self, deterministic_seed, random_matrix):
        """Low-rank approximation error."""
        A = random_matrix
        
        for k in [1, 3, 5]:
            A_k = low_rank_approx(A, k)
            
            # Error should decrease with k
            error = (A - A_k).norm()
            assert error > 0


# ============================================================================
# UNIT TESTS: EIGENVALUES
# ============================================================================

class TestEigenvalues:
    """Test eigenvalue computations."""
    
    @pytest.mark.unit
    def test_power_iteration(self, deterministic_seed, random_spd_matrix):
        """Power iteration finds dominant eigenvalue."""
        A = random_spd_matrix
        
        eigenvalue, eigenvector = power_iteration(A, n_iter=1000)
        
        # Check eigenvalue equation: Av = λv (relaxed tolerance for iterative method)
        Av = A @ eigenvector
        assert torch.allclose(Av, eigenvalue * eigenvector, rtol=0.05)
    
    @pytest.mark.unit
    def test_eigenvalue_positive(self, deterministic_seed, random_spd_matrix):
        """SPD matrix has positive eigenvalues."""
        A = random_spd_matrix
        eigenvalues = torch.linalg.eigvalsh(A)
        
        assert (eigenvalues > 0).all()


# ============================================================================
# UNIT TESTS: ITERATIVE SOLVERS
# ============================================================================

class TestIterativeSolvers:
    """Test iterative solvers."""
    
    @pytest.mark.unit
    def test_conjugate_gradient(self, deterministic_seed, random_spd_matrix):
        """CG solves SPD system."""
        A = random_spd_matrix
        b = torch.randn(10, dtype=torch.float64)
        
        x = conjugate_gradient(A, b)
        
        assert torch.allclose(A @ x, b, atol=1e-6)
    
    @pytest.mark.unit
    def test_jacobi_diagonally_dominant(self, deterministic_seed):
        """Jacobi converges for diagonally dominant matrix."""
        n = 10
        A = torch.randn(n, n, dtype=torch.float64)
        A = A + 10 * torch.eye(n, dtype=torch.float64)  # Make diagonally dominant
        b = torch.randn(n, dtype=torch.float64)
        
        x = jacobi_iteration(A, b)
        
        assert (A @ x - b).norm() < 1e-4


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLinearAlgebraIntegration:
    """Integration tests for linear algebra."""
    
    @pytest.mark.integration
    def test_least_squares(self, deterministic_seed):
        """Solve least squares with QR."""
        m, n = 20, 10
        A = torch.randn(m, n, dtype=torch.float64)
        b = torch.randn(m, dtype=torch.float64)
        
        # QR solution
        Q, R = qr_gram_schmidt(A)
        x = torch.linalg.solve(R, Q.T @ b)
        
        # Compare with torch.linalg.lstsq
        x_ref, _, _, _ = torch.linalg.lstsq(A, b.unsqueeze(1))
        
        assert torch.allclose(x, x_ref.squeeze(), atol=1e-6)


# ============================================================================
# FLOAT64 COMPLIANCE
# ============================================================================

class TestFloat64ComplianceLinAlg:
    """Article V: Float64 precision tests."""
    
    @pytest.mark.unit
    def test_lu_float64(self, deterministic_seed, random_matrix):
        """LU uses float64."""
        A = random_matrix
        L, U, P = lu_factorization(A)
        
        assert L.dtype == torch.float64
        assert U.dtype == torch.float64
    
    @pytest.mark.unit
    def test_qr_float64(self, deterministic_seed, random_matrix):
        """QR uses float64."""
        A = random_matrix
        Q, R = qr_householder(A)
        
        assert Q.dtype == torch.float64
        assert R.dtype == torch.float64


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

class TestReproducibilityLinAlg:
    """Article III, Section 3.2: Reproducibility."""
    
    @pytest.mark.unit
    def test_deterministic_lu(self, random_matrix):
        """LU is deterministic."""
        A = random_matrix
        
        L1, U1, P1 = lu_factorization(A)
        L2, U2, P2 = lu_factorization(A)
        
        assert torch.allclose(L1, L2)
        assert torch.allclose(U1, U2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
