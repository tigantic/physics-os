"""
QTT Kernel Matrix Operations

Implements QTT-compressed kernel matrix storage and operations.

The key insight: for hierarchical data structures or low-dimensional
manifolds, the kernel matrix often has low TT rank due to smooth
distance functions.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import torch

from ontic.genesis.core.rsvd import rsvd_gpu
from .kernels import Kernel, RBFKernel


@dataclass
class QTTKernelMatrix:
    """
    QTT-compressed kernel matrix.
    
    For N = 2^L points, represents K ∈ ℝ^{N×N} with TT cores.
    
    The QTT representation exploits:
    1. Smooth kernel functions → low numerical rank
    2. Hierarchical structure → fast multiplication
    
    Attributes:
        cores: List of TT cores, each (r_{k-1}, 2, 2, r_k)
        n_bits: Number of bits L (N = 2^L)
        kernel: Original kernel function
    """
    cores: List[torch.Tensor] = field(default_factory=list)
    n_bits: int = 0
    kernel: Optional[Kernel] = None
    
    @classmethod
    def from_dense(cls, K: torch.Tensor, 
                   max_rank: int = 50,
                   tol: float = 1e-10) -> 'QTTKernelMatrix':
        """
        Compress dense kernel matrix to QTT format.
        
        Uses TT-SVD algorithm for matrix decomposition.
        
        Args:
            K: Dense kernel matrix, shape (N, N) where N = 2^L
            max_rank: Maximum TT rank
            tol: SVD truncation tolerance
            
        Returns:
            QTT kernel matrix
        """
        N = K.shape[0]
        assert K.shape[1] == N, "Matrix must be square"
        
        # Check N is power of 2
        L = int(math.log2(N))
        assert 2 ** L == N, f"N must be power of 2, got {N}"
        
        # Use simpler TT decomposition via repeated SVD
        # Reshape K as vector then apply TT-SVD
        cores = []
        
        # Flatten matrix to vector for TT decomposition
        vec = K.reshape(-1)  # N^2 = 2^{2L}
        
        # TT decomposition of vec with 2L modes of size 2
        current = vec.reshape(2, -1)  # (2, 2^{2L-1})
        
        for k in range(2 * L - 1):
            m, n = current.shape
            
            # SVD truncation via randomized SVD
            U, S, Vh = rsvd_gpu(current, k=max_rank, tol=tol)
            
            # Form core
            if k == 0:
                core = U.reshape(1, 2, rank)
            else:
                r_prev = cores[-1].shape[-1]
                # Ensure dimensions match
                if m != r_prev * 2:
                    # Pad or truncate if needed
                    core = U.reshape(-1, 2, rank)[:r_prev, :, :]
                else:
                    core = U.reshape(r_prev, 2, rank)
            
            cores.append(core)
            
            # Continue with remaining
            if k < 2 * L - 2:
                remaining = torch.diag(S) @ Vh
                current = remaining.reshape(rank * 2, -1)
            else:
                # Last core
                core = (torch.diag(S) @ Vh).reshape(rank, -1, 1)
                if core.shape[1] != 2:
                    # Handle non-standard final shape
                    core = core[:, :min(2, core.shape[1]), :]
                    if core.shape[1] < 2:
                        core = torch.nn.functional.pad(core, (0, 0, 0, 2 - core.shape[1]))
                cores.append(core)
        
        return cls(cores=cores, n_bits=L, kernel=None)
        
        return cls(cores=cores, n_bits=L, kernel=None)
    
    @classmethod
    def from_kernel(cls, kernel: Kernel,
                    x: torch.Tensor,
                    max_rank: int = 50,
                    tol: float = 1e-10) -> 'QTTKernelMatrix':
        """
        Build QTT kernel matrix from kernel and data points.
        
        Args:
            kernel: Kernel function
            x: Data points, shape (N, d) where N = 2^L
            max_rank: Maximum TT rank
            tol: SVD truncation tolerance
            
        Returns:
            QTT kernel matrix
        """
        K = kernel.matrix(x)
        result = cls.from_dense(K, max_rank=max_rank, tol=tol)
        result.kernel = kernel
        return result
    
    def to_dense(self) -> torch.Tensor:
        """
        Reconstruct dense matrix from QTT format.
        
        Returns:
            Dense matrix, shape (N, N)
        """
        if not self.cores:
            return torch.tensor([[]])
        
        # Contract all cores
        result = self.cores[0]  # (1, 2, r_1)
        
        for core in self.cores[1:]:
            # result: (1, 2^k, r_k)
            # core: (r_k, 2, r_{k+1})
            r_prev = result.shape[-1]
            n = result.shape[1]
            
            # Contract: sum over r_k
            # result[0, i, :] @ core[:, j, :] → new[0, i*2+j, :]
            result = torch.einsum('aib,bjc->aijc', result, core)
            result = result.reshape(1, n * 2, -1)
        
        N = result.shape[1]
        return result.reshape(N, -1)[:, :N]
    
    @property
    def ranks(self) -> List[int]:
        """Get TT ranks."""
        return [c.shape[-1] for c in self.cores[:-1]] + [1] if self.cores else []
    
    @property
    def max_rank(self) -> int:
        """Get maximum TT rank."""
        return max(self.ranks) if self.ranks else 0
    
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """
        Matrix-vector product K @ v in O(r² L N) time.
        
        Uses QTT structure for efficient contraction without forming dense K.
        
        For matrix QTT with 2L cores (L bits for rows, L for cols),
        we contract the column indices with v and sum over them.
        
        Algorithm:
        1. Reshape v as QTT: (2, 2, ..., 2) with L modes
        2. Contract matrix cores with vector cores from right
        3. Result is QTT of output vector
        
        Complexity: O(r² L) contractions, each O(r²) → O(r² L) total
        vs O(N²) = O(4^L) for dense matvec.
        
        Args:
            v: Vector, shape (N,) where N = 2^L
            
        Returns:
            Result vector, shape (N,)
        """
        N = v.shape[0]
        L = self.n_bits
        
        # For small matrices, dense is faster due to BLAS optimization
        if N <= 256:
            K = self.to_dense()
            return K @ v
        
        # Reshape v to QTT-like structure: (2, 2, ..., 2) with L modes
        v_reshaped = v.reshape((2,) * L)
        
        # The matrix QTT has 2L cores for an N×N matrix where N=2^L
        # Cores alternate between row and column indices
        # We need to contract over column indices (every other mode)
        
        if len(self.cores) == 2 * L:
            # Full matrix-QTT with 2L cores
            # Contract from right to left for numerical stability
            result = None
            v_idx = L - 1  # Start from last v index
            
            for k in range(2 * L - 1, -1, -1):
                core = self.cores[k]  # (r_{k-1}, 2, r_k)
                
                if k % 2 == 1:  # Column index - contract with v
                    if v_idx >= 0:
                        # Contract core with v slice
                        v_slice = v_reshaped.select(v_idx, 0) if v_idx == 0 else v_reshaped
                        # Sum over mode 1 (the column index)
                        if result is None:
                            # First contraction: (r_{k-1}, 2, r_k) @ v[..., 2] → (r_{k-1}, ..., r_k)
                            result = torch.einsum('abc,c->ab', core, torch.ones(core.shape[-1], device=core.device))
                        else:
                            # Contract core with current result
                            result = torch.einsum('abc,cd->abd', core, result)
                            result = result.sum(dim=1)  # Sum over column mode
                        v_idx -= 1
                else:  # Row index - keep in result
                    if result is None:
                        result = core
                    else:
                        # Contract rank indices
                        result = torch.einsum('abc,cd->abd', core, result)
            
            # Flatten result to vector
            if result is not None:
                return result.flatten()[:N]
        
        # Fallback: MPO-MPS contraction for general case
        # Convert v to MPS (TT format)
        v_cores = self._vector_to_tt(v, max_rank=max(self.ranks) if self.ranks else 10)
        
        # Contract MPO (matrix QTT) with MPS (vector TT)
        # Result is MPS which we convert back to dense
        result_cores = self._mpo_mps_contraction(self.cores, v_cores)
        
        # Convert result TT back to vector
        result = self._tt_to_vector(result_cores)
        return result[:N]
    
    def _vector_to_tt(self, v: torch.Tensor, max_rank: int = 50) -> List[torch.Tensor]:
        """Convert dense vector to TT format using rSVD."""
        N = v.shape[0]
        L = int(math.log2(N))
        
        v_reshaped = v.reshape((2,) * L)
        cores = []
        current = v_reshaped.reshape(2, -1)
        
        for k in range(L - 1):
            m, n = current.shape
            q = min(max_rank + 5, min(m, n))
            
            U, S, Vh = rsvd_gpu(current, k=max_rank, tol=1e-10)
            V = Vh.T
            
            rank = S.shape[0]
            
            if k == 0:
                cores.append(U.reshape(1, 2, rank))
            else:
                r_prev = cores[-1].shape[-1]
                cores.append(U.reshape(r_prev, 2, rank))
            
            current = (torch.diag(S) @ V.T).reshape(rank * 2, -1)
        
        # Last core
        r_prev = cores[-1].shape[-1] if cores else 1
        cores.append(current.reshape(r_prev, 2, 1))
        
        return cores
    
    def _mpo_mps_contraction(self, mpo_cores: List[torch.Tensor], 
                              mps_cores: List[torch.Tensor]) -> List[torch.Tensor]:
        """Contract MPO with MPS to get result MPS."""
        L = len(mps_cores)
        result_cores = []
        
        # For 2L MPO cores contracting with L MPS cores
        # We pair MPO cores and contract with corresponding MPS core
        
        for k in range(L):
            mps_core = mps_cores[k]  # (r_in, 2, r_out)
            
            if 2 * k < len(mpo_cores):
                # Row core
                mpo_row = mpo_cores[2 * k] if 2 * k < len(mpo_cores) else None
                # Column core  
                mpo_col = mpo_cores[2 * k + 1] if 2 * k + 1 < len(mpo_cores) else None
                
                if mpo_row is not None and mpo_col is not None:
                    # Contract: mpo_row[a,i,b] @ mpo_col[b,j,c] @ mps[d,j,e] → result[a*d, i, c*e]
                    # First contract mpo cores
                    mpo_local = torch.einsum('aib,bjc->aijc', mpo_row, mpo_col)
                    # Then contract j with mps
                    result = torch.einsum('aijc,dje->adiec', mpo_local, mps_core)
                    # Reshape to (r_left, 2, r_right)
                    r_left = result.shape[0] * result.shape[1]
                    r_right = result.shape[3] * result.shape[4]
                    result_cores.append(result.reshape(r_left, 2, r_right))
                else:
                    result_cores.append(mps_core)
            else:
                result_cores.append(mps_core)
        
        return result_cores
    
    def _tt_to_vector(self, cores: List[torch.Tensor]) -> torch.Tensor:
        """Convert TT cores back to dense vector."""
        result = cores[0]
        for core in cores[1:]:
            result = torch.einsum('...a,abc->...bc', result, core)
        return result.flatten()
    
    def solve(self, b: torch.Tensor, 
              reg: float = 1e-6) -> torch.Tensor:
        """
        Solve K x = b using regularized inverse.
        
        Args:
            b: Right-hand side, shape (N,) or (N, m)
            reg: Regularization parameter
            
        Returns:
            Solution x
        """
        K = self.to_dense()
        N = K.shape[0]
        K_reg = K + reg * torch.eye(N)
        
        return torch.linalg.solve(K_reg, b)
    
    def logdet(self, reg: float = 1e-6) -> float:
        """
        Compute log determinant log|K + λI|.
        
        Args:
            reg: Regularization parameter λ
            
        Returns:
            Log determinant
        """
        K = self.to_dense()
        N = K.shape[0]
        K_reg = K + reg * torch.eye(N)
        
        eigenvalues = torch.linalg.eigvalsh(K_reg)
        return torch.log(eigenvalues).sum().item()
    
    def trace(self) -> float:
        """Compute trace of kernel matrix."""
        K = self.to_dense()
        return torch.trace(K).item()


def kernel_matrix(kernel: Kernel,
                  x: torch.Tensor,
                  y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute kernel matrix K[i,j] = k(x_i, y_j).
    
    Args:
        kernel: Kernel function
        x: First set of points, shape (n, d)
        y: Second set of points, shape (m, d), or None for x
        
    Returns:
        Kernel matrix, shape (n, m)
    """
    return kernel.matrix(x, y)


def kernel_vector(kernel: Kernel,
                  x: torch.Tensor,
                  x_star: torch.Tensor) -> torch.Tensor:
    """
    Compute kernel vector k(x_i, x_*) for fixed x_*.
    
    Args:
        kernel: Kernel function
        x: Training points, shape (n, d)
        x_star: Query point, shape (d,)
        
    Returns:
        Kernel vector, shape (n,)
    """
    return kernel(x, x_star.unsqueeze(0)).squeeze(-1)


def nystrom_approximation(kernel: Kernel,
                          x: torch.Tensor,
                          m: int,
                          seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nyström low-rank approximation of kernel matrix.
    
    Approximates K ≈ K_nm K_mm^{-1} K_mn
    
    Args:
        kernel: Kernel function
        x: Data points, shape (n, d)
        m: Number of landmark points
        seed: Random seed
        
    Returns:
        (L, indices) where K ≈ L L^T and indices are landmarks
    """
    torch.manual_seed(seed)
    n = x.shape[0]
    
    # Select random landmarks
    indices = torch.randperm(n)[:m]
    landmarks = x[indices]
    
    # Compute K_mm and K_nm
    K_mm = kernel.matrix(landmarks)
    K_nm = kernel.matrix(x, landmarks)
    
    # Eigendecomposition of K_mm
    eigenvalues, eigenvectors = torch.linalg.eigh(K_mm)
    
    # Clamp small eigenvalues
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)
    
    # L = K_nm Λ^{-1/2} V^T
    L = K_nm @ (eigenvectors * (1.0 / torch.sqrt(eigenvalues)))
    
    return L, indices


def random_fourier_features(kernel: RBFKernel,
                            x: torch.Tensor,
                            n_features: int,
                            seed: int = 42) -> torch.Tensor:
    """
    Random Fourier feature approximation for RBF kernel.
    
    Uses Bochner's theorem: k(x,y) ≈ φ(x)^T φ(y)
    where φ are random Fourier features.
    
    Args:
        kernel: RBF kernel
        x: Data points, shape (n, d)
        n_features: Number of random features
        seed: Random seed
        
    Returns:
        Feature matrix, shape (n, n_features)
    """
    torch.manual_seed(seed)
    n, d = x.shape
    
    # Sample frequencies from N(0, 1/ℓ²)
    omega = torch.randn(d, n_features) / kernel.length_scale
    
    # Sample phases from U[0, 2π]
    b = torch.rand(n_features) * 2 * math.pi
    
    # Compute features: sqrt(2σ²/D) cos(ωx + b)
    projection = x @ omega + b
    scale = math.sqrt(2 * kernel.variance / n_features)
    
    return scale * torch.cos(projection)


def incomplete_cholesky(K: torch.Tensor,
                        max_rank: int,
                        tol: float = 1e-10) -> torch.Tensor:
    """
    Incomplete Cholesky decomposition for low-rank approximation.
    
    Computes K ≈ L L^T where L has at most max_rank columns.
    
    Args:
        K: Positive semi-definite matrix, shape (n, n)
        max_rank: Maximum rank
        tol: Tolerance for early stopping
        
    Returns:
        Lower triangular factor L, shape (n, r)
    """
    n = K.shape[0]
    d = K.diagonal().clone()
    
    L = torch.zeros(n, max_rank)
    
    for k in range(max_rank):
        # Find pivot (largest remaining diagonal)
        pivot = d.argmax().item()
        
        if d[pivot] < tol:
            return L[:, :k]
        
        # Compute column
        L[pivot, k] = math.sqrt(d[pivot])
        
        for i in range(n):
            if i != pivot:
                L[i, k] = (K[i, pivot] - L[i, :k] @ L[pivot, :k]) / L[pivot, k]
        
        # Update diagonal
        d = d - L[:, k] ** 2
        d = torch.clamp(d, min=0)
    
    return L
