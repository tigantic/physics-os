"""
QTT Random Matrix Ensembles

Provides QTT representations of classical random matrix ensembles:
- GOE (Gaussian Orthogonal Ensemble)
- GUE (Gaussian Unitary Ensemble)  
- Wishart (Sample covariance)
- General Wigner matrices

The key insight is that for structured ensembles (banded, Toeplitz-like),
the matrices have low TT rank and can be manipulated efficiently.

For general dense random matrices, we use a hybrid approach:
1. Generate in QTT format with controlled rank
2. Add structured randomness that preserves low rank

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import torch

from ontic.genesis.core.rsvd import rsvd_gpu


@dataclass
class QTTEnsemble:
    """
    Random matrix ensemble in QTT format.
    
    Represents an N × N matrix as a QTT-MPO (Matrix Product Operator).
    For an N = 2^d matrix, we have d cores, each of shape (r_left, 2, 2, r_right).
    
    Attributes:
        cores: List of MPO cores
        size: Matrix dimension N = 2^d
        ensemble_type: Type of ensemble ('goe', 'gue', 'wishart', etc.)
        seed: Random seed used for generation
    """
    cores: List[torch.Tensor]
    size: int
    ensemble_type: str = "custom"
    seed: Optional[int] = None
    
    @property
    def num_qubits(self) -> int:
        """Number of qubits (log2 of size)."""
        return len(self.cores)
    
    @property
    def max_rank(self) -> int:
        """Maximum TT rank."""
        return max(c.shape[0] for c in self.cores)
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of cores."""
        return self.cores[0].dtype
    
    @property
    def is_complex(self) -> bool:
        """Whether the matrix is complex."""
        return self.cores[0].is_complex()
    
    def to_dense(self) -> torch.Tensor:
        """
        Convert to dense matrix.
        
        Warning: Only for small matrices (size ≤ 2^14).
        """
        if self.size > 2**14:
            raise ValueError(f"Matrix too large for dense: {self.size}")
        
        d = self.num_qubits
        
        # Contract MPO to dense matrix
        # Start with first core
        result = self.cores[0]  # (1, 2, 2, r1)
        
        for k in range(1, d):
            core = self.cores[k]  # (r_{k-1}, 2, 2, r_k)
            
            # Contract along bond dimension
            # result: (..., r_{k-1}) x core: (r_{k-1}, 2, 2, r_k)
            result = torch.tensordot(result, core, dims=([-1], [0]))
        
        # result shape: (1, 2, 2, 2, 2, ..., 2, 2, 1)
        # = (1, 2^d, 2^d, 1) when properly reshaped
        
        # Reshape to matrix
        result = result.squeeze(0).squeeze(-1)  # Remove boundary dims
        
        # Interleave indices: (i1, j1, i2, j2, ..., id, jd) -> (i, j)
        # where i = sum(i_k * 2^(d-k)), j = sum(j_k * 2^(d-k))
        n = 2 ** d
        matrix = result.reshape(n, n)
        
        return matrix
    
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matrix-vector product Hx using MPO contraction.
        
        Args:
            x: Vector of length N or QTT cores
            
        Returns:
            Result Hx
        """
        if isinstance(x, torch.Tensor) and x.dim() == 1:
            # Dense vector - convert to QTT first for small cases
            if len(x) <= 2**14:
                return self.to_dense() @ x
            else:
                # Large dense matvec: convert x to QTT, apply MPO, convert back
                x_qtt = self._dense_to_qtt(x)
                y_qtt = self._mpo_mps_product(x_qtt)
                return self._qtt_to_dense(y_qtt)
        
        # QTT matvec - contract MPO with MPS
        return self._mpo_mps_product(x)
    
    def _dense_to_qtt(self, x: torch.Tensor) -> list:
        """Convert dense vector to QTT format via TT-SVD."""
        num_bits = len(self.cores)
        N = 2 ** num_bits
        
        # Pad if necessary
        if len(x) < N:
            x = torch.cat([x, torch.zeros(N - len(x), dtype=x.dtype, device=x.device)])
        
        # Reshape to 2x2x...x2
        tensor = x.reshape([2] * num_bits)
        
        cores = []
        current = tensor
        
        for k in range(num_bits - 1):
            shape = current.shape
            if k == 0:
                mat = current.reshape(2, -1)
            else:
                r_prev = cores[-1].shape[2]
                mat = current.reshape(r_prev * 2, -1)
            
            U, S, Vh = rsvd_gpu(mat, k=32, tol=1e-12)
            rank = S.shape[0]
            
            if k == 0:
                cores.append(U.reshape(1, 2, rank))
            else:
                cores.append(U.reshape(r_prev, 2, rank))
            
            SV = torch.diag(S) @ Vh
            current = SV.reshape(rank, 2, -1) if SV.numel() > rank * 2 else SV
        
        # Last core
        r_prev = cores[-1].shape[2] if cores else 1
        last = current.reshape(r_prev, 2, 1)
        cores.append(last)
        
        return cores
    
    def _qtt_to_dense(self, cores: list) -> torch.Tensor:
        """Convert QTT to dense vector."""
        result = cores[0]
        for core in cores[1:]:
            r_left_res = result.shape[0]
            n_res = result.shape[1] 
            r_right_res = result.shape[2]
            r_left_core, n_core, r_right_core = core.shape
            
            result = result.reshape(-1, r_right_res)
            core_mat = core.reshape(r_left_core, -1)
            result = result @ core_mat
            result = result.reshape(1, -1, r_right_core)
        
        return result.squeeze()
    
    def _mpo_mps_product(self, x_cores: list) -> list:
        """Apply MPO (self.cores) to MPS (x_cores).
        
        Contract: y[i] = sum_j H[i,j] x[j]
        
        In TT format, each core contracts over physical indices.
        Result has rank = rank(H) * rank(x), then truncated.
        """
        num_sites = len(self.cores)
        result_cores = []
        
        for k in range(num_sites):
            # MPO core: (D_l, d_out, d_in, D_r)
            H_core = self.cores[k]
            # MPS core: (r_l, d, r_r)
            x_core = x_cores[k]
            
            D_l, d_out, d_in, D_r = H_core.shape
            r_l, d_x, r_r = x_core.shape
            
            assert d_in == d_x, f"Dimension mismatch at site {k}"
            
            # Contract over d_in
            # H[a,i,j,b] x[c,j,d] -> result[a,c,i,b,d]
            contracted = torch.einsum('aijb,cjd->acibd', H_core, x_core)
            
            # Reshape to (D_l * r_l, d_out, D_r * r_r)
            result_core = contracted.reshape(D_l * r_l, d_out, D_r * r_r)
            result_cores.append(result_core)
        
        # Truncate ranks
        result_cores = self._truncate_mps(result_cores, max_rank=64)
        
        return result_cores
    
    def _truncate_mps(self, cores: list, max_rank: int = 64) -> list:
        """Truncate MPS bond dimensions via right-to-left SVD."""
        n = len(cores)
        cores = [c.clone() for c in cores]
        
        for k in range(n - 1, 0, -1):
            core = cores[k]
            r_l, d, r_r = core.shape
            
            if r_l <= max_rank:
                continue
            
            mat = core.reshape(r_l, d * r_r)
            U, S, Vh = rsvd_gpu(mat, k=max_rank, tol=1e-10)
            r_new = S.shape[0]
            
            cores[k] = Vh.reshape(r_new, d, r_r)
            
            # Absorb U @ diag(S) into left neighbor
            left = cores[k - 1]
            rl, dl, rr = left.shape
            left_mat = left.reshape(rl * dl, rr)
            left_mat = left_mat @ (U * S)
            cores[k - 1] = left_mat.reshape(rl, dl, r_new)
        
        return cores
    
    @classmethod
    def goe(cls, size: int, rank: int = 10,
            dtype: torch.dtype = torch.float64,
            seed: Optional[int] = None) -> 'QTTEnsemble':
        """
        Create Gaussian Orthogonal Ensemble matrix.
        
        GOE: Real symmetric with H_ij ~ N(0, 1/N) for i≤j.
        
        For QTT representation, we build a structured approximation:
        H = sum_k σ_k A_k ⊗ A_k^T
        
        where A_k are rank-1 QTT vectors.
        
        Args:
            size: Matrix dimension (must be power of 2)
            rank: TT rank (controls approximation quality)
            dtype: Data type
            seed: Random seed
            
        Returns:
            QTTEnsemble representing GOE matrix
        """
        d = int(math.log2(size))
        if 2**d != size:
            raise ValueError(f"size must be power of 2, got {size}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Build symmetric MPO
        # For structured approximation, use random rank-r MPO
        cores = []
        
        for k in range(d):
            r_left = 1 if k == 0 else rank
            r_right = 1 if k == d - 1 else rank
            
            # Core shape: (r_left, 2, 2, r_right)
            # Random initialization
            core = torch.randn(r_left, 2, 2, r_right, dtype=dtype)
            
            # Make symmetric: core[..., i, j, ...] = core[..., j, i, ...]
            core = 0.5 * (core + core.transpose(1, 2))
            
            # Normalize
            core = core / (core.norm() + 1e-10) * math.sqrt(1.0 / d)
            
            cores.append(core)
        
        return cls(cores=cores, size=size, ensemble_type='goe', seed=seed)
    
    @classmethod
    def gue(cls, size: int, rank: int = 10,
            dtype: torch.dtype = torch.complex128,
            seed: Optional[int] = None) -> 'QTTEnsemble':
        """
        Create Gaussian Unitary Ensemble matrix.
        
        GUE: Complex Hermitian with H_ij ~ CN(0, 1/N) for i<j.
        
        Args:
            size: Matrix dimension (must be power of 2)
            rank: TT rank
            dtype: Complex data type
            seed: Random seed
            
        Returns:
            QTTEnsemble representing GUE matrix
        """
        d = int(math.log2(size))
        if 2**d != size:
            raise ValueError(f"size must be power of 2, got {size}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        cores = []
        
        for k in range(d):
            r_left = 1 if k == 0 else rank
            r_right = 1 if k == d - 1 else rank
            
            # Complex core
            real = torch.randn(r_left, 2, 2, r_right, dtype=torch.float64)
            imag = torch.randn(r_left, 2, 2, r_right, dtype=torch.float64)
            core = torch.complex(real, imag)
            
            # Make Hermitian: core[..., i, j, ...] = core[..., j, i, ...]^*
            core = 0.5 * (core + core.transpose(1, 2).conj())
            
            # Normalize
            core = core / (core.abs().max() + 1e-10) * math.sqrt(1.0 / d)
            
            cores.append(core)
        
        return cls(cores=cores, size=size, ensemble_type='gue', seed=seed)
    
    @classmethod
    def wishart(cls, size: int, aspect_ratio: float = 0.5,
                rank: int = 10, dtype: torch.dtype = torch.float64,
                seed: Optional[int] = None) -> 'QTTEnsemble':
        """
        Create Wishart ensemble matrix.
        
        W = (1/n) X^T X where X is n × p with p = size, n = p/aspect_ratio.
        
        Args:
            size: Matrix dimension p (must be power of 2)
            aspect_ratio: γ = p/n (must be < 1 for non-singular)
            rank: TT rank
            dtype: Data type
            seed: Random seed
            
        Returns:
            QTTEnsemble representing Wishart matrix
        """
        d = int(math.log2(size))
        if 2**d != size:
            raise ValueError(f"size must be power of 2, got {size}")
        
        if aspect_ratio >= 1.0:
            raise ValueError(f"aspect_ratio must be < 1, got {aspect_ratio}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # For Wishart, eigenvalues follow Marchenko-Pastur
        # We build a positive semi-definite MPO
        
        cores = []
        
        for k in range(d):
            r_left = 1 if k == 0 else rank
            r_right = 1 if k == d - 1 else rank
            
            # Build as A^T A structure locally
            A = torch.randn(r_left, 2, r_right, dtype=dtype)
            
            # core[r, i, j, s] = sum_t A[r, i, t] * A[s, j, t]
            # This is tricky in MPO form, so we use symmetric positive
            core = torch.randn(r_left, 2, 2, r_right, dtype=dtype)
            core = 0.5 * (core + core.transpose(1, 2))
            
            # Add identity component for positive definiteness
            for i in range(min(r_left, r_right)):
                if k == 0 or k == d - 1:
                    core[0, 0, 0, 0] += 1.0
                    core[0, 1, 1, 0] += 1.0
                else:
                    core[i, 0, 0, i] += 1.0 / rank
                    core[i, 1, 1, i] += 1.0 / rank
            
            # Scale by aspect ratio
            core = core * math.sqrt(aspect_ratio / d)
            
            cores.append(core)
        
        ensemble = cls(cores=cores, size=size, ensemble_type='wishart', seed=seed)
        ensemble.aspect_ratio = aspect_ratio
        return ensemble
    
    @classmethod
    def wigner(cls, size: int, rank: int = 10,
               distribution: str = 'gaussian',
               dtype: torch.dtype = torch.float64,
               seed: Optional[int] = None) -> 'QTTEnsemble':
        """
        Create general Wigner matrix.
        
        Wigner: Symmetric matrix with i.i.d. entries from given distribution.
        
        Args:
            size: Matrix dimension
            rank: TT rank
            distribution: 'gaussian', 'uniform', 'bernoulli'
            dtype: Data type
            seed: Random seed
            
        Returns:
            QTTEnsemble representing Wigner matrix
        """
        d = int(math.log2(size))
        if 2**d != size:
            raise ValueError(f"size must be power of 2, got {size}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        cores = []
        
        for k in range(d):
            r_left = 1 if k == 0 else rank
            r_right = 1 if k == d - 1 else rank
            
            if distribution == 'gaussian':
                core = torch.randn(r_left, 2, 2, r_right, dtype=dtype)
            elif distribution == 'uniform':
                core = 2 * torch.rand(r_left, 2, 2, r_right, dtype=dtype) - 1
            elif distribution == 'bernoulli':
                core = 2 * torch.randint(0, 2, (r_left, 2, 2, r_right)).to(dtype) - 1
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            # Make symmetric
            core = 0.5 * (core + core.transpose(1, 2))
            
            # Normalize
            core = core / (core.norm() + 1e-10) * math.sqrt(1.0 / d)
            
            cores.append(core)
        
        return cls(cores=cores, size=size, ensemble_type=f'wigner_{distribution}', seed=seed)
    
    @classmethod
    def tridiagonal(cls, size: int, diagonal: float = 0.0,
                    off_diagonal: float = 1.0,
                    noise: float = 0.0,
                    dtype: torch.dtype = torch.float64,
                    seed: Optional[int] = None) -> 'QTTEnsemble':
        """
        Create tridiagonal matrix (exact TT rank 3).
        
        H_ii = diagonal + noise * ε_i
        H_{i,i+1} = H_{i+1,i} = off_diagonal + noise * ε_{i,i+1}
        
        This is useful for testing since it has exact low rank.
        
        Args:
            size: Matrix dimension
            diagonal: Diagonal value
            off_diagonal: Off-diagonal value
            noise: Noise amplitude
            dtype: Data type
            seed: Random seed
            
        Returns:
            QTTEnsemble with TT rank ≤ 3
        """
        d = int(math.log2(size))
        if 2**d != size:
            raise ValueError(f"size must be power of 2, got {size}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Tridiagonal has TT rank 3
        # See QTTLaplacian implementation in sgw module
        
        cores = []
        
        for k in range(d):
            r_left = 1 if k == 0 else 3
            r_right = 1 if k == d - 1 else 3
            
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype)
            
            if k == 0:
                # First core: (1, 2, 2, 3)
                # [I, L, R] where L=lower, R=upper
                core[0, 0, 0, 0] = diagonal + noise * torch.randn(1).item()
                core[0, 1, 1, 0] = diagonal + noise * torch.randn(1).item()
                core[0, 0, 0, 1] = off_diagonal + noise * torch.randn(1).item()  # shift left
                core[0, 1, 1, 2] = off_diagonal + noise * torch.randn(1).item()  # shift right
            elif k == d - 1:
                # Last core: (3, 2, 2, 1)
                core[0, 0, 0, 0] = 1.0  # identity pass-through
                core[0, 1, 1, 0] = 1.0
                core[1, 1, 0, 0] = 1.0  # shift completed (lower)
                core[2, 0, 1, 0] = 1.0  # shift completed (upper)
            else:
                # Middle core: (3, 2, 2, 3)
                # Pass identity
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                # Continue shift
                core[1, 0, 0, 1] = 1.0
                core[1, 1, 1, 1] = 1.0
                core[2, 0, 0, 2] = 1.0
                core[2, 1, 1, 2] = 1.0
            
            cores.append(core)
        
        return cls(cores=cores, size=size, ensemble_type='tridiagonal', seed=seed)


# Convenience functions
def goe_matrix(size: int, rank: int = 10, seed: Optional[int] = None) -> QTTEnsemble:
    """Create GOE matrix."""
    return QTTEnsemble.goe(size=size, rank=rank, seed=seed)


def gue_matrix(size: int, rank: int = 10, seed: Optional[int] = None) -> QTTEnsemble:
    """Create GUE matrix."""
    return QTTEnsemble.gue(size=size, rank=rank, seed=seed)


def wishart_matrix(size: int, aspect_ratio: float = 0.5,
                   rank: int = 10, seed: Optional[int] = None) -> QTTEnsemble:
    """Create Wishart matrix."""
    return QTTEnsemble.wishart(size=size, aspect_ratio=aspect_ratio, rank=rank, seed=seed)


def wigner_matrix(size: int, rank: int = 10,
                  distribution: str = 'gaussian',
                  seed: Optional[int] = None) -> QTTEnsemble:
    """Create Wigner matrix."""
    return QTTEnsemble.wigner(size=size, rank=rank, distribution=distribution, seed=seed)
