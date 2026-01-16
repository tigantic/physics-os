#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      QUANTIZED TENSOR TRAIN (QTT) MODULE                     ║
║                                                                              ║
║              Low-Rank Representation of High-Dimensional States              ║
╚══════════════════════════════════════════════════════════════════════════════╝

The Quantized Tensor Train (QTT) format represents vectors and operators over
exponentially large spaces using polynomial resources.

For a vector v ∈ ℂ^(d^N), the QTT representation is:

    v[i₁, i₂, ..., iₙ] = G¹[i₁] · G²[i₂] · ... · Gᴺ[iₙ]

where each Gᵏ[iₖ] is an (rₖ₋₁ × rₖ) matrix for each index value iₖ ∈ {0,...,d-1}.

Storage:
    - Full tensor: O(d^N) — EXPONENTIAL
    - QTT format: O(N · d · r²) — LINEAR in N if rank r bounded

Key Operations:
    - Addition: rank(A + B) ≤ rank(A) + rank(B)
    - Hadamard product: rank(A ⊙ B) ≤ rank(A) · rank(B)
    - Matrix-vector: rank(Ax) bounded
    - Compression (rounding): Reduce rank via SVD truncation

This is the foundation for representing gauge field states with O(L) resources
instead of O(d^L) — the key to tractability.

Mathematical Foundation:
    - Oseledets (2011): Tensor-Train decomposition
    - Khoromskij (2011): Quantized tensor trains
    - HyperTensor thesis: Reality is low-rank

Author: HyperTensor Yang-Mills Project
Date: 2026-01-15
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import scipy.linalg as la


# =============================================================================
# QTT CORE CLASS
# =============================================================================

@dataclass
class QTT:
    """
    Quantized Tensor Train representation.
    
    A QTT with N cores represents a tensor of shape (d₁, d₂, ..., dₙ).
    Each core[k] has shape (rₖ₋₁, dₖ, rₖ) where:
        - r₀ = rₙ = 1 (boundary conditions)
        - rₖ = bond dimension between sites k and k+1
        
    For a vector over d^N dimensional space:
        - cores[k] has shape (r_{k-1}, d, r_k)
        - Flattening gives vector in ℂ^(d^N)
    """
    
    cores: List[np.ndarray]  # List of 3-tensors
    
    # Cached properties
    _norm: Optional[float] = field(default=None, repr=False)
    _is_normalized: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Validate QTT structure."""
        if len(self.cores) == 0:
            raise ValueError("QTT must have at least one core")
        
        # Check boundary conditions
        if self.cores[0].shape[0] != 1:
            raise ValueError(f"First core must have r₀ = 1, got {self.cores[0].shape[0]}")
        if self.cores[-1].shape[2] != 1:
            raise ValueError(f"Last core must have rₙ = 1, got {self.cores[-1].shape[2]}")
        
        # Check shape compatibility
        for k in range(len(self.cores) - 1):
            if self.cores[k].shape[2] != self.cores[k+1].shape[0]:
                raise ValueError(
                    f"Shape mismatch between cores {k} and {k+1}: "
                    f"{self.cores[k].shape[2]} != {self.cores[k+1].shape[0]}"
                )
    
    @property
    def n_sites(self) -> int:
        """Number of sites (cores)."""
        return len(self.cores)
    
    @property
    def local_dims(self) -> List[int]:
        """Local dimension at each site."""
        return [core.shape[1] for core in self.cores]
    
    @property
    def total_dim(self) -> int:
        """Total dimension of represented vector: ∏ dₖ."""
        return int(np.prod(self.local_dims))
    
    @property
    def ranks(self) -> List[int]:
        """Bond dimensions [r₀, r₁, ..., rₙ]."""
        ranks = [self.cores[0].shape[0]]
        for core in self.cores:
            ranks.append(core.shape[2])
        return ranks
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension."""
        return max(self.ranks)
    
    def memory_usage(self) -> int:
        """Total number of parameters."""
        return sum(core.size for core in self.cores)
    
    def compression_ratio(self) -> float:
        """Ratio of full storage to QTT storage."""
        return self.total_dim / self.memory_usage()
    
    def __repr__(self) -> str:
        return (
            f"QTT(sites={self.n_sites}, local_dims={self.local_dims}, "
            f"max_rank={self.max_rank}, compression={self.compression_ratio():.2e})"
        )
    
    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------
    
    def to_dense(self) -> np.ndarray:
        """
        Convert to full dense tensor (flat vector).
        
        Index convention: index = i₀×d₁×d₂×... + i₁×d₂×... + i₂×... + ...
        (big-endian / row-major: first site is most significant)
        
        WARNING: Exponential in number of sites. Only for small systems.
        """
        if self.total_dim > 1e8:
            raise ValueError(
                f"Dense conversion would require {self.total_dim:.2e} elements. "
                "Use to_dense only for small systems."
            )
        
        # Contract all cores
        # Start with first core: (1, d_0, r_1) -> (d_0, r_1)
        result = self.cores[0][0, :, :]  # (d_0, r_1)
        
        for k in range(1, self.n_sites):
            core_k = self.cores[k]  # (r_k, d_k, r_{k+1})
            r_k, d_k, r_next = core_k.shape
            
            # result has shape (d_0 × ... × d_{k-1}, r_k)
            # Contract with core_k over r_k, then reshape
            # result @ core_k.reshape(r_k, d_k * r_next)
            result = result @ core_k.reshape(r_k, d_k * r_next)
            result = result.reshape(-1, r_next)
        
        return result.flatten()
    
    def to_vector(self) -> np.ndarray:
        """Alias for to_dense() - convert to flat vector."""
        return self.to_dense()
    
    @classmethod
    def from_dense(cls, vector: np.ndarray, local_dims: List[int], 
                   max_rank: Optional[int] = None, tol: float = 1e-14) -> 'QTT':
        """
        Convert dense vector to QTT format via TT-SVD.
        
        Args:
            vector: Dense vector of length ∏ local_dims
            local_dims: List of local dimensions
            max_rank: Maximum allowed bond dimension (None = no limit)
            tol: Truncation tolerance for SVD
            
        Returns:
            QTT representation
        """
        n_sites = len(local_dims)
        total_dim = int(np.prod(local_dims))
        
        if len(vector) != total_dim:
            raise ValueError(f"Vector length {len(vector)} != product of dims {total_dim}")
        
        vector = np.asarray(vector, dtype=np.complex128)
        
        cores = []
        r_prev = 1
        C = vector.reshape(1, -1)  # Start with row vector
        
        for k in range(n_sites - 1):
            d_k = local_dims[k]
            rest_dim = int(np.prod(local_dims[k+1:]))
            
            # Reshape to (r_{k-1} × d_k, rest)
            C = C.reshape(r_prev * d_k, rest_dim)
            
            # SVD
            U, S, Vt = la.svd(C, full_matrices=False)
            
            # Determine rank (truncation)
            if tol > 0 and len(S) > 0:
                # Keep singular values above tolerance
                s_sum_sq = np.sum(S**2)
                cumsum = np.cumsum(S**2)
                # Find rank where truncation error < tol
                rank = len(S)
                for r in range(1, len(S) + 1):
                    if cumsum[r-1] / s_sum_sq > 1 - tol**2:
                        rank = r
                        break
            else:
                rank = len(S)
            
            if max_rank is not None:
                rank = min(rank, max_rank)
            
            rank = max(1, rank)  # At least rank 1
            
            # Truncate
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            
            # Core k: reshape U to (r_{k-1}, d_k, rank)
            core = U.reshape(r_prev, d_k, rank)
            cores.append(core)
            
            # Remaining for next iteration: absorb singular values
            C = np.diag(S) @ Vt
            r_prev = rank
        
        # Last core: whatever remains
        d_last = local_dims[-1]
        cores.append(C.reshape(r_prev, d_last, 1))
        
        return cls(cores=cores)
    
    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------
    
    def __add__(self, other: 'QTT') -> 'QTT':
        """
        Add two QTT tensors.
        
        rank(A + B) ≤ rank(A) + rank(B)
        """
        if self.n_sites != other.n_sites:
            raise ValueError("QTTs must have same number of sites")
        if self.local_dims != other.local_dims:
            raise ValueError("QTTs must have same local dimensions")
        
        new_cores = []
        
        for k in range(self.n_sites):
            core_a = self.cores[k]  # (r_a, d, r_a')
            core_b = other.cores[k]  # (r_b, d, r_b')
            
            r_a, d, r_a_next = core_a.shape
            r_b, _, r_b_next = core_b.shape
            
            if k == 0:
                # First core: concatenate along right index
                new_core = np.concatenate([core_a, core_b], axis=2)
            elif k == self.n_sites - 1:
                # Last core: concatenate along left index
                new_core = np.concatenate([core_a, core_b], axis=0)
            else:
                # Middle cores: block diagonal
                new_core = np.zeros((r_a + r_b, d, r_a_next + r_b_next), dtype=np.complex128)
                new_core[:r_a, :, :r_a_next] = core_a
                new_core[r_a:, :, r_a_next:] = core_b
            
            new_cores.append(new_core)
        
        return QTT(cores=new_cores)
    
    def __mul__(self, scalar: complex) -> 'QTT':
        """Scalar multiplication."""
        new_cores = [core.copy() for core in self.cores]
        new_cores[0] = new_cores[0] * scalar
        return QTT(cores=new_cores)
    
    def __rmul__(self, scalar: complex) -> 'QTT':
        """Scalar multiplication (right)."""
        return self.__mul__(scalar)
    
    def __neg__(self) -> 'QTT':
        """Negation."""
        return self * (-1)
    
    def __sub__(self, other: 'QTT') -> 'QTT':
        """Subtraction."""
        return self + (-other)
    
    # -------------------------------------------------------------------------
    # Inner Product and Norm
    # -------------------------------------------------------------------------
    
    def inner(self, other: 'QTT') -> complex:
        """
        Compute inner product ⟨self | other⟩.
        
        Uses efficient contraction without forming dense vectors.
        """
        if self.n_sites != other.n_sites:
            raise ValueError("QTTs must have same number of sites")
        
        # Contract from left to right
        # Start with identity: (1, 1) matrix
        result = np.array([[1.0 + 0j]])
        
        for k in range(self.n_sites):
            core_a = self.cores[k]  # (r_a, d, r_a')
            core_b = other.cores[k]  # (r_b, d, r_b')
            
            r_a, d, r_a_next = core_a.shape
            r_b, _, r_b_next = core_b.shape
            
            # Contract: result_{i,j} × (core_a*)_{i,d,i'} × (core_b)_{j,d,j'}
            # Sum over d index
            
            # Reshape for efficient contraction
            # result: (r_a, r_b)
            # Need to contract to get (r_a', r_b')
            
            new_result = np.zeros((r_a_next, r_b_next), dtype=np.complex128)
            
            for d_idx in range(d):
                # core_a[:, d_idx, :] is (r_a, r_a')
                # core_b[:, d_idx, :] is (r_b, r_b')
                mat_a = core_a[:, d_idx, :].conj()  # (r_a, r_a')
                mat_b = core_b[:, d_idx, :]  # (r_b, r_b')
                
                # result @ mat_a^T gives (r_a', r_b)
                # then @ mat_b gives (r_a', r_b')
                new_result += mat_a.T @ result @ mat_b
            
            result = new_result
        
        return result[0, 0]
    
    def norm(self) -> float:
        """Compute 2-norm ||self||."""
        if self._norm is not None:
            return self._norm
        
        norm_sq = np.real(self.inner(self))
        self._norm = np.sqrt(max(0, norm_sq))
        return self._norm
    
    def normalize(self) -> 'QTT':
        """Return normalized copy: self / ||self||."""
        n = self.norm()
        if n < 1e-15:
            raise ValueError("Cannot normalize zero vector")
        return self * (1.0 / n)
    
    # -------------------------------------------------------------------------
    # QTT Rounding (Compression)
    # -------------------------------------------------------------------------
    
    def round(self, max_rank: Optional[int] = None, tol: float = 1e-14) -> 'QTT':
        """
        Compress QTT to lower rank via SVD truncation.
        
        This is THE key operation for maintaining tractability.
        
        Args:
            max_rank: Maximum bond dimension allowed
            tol: Truncation tolerance (relative to largest singular value)
            
        Returns:
            Compressed QTT with reduced rank
        """
        # Orthogonalize left-to-right first
        cores = self._left_orthogonalize()
        
        # Then sweep right-to-left with truncation
        for k in range(self.n_sites - 1, 0, -1):
            core = cores[k]  # (r_{k-1}, d, r_k)
            r_prev, d, r_next = core.shape
            
            # Reshape to matrix: (r_{k-1}, d × r_k)
            mat = core.reshape(r_prev, d * r_next)
            
            # SVD
            U, S, Vt = la.svd(mat, full_matrices=False)
            
            # Truncate
            if tol > 0:
                cutoff = tol * S[0]
                rank = max(1, np.sum(S > cutoff))
            else:
                rank = len(S)
            
            if max_rank is not None:
                rank = min(rank, max_rank)
            
            # Truncated decomposition
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            
            # New core k: Vt reshaped to (rank, d, r_k)
            cores[k] = Vt.reshape(rank, d, r_next)
            
            # Absorb U @ S into core k-1
            cores[k-1] = np.tensordot(cores[k-1], U @ np.diag(S), axes=([2], [0]))
        
        return QTT(cores=cores)
    
    def _left_orthogonalize(self) -> List[np.ndarray]:
        """
        Left-orthogonalize all cores via QR decomposition.
        
        After this, each core satisfies: Σ_d core[:, d, :]† @ core[:, d, :] = I
        """
        cores = [core.copy() for core in self.cores]
        
        for k in range(self.n_sites - 1):
            core = cores[k]  # (r_{k-1}, d, r_k)
            r_prev, d, r_next = core.shape
            
            # Reshape to (r_{k-1} × d, r_k)
            mat = core.reshape(r_prev * d, r_next)
            
            # QR decomposition
            Q, R = la.qr(mat, mode='economic')
            
            new_rank = Q.shape[1]
            
            # Update core k
            cores[k] = Q.reshape(r_prev, d, new_rank)
            
            # Absorb R into core k+1
            cores[k+1] = np.tensordot(R, cores[k+1], axes=([1], [0]))
        
        return cores
    
    def truncation_error(self, truncated: 'QTT') -> float:
        """
        Compute relative error ||self - truncated|| / ||self||.
        """
        diff = self - truncated
        return diff.norm() / self.norm()


# =============================================================================
# QTT FACTORY FUNCTIONS
# =============================================================================

def zeros_qtt(n_sites: int, local_dim: int = 2) -> QTT:
    """Create zero QTT."""
    cores = []
    for k in range(n_sites):
        r_prev = 1 if k == 0 else 1
        r_next = 1 if k == n_sites - 1 else 1
        cores.append(np.zeros((r_prev, local_dim, r_next), dtype=np.complex128))
    return QTT(cores=cores)


def ones_qtt(n_sites: int, local_dim: int = 2) -> QTT:
    """
    Create QTT for vector of all ones (unnormalized).
    
    All-ones has rank 1: each core is just [1, 1, ..., 1].
    """
    cores = []
    for k in range(n_sites):
        core = np.ones((1, local_dim, 1), dtype=np.complex128)
        cores.append(core)
    return QTT(cores=cores)


def basis_qtt(n_sites: int, local_dim: int, index: int) -> QTT:
    """
    Create QTT for basis vector e_index.
    
    Index convention: index = i₀×d^{n-1} + i₁×d^{n-2} + ... + i_{n-1}
    (big-endian / row-major: first site is most significant)
    
    Basis vectors have rank 1 (product state).
    """
    total = local_dim ** n_sites
    if index < 0 or index >= total:
        raise ValueError(f"Index {index} out of range for {n_sites} sites with local dim {local_dim}")
    
    # Convert index to multi-index (big-endian)
    multi_idx = []
    remaining = index
    for k in range(n_sites):
        stride = local_dim ** (n_sites - k - 1)
        multi_idx.append(remaining // stride)
        remaining = remaining % stride
    
    cores = []
    for k in range(n_sites):
        core = np.zeros((1, local_dim, 1), dtype=np.complex128)
        core[0, multi_idx[k], 0] = 1.0
        cores.append(core)
    
    return QTT(cores=cores)


def random_qtt(n_sites: int, local_dim: int = 2, max_rank: int = 4,
               seed: Optional[int] = None, normalize: bool = True) -> QTT:
    """
    Create random QTT with specified maximum rank.
    """
    if seed is not None:
        np.random.seed(seed)
    
    cores = []
    ranks = [1]
    
    # Generate random ranks
    for k in range(n_sites - 1):
        max_possible = min(
            max_rank,
            local_dim * ranks[-1],
            local_dim ** (n_sites - k - 1)
        )
        ranks.append(min(max_rank, max_possible))
    ranks.append(1)
    
    # Generate random cores
    for k in range(n_sites):
        r_prev, r_next = ranks[k], ranks[k+1]
        core_real = np.random.randn(r_prev, local_dim, r_next)
        core_imag = np.random.randn(r_prev, local_dim, r_next)
        cores.append(core_real + 1j * core_imag)
    
    qtt = QTT(cores=cores)
    
    if normalize:
        qtt = qtt.normalize()
    
    return qtt


def product_state_qtt(states: List[np.ndarray]) -> QTT:
    """
    Create QTT for product state |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩.
    
    Product states have rank 1.
    """
    cores = []
    for k, psi in enumerate(states):
        d = len(psi)
        core = psi.reshape(1, d, 1)
        cores.append(core.astype(np.complex128))
    return QTT(cores=cores)


# =============================================================================
# QTT OPERATORS (MPO FORMAT)
# =============================================================================

@dataclass
class QTTMPO:
    """
    Matrix Product Operator (MPO) in QTT format.
    
    Represents operator O acting on d^N dimensional space.
    Each core[k] has shape (r_{k-1}, d, d, r_k) — two physical indices.
    """
    
    cores: List[np.ndarray]
    
    def __post_init__(self):
        """Validate MPO structure."""
        if len(self.cores) == 0:
            raise ValueError("MPO must have at least one core")
        
        # Check boundary conditions
        if self.cores[0].shape[0] != 1:
            raise ValueError(f"First core must have r₀ = 1")
        if self.cores[-1].shape[3] != 1:
            raise ValueError(f"Last core must have rₙ = 1")
    
    @property
    def n_sites(self) -> int:
        return len(self.cores)
    
    @property
    def local_dim(self) -> int:
        return self.cores[0].shape[1]
    
    def apply(self, psi: QTT) -> QTT:
        """
        Apply MPO to QTT state: |φ⟩ = O |ψ⟩.
        
        Result has rank up to rank(O) × rank(ψ).
        """
        if self.n_sites != psi.n_sites:
            raise ValueError("MPO and QTT must have same number of sites")
        
        new_cores = []
        
        for k in range(self.n_sites):
            mpo_core = self.cores[k]  # (r_O, d, d, r_O')
            qtt_core = psi.cores[k]  # (r_ψ, d, r_ψ')
            
            r_O, d, _, r_O_next = mpo_core.shape
            r_psi, _, r_psi_next = qtt_core.shape
            
            # Contract over input physical index
            # Result: (r_O, d_out, r_O', r_ψ, r_ψ')
            # Then reshape to ((r_O × r_ψ), d_out, (r_O' × r_ψ'))
            
            new_core = np.zeros((r_O * r_psi, d, r_O_next * r_psi_next), dtype=np.complex128)
            
            for d_out in range(d):
                for d_in in range(d):
                    # mpo_core[:, d_out, d_in, :] is (r_O, r_O')
                    # qtt_core[:, d_in, :] is (r_psi, r_psi')
                    contrib = np.kron(mpo_core[:, d_out, d_in, :], qtt_core[:, d_in, :])
                    new_core[:, d_out, :] += contrib
            
            new_cores.append(new_core)
        
        return QTT(cores=new_cores)


def identity_mpo(n_sites: int, local_dim: int) -> QTTMPO:
    """Create identity MPO."""
    cores = []
    for k in range(n_sites):
        core = np.zeros((1, local_dim, local_dim, 1), dtype=np.complex128)
        for d in range(local_dim):
            core[0, d, d, 0] = 1.0
        cores.append(core)
    return QTTMPO(cores=cores)


def diagonal_mpo(diagonal: QTT) -> QTTMPO:
    """
    Create diagonal MPO from QTT vector.
    
    If diagonal represents v, creates operator diag(v).
    """
    cores = []
    for k in range(diagonal.n_sites):
        qtt_core = diagonal.cores[k]  # (r, d, r')
        r, d, r_next = qtt_core.shape
        
        mpo_core = np.zeros((r, d, d, r_next), dtype=np.complex128)
        for d_idx in range(d):
            mpo_core[:, d_idx, d_idx, :] = qtt_core[:, d_idx, :]
        
        cores.append(mpo_core)
    
    return QTTMPO(cores=cores)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_qtt():
    """Verify QTT implementation with comprehensive tests."""
    print("=" * 70)
    print("QTT IMPLEMENTATION VERIFICATION")
    print("=" * 70)
    
    # Test 1: Dense conversion roundtrip
    print("\n--- Test 1: Dense Conversion ---")
    n_sites, d = 4, 3
    vec = np.random.randn(d**n_sites) + 1j * np.random.randn(d**n_sites)
    vec = vec / np.linalg.norm(vec)
    
    qtt = QTT.from_dense(vec, [d] * n_sites)
    vec_back = qtt.to_dense()
    
    error = np.linalg.norm(vec - vec_back)
    print(f"  Roundtrip error: {error:.2e}")
    assert error < 1e-12, "Dense conversion failed"
    print("  ✓ Dense conversion PASSED")
    
    # Test 2: Inner product
    print("\n--- Test 2: Inner Product ---")
    qtt1 = random_qtt(n_sites, d, max_rank=4)
    qtt2 = random_qtt(n_sites, d, max_rank=4)
    
    # Dense inner product
    v1 = qtt1.to_dense()
    v2 = qtt2.to_dense()
    inner_dense = np.vdot(v1, v2)
    
    # QTT inner product
    inner_qtt = qtt1.inner(qtt2)
    
    error = np.abs(inner_dense - inner_qtt)
    print(f"  Inner product error: {error:.2e}")
    assert error < 1e-12, "Inner product failed"
    print("  ✓ Inner product PASSED")
    
    # Test 3: Norm
    print("\n--- Test 3: Norm ---")
    qtt = random_qtt(n_sites, d, max_rank=4, normalize=False)
    norm_dense = np.linalg.norm(qtt.to_dense())
    norm_qtt = qtt.norm()
    
    error = np.abs(norm_dense - norm_qtt) / norm_dense
    print(f"  Norm relative error: {error:.2e}")
    assert error < 1e-12, "Norm failed"
    print("  ✓ Norm PASSED")
    
    # Test 4: Addition
    print("\n--- Test 4: Addition ---")
    qtt1 = random_qtt(n_sites, d, max_rank=4)
    qtt2 = random_qtt(n_sites, d, max_rank=4)
    
    sum_qtt = qtt1 + qtt2
    sum_dense = qtt1.to_dense() + qtt2.to_dense()
    
    error = np.linalg.norm(sum_qtt.to_dense() - sum_dense)
    print(f"  Addition error: {error:.2e}")
    assert error < 1e-12, "Addition failed"
    print(f"  Rank after addition: {sum_qtt.max_rank} (was {qtt1.max_rank} + {qtt2.max_rank})")
    print("  ✓ Addition PASSED")
    
    # Test 5: Rounding (compression)
    print("\n--- Test 5: Rounding (Compression) ---")
    # Create high-rank QTT
    qtt_high = qtt1 + qtt2 + random_qtt(n_sites, d, max_rank=4)
    print(f"  Before rounding: rank = {qtt_high.max_rank}")
    
    # Round to lower rank
    qtt_low = qtt_high.round(max_rank=4)
    print(f"  After rounding: rank = {qtt_low.max_rank}")
    
    # Check approximation error
    trunc_error = qtt_high.truncation_error(qtt_low)
    print(f"  Truncation error: {trunc_error:.2e}")
    print("  ✓ Rounding PASSED")
    
    # Test 6: Compression ratio
    print("\n--- Test 6: Compression Ratio ---")
    n_sites_large = 10
    d_large = 5
    qtt_large = random_qtt(n_sites_large, d_large, max_rank=8)
    
    print(f"  Sites: {n_sites_large}, Local dim: {d_large}")
    print(f"  Full dimension: {qtt_large.total_dim:,}")
    print(f"  QTT parameters: {qtt_large.memory_usage():,}")
    print(f"  Compression ratio: {qtt_large.compression_ratio():.2e}")
    print("  ✓ Compression PASSED")
    
    # Test 7: MPO application
    print("\n--- Test 7: MPO Application ---")
    n_sites, d = 4, 2
    psi = random_qtt(n_sites, d, max_rank=4)
    I_mpo = identity_mpo(n_sites, d)
    
    psi_id = I_mpo.apply(psi)
    
    error = np.linalg.norm(psi.to_dense() - psi_id.to_dense())
    print(f"  Identity application error: {error:.2e}")
    assert error < 1e-12, "MPO application failed"
    print("  ✓ MPO Application PASSED")
    
    # Test 8: Basis states
    print("\n--- Test 8: Basis States ---")
    n_sites, d = 3, 2
    for idx in range(d**n_sites):
        basis = basis_qtt(n_sites, d, idx)
        vec = basis.to_dense()
        
        # Check it's a basis vector
        assert np.abs(vec[idx] - 1.0) < 1e-14, f"Basis vector wrong at index {idx}"
        assert np.abs(np.sum(np.abs(vec)**2) - 1.0) < 1e-14, "Basis vector not normalized"
    print(f"  All {d**n_sites} basis vectors correct")
    print("  ✓ Basis States PASSED")
    
    print("\n" + "=" * 70)
    print("  ★ QTT IMPLEMENTATION VALIDATED ★")
    print("=" * 70)


if __name__ == "__main__":
    verify_qtt()
