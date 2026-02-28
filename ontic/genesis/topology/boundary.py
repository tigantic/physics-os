"""
Boundary Operators for Homology Computation

Implements boundary matrices ∂_k: C_k → C_{k-1} in sparse format
with optional QTT compression.

The boundary of a k-simplex [v_0, ..., v_k] is:
∂[v_0, ..., v_k] = Σ_{i=0}^k (-1)^i [v_0, ..., v̂_i, ..., v_k]

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import torch

from ontic.genesis.core.rsvd import rsvd_gpu
from .simplicial import Simplex, SimplicialComplex


def boundary_matrix(complex: SimplicialComplex, k: int) -> torch.Tensor:
    """
    Construct the k-th boundary matrix ∂_k: C_k → C_{k-1}.
    
    The boundary matrix has:
    - Rows indexed by (k-1)-simplices
    - Columns indexed by k-simplices
    - Entry (face, simplex) = boundary coefficient
    
    Args:
        complex: Simplicial complex
        k: Dimension (must be ≥ 1)
        
    Returns:
        Boundary matrix, shape (n_{k-1}, n_k)
    """
    if k < 1:
        raise ValueError("k must be >= 1 for boundary matrix")
    
    k_simplices = complex.simplices_of_dim(k)
    k_minus_1_simplices = complex.simplices_of_dim(k - 1)
    
    n_rows = len(k_minus_1_simplices)
    n_cols = len(k_simplices)
    
    if n_rows == 0 or n_cols == 0:
        return torch.zeros(max(n_rows, 1), max(n_cols, 1))
    
    # Create index mappings
    face_to_idx = {s: i for i, s in enumerate(k_minus_1_simplices)}
    
    # Build matrix
    D = torch.zeros(n_rows, n_cols)
    
    for j, simplex in enumerate(k_simplices):
        for face in simplex.faces():
            if face in face_to_idx:
                i = face_to_idx[face]
                coeff = simplex.boundary_coefficient(face)
                D[i, j] = coeff
    
    return D


def boundary_matrix_sparse(complex: SimplicialComplex, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct sparse boundary matrix in COO format.
    
    Args:
        complex: Simplicial complex
        k: Dimension
        
    Returns:
        (indices, values, shape) for sparse tensor
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    
    k_simplices = complex.simplices_of_dim(k)
    k_minus_1_simplices = complex.simplices_of_dim(k - 1)
    
    face_to_idx = {s: i for i, s in enumerate(k_minus_1_simplices)}
    
    rows = []
    cols = []
    vals = []
    
    for j, simplex in enumerate(k_simplices):
        for face in simplex.faces():
            if face in face_to_idx:
                rows.append(face_to_idx[face])
                cols.append(j)
                vals.append(float(simplex.boundary_coefficient(face)))
    
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals)
    shape = torch.Size([len(k_minus_1_simplices), len(k_simplices)])
    
    return indices, values, shape


def coboundary_matrix(complex: SimplicialComplex, k: int) -> torch.Tensor:
    """
    Construct the k-th coboundary matrix δ^k: C^k → C^{k+1}.
    
    The coboundary is the transpose of the boundary: δ^k = (∂_{k+1})^T
    
    Args:
        complex: Simplicial complex
        k: Dimension
        
    Returns:
        Coboundary matrix
    """
    return boundary_matrix(complex, k + 1).T


@dataclass
class QTTBoundaryOperator:
    """
    QTT-compressed boundary operator.
    
    For structured complexes (lattice, grid-like), the boundary matrix
    has low TT rank due to the local structure of simplicial boundaries.
    
    Attributes:
        complex: The simplicial complex
        dimension: Which boundary operator ∂_k
        cores: TT cores representing the boundary matrix
        row_simplices: (k-1)-simplices (row indices)
        col_simplices: k-simplices (column indices)
    """
    complex: SimplicialComplex
    dimension: int
    cores: List[torch.Tensor] = field(default_factory=list)
    row_simplices: List[Simplex] = field(default_factory=list)
    col_simplices: List[Simplex] = field(default_factory=list)
    
    @classmethod
    def from_complex(cls, complex: SimplicialComplex, k: int,
                     max_rank: int = 50) -> 'QTTBoundaryOperator':
        """
        Build QTT boundary operator from simplicial complex.
        
        Args:
            complex: Simplicial complex
            k: Dimension for ∂_k
            max_rank: Maximum TT rank
            
        Returns:
            QTT boundary operator
        """
        row_simplices = complex.simplices_of_dim(k - 1)
        col_simplices = complex.simplices_of_dim(k)
        
        # Build dense matrix first (for compression)
        D = boundary_matrix(complex, k)
        
        # TT-SVD compression
        cores = cls._tt_svd(D, max_rank)
        
        return cls(
            complex=complex,
            dimension=k,
            cores=cores,
            row_simplices=row_simplices,
            col_simplices=col_simplices
        )
    
    @staticmethod
    def _tt_svd(matrix: torch.Tensor, max_rank: int) -> List[torch.Tensor]:
        """
        TT-SVD decomposition of a matrix.
        
        For true QTT, would reshape to 2x2x...x2 tensor first.
        Here we do standard TT for rectangular matrices.
        """
        m, n = matrix.shape
        
        # Simple 2-core TT: matrix as outer product approximation via randomized SVD
        U, S, Vh = rsvd_gpu(matrix, k=max_rank, tol=1e-10)
        rank = S.shape[0]
        
        # Core 1: (1, m, r)
        core1 = (U * S).unsqueeze(0)
        
        # Core 2: (r, n, 1)  
        core2 = Vh.unsqueeze(-1)
        
        return [core1, core2]
    
    def to_dense(self) -> torch.Tensor:
        """Reconstruct dense boundary matrix."""
        if not self.cores:
            return torch.zeros(len(self.row_simplices), len(self.col_simplices))
        
        # For 2-core TT decomposition:
        # core1: (1, m, r), core2: (r, n, 1)
        # Result: (m, n) = core1[0, :, :] @ core2[:, :, 0]
        core1 = self.cores[0].squeeze(0)  # (m, r)
        core2 = self.cores[1].squeeze(-1)  # (r, n)
        
        return core1 @ core2  # (m, n)
    
    def apply(self, chain: torch.Tensor) -> torch.Tensor:
        """
        Apply boundary operator to a chain.
        
        Args:
            chain: Coefficients for k-simplices, shape (n_k,)
            
        Returns:
            Boundary chain, shape (n_{k-1},)
        """
        D = self.to_dense()
        return D @ chain
    
    @property
    def ranks(self) -> List[int]:
        """TT ranks."""
        return [c.shape[-1] for c in self.cores[:-1]] if self.cores else []


def chain_complex_matrices(complex: SimplicialComplex) -> Dict[int, torch.Tensor]:
    """
    Build all boundary matrices for the chain complex.
    
    Returns dictionary mapping k -> ∂_k matrix.
    
    Args:
        complex: Simplicial complex
        
    Returns:
        {k: ∂_k} for k = 1, ..., max_dim
    """
    matrices = {}
    for k in range(1, complex.max_dim + 1):
        matrices[k] = boundary_matrix(complex, k)
    return matrices


def verify_boundary_squared_zero(complex: SimplicialComplex) -> Tuple[bool, str]:
    """
    Verify ∂_{k-1} ∘ ∂_k = 0 for all k.
    
    This is the fundamental property of a chain complex.
    
    Args:
        complex: Simplicial complex
        
    Returns:
        (passed, message)
    """
    for k in range(2, complex.max_dim + 1):
        D_k = boundary_matrix(complex, k)
        D_k_minus_1 = boundary_matrix(complex, k - 1)
        
        # Check ∂_{k-1} ∘ ∂_k = 0
        product = D_k_minus_1 @ D_k
        error = product.abs().max().item()
        
        if error > 1e-10:
            return False, f"∂_{k-1} ∘ ∂_k ≠ 0 (error={error:.2e})"
    
    return True, "∂² = 0 verified"


def betti_numbers_from_boundary(complex: SimplicialComplex) -> List[int]:
    """
    Compute Betti numbers from boundary matrices.
    
    β_k = dim(ker ∂_k) - dim(im ∂_{k+1})
        = n_k - rank(∂_k) - rank(∂_{k+1})
    
    Args:
        complex: Simplicial complex
        
    Returns:
        List of Betti numbers [β_0, β_1, ...]
    """
    betti = []
    
    for k in range(complex.max_dim + 1):
        n_k = complex.num_simplices(k)
        
        # Rank of ∂_k (0 for k=0)
        if k == 0:
            rank_k = 0
        else:
            D_k = boundary_matrix(complex, k)
            rank_k = torch.linalg.matrix_rank(D_k).item()
        
        # Rank of ∂_{k+1}
        if k == complex.max_dim:
            rank_k_plus_1 = 0
        else:
            D_k_plus_1 = boundary_matrix(complex, k + 1)
            rank_k_plus_1 = torch.linalg.matrix_rank(D_k_plus_1).item()
        
        beta_k = n_k - rank_k - rank_k_plus_1
        betti.append(beta_k)
    
    return betti
