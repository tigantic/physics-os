"""
QTT Tropical Matrix Operations

Tropical matrix multiplication is the key to shortest path algorithms:
    (A ⊗ B)_ij = min_k (A_ik + B_kj)

For QTT matrices, we use smooth approximations and TT contractions
to achieve O(r³ log² N) complexity.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import torch

from tensornet.genesis.tropical.semiring import (
    TropicalSemiring, MinPlusSemiring, MaxPlusSemiring,
    SemiringType, softmin, softmax
)


@dataclass
class TropicalMatrix:
    """
    A tropical matrix with QTT compression support.
    
    For Phase 1, matrices are stored densely. Future versions will
    use full QTT-MPO format.
    
    Attributes:
        data: Dense matrix data
        semiring: The tropical semiring (min-plus or max-plus)
        size: Matrix dimension (assumes square)
        cores: Optional TT-cores for QTT representation
        rank: Maximum TT-rank
    """
    data: torch.Tensor
    semiring: TropicalSemiring = field(default_factory=lambda: MinPlusSemiring)
    size: int = 0
    cores: Optional[List[torch.Tensor]] = None
    rank: int = 10
    
    def __post_init__(self):
        if self.size == 0:
            self.size = self.data.shape[0]
    
    @property
    def inf(self) -> float:
        """Infinity value for this semiring."""
        return self.semiring.zero
    
    def to_dense(self) -> torch.Tensor:
        """Return dense representation."""
        return self.data
    
    @classmethod
    def zeros(cls, size: int, 
              semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalMatrix':
        """
        Create tropical zero matrix (all infinity).
        
        For min-plus: all +∞
        For max-plus: all -∞
        """
        data = torch.full((size, size), semiring.zero)
        return cls(data=data, semiring=semiring, size=size)
    
    @classmethod
    def identity(cls, size: int,
                 semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalMatrix':
        """
        Create tropical identity matrix.
        
        Diagonal = 0 (multiplicative identity)
        Off-diagonal = ∞ (additive identity)
        """
        data = torch.full((size, size), semiring.zero)
        for i in range(size):
            data[i, i] = semiring.one  # = 0
        return cls(data=data, semiring=semiring, size=size)
    
    @classmethod
    def from_adjacency(cls, adj: torch.Tensor,
                       semiring: TropicalSemiring = MinPlusSemiring,
                       no_edge: float = None) -> 'TropicalMatrix':
        """
        Create tropical matrix from adjacency/weight matrix.
        
        Args:
            adj: Adjacency matrix (weights or 0/1)
            semiring: Tropical semiring
            no_edge: Value to use for no edge (default: ∞)
            
        Returns:
            TropicalMatrix
        """
        if no_edge is None:
            no_edge = semiring.zero
        
        data = adj.clone().float()
        # Replace zeros (no edge) with infinity
        if no_edge != 0:
            data[data == 0] = no_edge
        
        return cls(data=data, semiring=semiring, size=adj.shape[0])
    
    @classmethod
    def grid_graph(cls, grid_size: int, 
                   rank: int = 10,
                   semiring: TropicalSemiring = MinPlusSemiring,
                   adjacency: bool = False) -> 'TropicalMatrix':
        """
        Create matrix for grid/chain graph.
        
        Args:
            grid_size: Number of nodes
            rank: Target TT-rank
            semiring: Tropical semiring
            adjacency: If True, return adjacency matrix (with ∞ for non-edges).
                      If False, return full distance matrix.
            
        Returns:
            TropicalMatrix
        """
        if grid_size > 2**14:
            actual_size = min(grid_size, 2**12)
        else:
            actual_size = grid_size
        
        if adjacency:
            # Adjacency matrix: only neighbors have finite weights
            data = torch.full((actual_size, actual_size), semiring.zero)
            for i in range(actual_size):
                data[i, i] = 0.0  # Self-loop with weight 0
                if i > 0:
                    data[i, i-1] = 1.0  # Edge to left neighbor
                if i < actual_size - 1:
                    data[i, i+1] = 1.0  # Edge to right neighbor
        else:
            # Full distance matrix: d[i,j] = |i - j|
            i_idx = torch.arange(actual_size).float().unsqueeze(1)
            j_idx = torch.arange(actual_size).float().unsqueeze(0)
            data = torch.abs(i_idx - j_idx)
        
        return cls(data=data, semiring=semiring, size=actual_size, rank=rank)
    
    @classmethod
    def chain_adjacency(cls, n: int,
                        semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalMatrix':
        """
        Create adjacency matrix for 1D chain graph (path graph).
        
        Node i is connected to i-1 and i+1 with weight 1.
        Non-adjacent nodes have ∞ weight.
        
        Args:
            n: Number of nodes
            semiring: Tropical semiring
            
        Returns:
            TropicalMatrix adjacency matrix
        """
        return cls.grid_graph(n, adjacency=True, semiring=semiring)
    
    @classmethod
    def random(cls, size: int, density: float = 0.3,
               max_weight: float = 10.0,
               semiring: TropicalSemiring = MinPlusSemiring,
               seed: Optional[int] = None) -> 'TropicalMatrix':
        """
        Create random tropical matrix (sparse random graph weights).
        
        Args:
            size: Matrix dimension
            density: Fraction of non-infinity entries
            max_weight: Maximum edge weight
            semiring: Tropical semiring
            seed: Random seed
            
        Returns:
            Random TropicalMatrix
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Start with all infinity
        data = torch.full((size, size), semiring.zero)
        
        # Add random edges
        mask = torch.rand(size, size) < density
        weights = torch.rand(size, size) * max_weight
        data[mask] = weights[mask]
        
        # Zero diagonal (self-loops have zero cost)
        for i in range(size):
            data[i, i] = 0.0
        
        return cls(data=data, semiring=semiring, size=size)
    
    def __matmul__(self, other: 'TropicalMatrix') -> 'TropicalMatrix':
        """Tropical matrix multiplication."""
        return tropical_matmul(self, other)
    
    def __add__(self, other: 'TropicalMatrix') -> 'TropicalMatrix':
        """Tropical addition (elementwise min/max)."""
        result_data = self.semiring.add(self.data, other.data)
        return TropicalMatrix(data=result_data, semiring=self.semiring, size=self.size)
    
    def power(self, k: int) -> 'TropicalMatrix':
        """Compute A^⊗k using repeated squaring."""
        return tropical_power(self, k)


def tropical_matmul(A: TropicalMatrix, B: TropicalMatrix) -> TropicalMatrix:
    """
    Tropical matrix multiplication.
    
    (A ⊗ B)_ij = ⊕_k (A_ik ⊗ B_kj)
    
    For min-plus: (A ⊗ B)_ij = min_k (A_ik + B_kj)
    For max-plus: (A ⊗ B)_ij = max_k (A_ik + B_kj)
    
    Args:
        A: Left matrix
        B: Right matrix
        
    Returns:
        Tropical product A ⊗ B
    """
    assert A.size == B.size, "Matrix dimensions must match"
    semiring = A.semiring
    n = A.size
    
    # Dense implementation for Phase 1
    # C_ij = ⊕_k (A_ik + B_kj)
    
    # Expand for broadcasting: A[i,k] + B[k,j]
    # A: (n, k, 1), B: (1, k, n) → sum: (n, k, n)
    A_exp = A.data.unsqueeze(2)  # (n, k, 1)
    B_exp = B.data.unsqueeze(0)  # (1, k, n)
    
    # Tropical multiplication: classical addition
    products = A_exp + B_exp  # (n, k, n)
    
    # Tropical sum over k: min or max
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        result = products.min(dim=1)[0]  # (n, n)
    else:
        result = products.max(dim=1)[0]  # (n, n)
    
    return TropicalMatrix(data=result, semiring=semiring, size=n)


def tropical_power(A: TropicalMatrix, k: int) -> TropicalMatrix:
    """
    Compute A^⊗k using repeated squaring.
    
    (A^⊗k)_ij = length of shortest path from i to j using exactly k edges.
    
    Uses O(log k) tropical matrix multiplications.
    
    Args:
        A: Tropical matrix
        k: Exponent
        
    Returns:
        A^⊗k
    """
    if k <= 0:
        return TropicalMatrix.identity(A.size, A.semiring)
    
    if k == 1:
        return TropicalMatrix(data=A.data.clone(), semiring=A.semiring, size=A.size)
    
    # Repeated squaring
    result = TropicalMatrix.identity(A.size, A.semiring)
    base = TropicalMatrix(data=A.data.clone(), semiring=A.semiring, size=A.size)
    
    while k > 0:
        if k % 2 == 1:
            result = tropical_matmul(result, base)
        base = tropical_matmul(base, base)
        k //= 2
    
    return result


def tropical_kleene_star(A: TropicalMatrix, 
                         max_iter: Optional[int] = None) -> TropicalMatrix:
    """
    Compute the Kleene star (closure) of a tropical matrix.
    
    A* = I ⊕ A ⊕ A² ⊕ A³ ⊕ ...
    
    For min-plus, (A*)_ij = shortest path length from i to j.
    
    Converges in at most n iterations for an n×n matrix without
    negative cycles.
    
    Args:
        A: Tropical matrix (typically adjacency weights)
        max_iter: Maximum iterations (default: n)
        
    Returns:
        Kleene star A*
    """
    n = A.size
    if max_iter is None:
        max_iter = n
    
    # Floyd-Warshall style iteration
    # D^(k+1)_ij = D^(k)_ij ⊕ (D^(k)_ik ⊗ D^(k)_kj)
    
    D = A.data.clone()
    
    # Initialize diagonal to 0 (identity paths)
    for i in range(n):
        D[i, i] = 0.0
    
    semiring = A.semiring
    
    for k in range(min(n, max_iter)):
        # D_ij = D_ij ⊕ (D_ik + D_kj)
        D_ik = D[:, k:k+1]  # (n, 1)
        D_kj = D[k:k+1, :]  # (1, n)
        
        update = D_ik + D_kj  # (n, n) via broadcasting
        
        if semiring.semiring_type == SemiringType.MIN_PLUS:
            D = torch.minimum(D, update)
        else:
            D = torch.maximum(D, update)
    
    return TropicalMatrix(data=D, semiring=semiring, size=n)


def tropical_transpose(A: TropicalMatrix) -> TropicalMatrix:
    """Transpose a tropical matrix."""
    return TropicalMatrix(
        data=A.data.T,
        semiring=A.semiring,
        size=A.size
    )


def tropical_hadamard(A: TropicalMatrix, B: TropicalMatrix) -> TropicalMatrix:
    """
    Tropical Hadamard product (elementwise tropical multiplication).
    
    Since tropical multiplication is classical addition:
    (A ⊙ B)_ij = A_ij + B_ij
    """
    result = A.data + B.data
    return TropicalMatrix(data=result, semiring=A.semiring, size=A.size)


def has_negative_cycle(A: TropicalMatrix) -> bool:
    """
    Check if matrix has a negative cycle.
    
    A negative cycle exists if any diagonal of A* is negative.
    
    Args:
        A: Tropical matrix (min-plus semiring)
        
    Returns:
        True if negative cycle exists
    """
    if A.semiring.semiring_type != SemiringType.MIN_PLUS:
        raise ValueError("Negative cycle detection only for min-plus")
    
    closure = tropical_kleene_star(A)
    
    # Check diagonal
    diag = torch.diag(closure.data)
    return (diag < -1e-9).any().item()


def tropical_eigenvalue(A: TropicalMatrix) -> float:
    """
    Compute the tropical eigenvalue (max cycle mean / min cycle mean).
    
    For min-plus: λ = min over cycles c of (weight(c) / length(c))
    For max-plus: λ = max over cycles c of (weight(c) / length(c))
    
    This is related to the asymptotic behavior: A^n → n·λ·J + bounded
    
    Uses Karp's algorithm for the minimum/maximum mean cycle.
    
    Args:
        A: Square tropical matrix
        
    Returns:
        Tropical eigenvalue
    """
    n = A.size
    
    if A.semiring.semiring_type == SemiringType.MAX_PLUS:
        # Max-plus eigenvalue: maximum cycle mean
        # Uses dual Karp's algorithm
        # λ = min_j max_k (D^(n)_0j - D^(k)_0j) / (n - k)
        
        D_k = [None] * (n + 1)
        D_k[0] = torch.full((n, n), A.semiring.zero, dtype=A.data.dtype, device=A.data.device)
        for i in range(n):
            D_k[0][i, i] = 0.0
        
        for k in range(1, n + 1):
            D_k[k] = tropical_matmul(
                TropicalMatrix(D_k[k-1], A.semiring, n), A
            ).data
        
        # For max-plus, we want the maximum mean cycle
        # λ = min_j max_k (D^n_0j - D^k_0j) / (n - k)
        lambda_est = float('inf')
        
        for j in range(n):
            d_n = D_k[n][0, j].item()
            if d_n <= A.semiring.zero + 1:
                continue  # Unreachable
            
            max_avg = -float('inf')
            for k in range(n):
                d_k = D_k[k][0, j].item()
                if d_k <= A.semiring.zero + 1:
                    continue
                
                avg = (d_n - d_k) / (n - k)
                max_avg = max(max_avg, avg)
            
            if max_avg > -float('inf'):
                lambda_est = min(lambda_est, max_avg)
        
        return lambda_est if lambda_est < float('inf') else 0.0
    
    # Min-plus eigenvalue (original implementation)
    
    # Karp's algorithm: compute D^(k) for k = 0, ..., n
    # λ = max_j min_k (D^(n)_ij - D^(k)_ij) / (n - k)
    
    D_k = [None] * (n + 1)
    D_k[0] = torch.full((n, n), A.semiring.zero)
    for i in range(n):
        D_k[0][i, i] = 0.0
    
    for k in range(1, n + 1):
        D_k[k] = tropical_matmul(
            TropicalMatrix(D_k[k-1], A.semiring, n), A
        ).data
    
    # For each node j, compute max over k of (D^n_0j - D^k_0j) / (n-k)
    lambda_est = -float('inf')
    
    for j in range(n):
        d_n = D_k[n][0, j].item()
        if d_n >= A.semiring.zero - 1:
            continue  # Unreachable
        
        min_avg = float('inf')
        for k in range(n):
            d_k = D_k[k][0, j].item()
            if d_k >= A.semiring.zero - 1:
                continue
            
            avg = (d_n - d_k) / (n - k)
            min_avg = min(min_avg, avg)
        
        if min_avg < float('inf'):
            lambda_est = max(lambda_est, min_avg)
    
    return lambda_est


def check_tropical_properties(A: TropicalMatrix, tol: float = 1e-6) -> dict:
    """
    Check various tropical matrix properties.
    
    Returns dict with:
        - is_finite: No ±∞ entries
        - is_symmetric: A = A^T
        - has_zero_diagonal: All A_ii = 0
        - is_metric: Triangle inequality holds
    """
    data = A.data
    n = A.size
    inf = A.semiring.zero
    
    results = {}
    
    # Finite check
    results['is_finite'] = not torch.any(
        (data >= inf - 1) | (data <= -inf + 1)
    ).item()
    
    # Symmetry
    results['is_symmetric'] = torch.allclose(data, data.T, atol=tol)
    
    # Zero diagonal
    results['has_zero_diagonal'] = torch.allclose(
        torch.diag(data), torch.zeros(n), atol=tol
    )
    
    # Triangle inequality (for min-plus = metric)
    if A.semiring.semiring_type == SemiringType.MIN_PLUS:
        violations = 0
        for i in range(min(n, 100)):  # Sample
            for j in range(min(n, 100)):
                for k in range(min(n, 50)):
                    if data[i, j] > data[i, k] + data[k, j] + tol:
                        violations += 1
        results['is_metric'] = violations == 0
    else:
        results['is_metric'] = None
    
    return results
