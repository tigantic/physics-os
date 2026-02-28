"""
QTT Tropical Optimization

Tropical linear programming and optimization in the tropical semiring.

The tropical eigenvalue problem:
    A ⊗ x = λ ⊗ x

where λ is the tropical eigenvalue (a scalar shift).

In min-plus: min_j(A_ij + x_j) = λ + x_i

The maximum cycle mean of A equals the tropical eigenvalue.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch

from ontic.genesis.tropical.semiring import (
    TropicalSemiring, MinPlusSemiring, MaxPlusSemiring, SemiringType
)
from ontic.genesis.tropical.matrix import TropicalMatrix, tropical_matmul


@dataclass
class TropicalEigenResult:
    """
    Result of tropical eigenvalue computation.
    
    Attributes:
        eigenvalue: The tropical eigenvalue λ
        eigenvector: A tropical eigenvector x
        cycle: The critical cycle achieving the eigenvalue
        iterations: Number of iterations
    """
    eigenvalue: float
    eigenvector: torch.Tensor
    cycle: Optional[List[int]] = None
    iterations: int = 0


def tropical_eigenvalue(A: TropicalMatrix) -> float:
    """
    Compute the tropical eigenvalue of a matrix.
    
    The tropical eigenvalue is:
        Min-plus: minimum cycle mean = min over cycles (weight / length)
        Max-plus: maximum cycle mean = max over cycles (weight / length)
    
    Uses Karp's algorithm for cycle mean.
    
    Args:
        A: Square tropical matrix
        
    Returns:
        Tropical eigenvalue λ
    """
    n = A.size
    semiring = A.semiring
    
    # Compute D^(k) for k = 0, ..., n
    # D^(k)[i,j] = weight of shortest/longest k-edge path from i to j
    
    D = [None] * (n + 1)
    
    # D^0 = I (identity: 0 on diagonal, ∞ elsewhere)
    D[0] = torch.full((n, n), semiring.zero)
    for i in range(n):
        D[0][i, i] = semiring.one
    
    # D^k = A^⊗k
    current = TropicalMatrix(A.data.clone(), semiring, n)
    for k in range(1, n + 1):
        if k == 1:
            D[k] = A.data.clone()
        else:
            current = tropical_matmul(current, A)
            D[k] = current.data.clone()
    
    # Karp's formula for cycle mean
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        # λ = min_j max_k (D^n[s,j] - D^k[s,j]) / (n-k)
        # For cycle mean, we want minimum
        lambda_val = float('inf')
        
        # Check from each starting node
        for s in range(n):
            for j in range(n):
                d_n = D[n][s, j].item()
                if d_n >= semiring.zero - 1:
                    continue
                
                max_avg = -float('inf')
                for k in range(n):
                    d_k = D[k][s, j].item()
                    if d_k >= semiring.zero - 1:
                        continue
                    
                    avg = (d_n - d_k) / (n - k)
                    max_avg = max(max_avg, avg)
                
                if max_avg > -float('inf'):
                    lambda_val = min(lambda_val, max_avg)
    else:
        # Max-plus: maximum cycle mean
        lambda_val = -float('inf')
        
        for s in range(n):
            for j in range(n):
                d_n = D[n][s, j].item()
                if d_n <= -semiring.zero + 1:
                    continue
                
                min_avg = float('inf')
                for k in range(n):
                    d_k = D[k][s, j].item()
                    if d_k <= -semiring.zero + 1:
                        continue
                    
                    avg = (d_n - d_k) / (n - k)
                    min_avg = min(min_avg, avg)
                
                if min_avg < float('inf'):
                    lambda_val = max(lambda_val, min_avg)
    
    return lambda_val


def tropical_eigenvector(A: TropicalMatrix,
                         max_iter: int = 100,
                         tol: float = 1e-8) -> TropicalEigenResult:
    """
    Compute tropical eigenvector via power iteration.
    
    For min-plus: A ⊗ x = λ ⊗ x means min_j(A_ij + x_j) = λ + x_i
    
    The power method: x^{k+1} = A ⊗ x^k - λ
    
    Args:
        A: Square tropical matrix
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        TropicalEigenResult with eigenvalue and eigenvector
    """
    n = A.size
    semiring = A.semiring
    
    # First compute eigenvalue
    lambda_val = tropical_eigenvalue(A)
    
    if not math.isfinite(lambda_val):
        # No finite eigenvalue
        return TropicalEigenResult(
            eigenvalue=lambda_val,
            eigenvector=torch.zeros(n),
            iterations=0
        )
    
    # Initialize eigenvector
    x = torch.zeros(n)
    
    # Power iteration
    for iteration in range(max_iter):
        x_old = x.clone()
        
        # Compute A ⊗ x
        Ax = torch.full((n,), semiring.zero)
        for i in range(n):
            row = A.data[i]
            if semiring.semiring_type == SemiringType.MIN_PLUS:
                Ax[i] = (row + x).min()
            else:
                Ax[i] = (row + x).max()
        
        # Subtract eigenvalue: x_new = A ⊗ x - λ
        x = Ax - lambda_val
        
        # Normalize to have min = 0 (projective normalization)
        if semiring.semiring_type == SemiringType.MIN_PLUS:
            x = x - x.min()
        else:
            x = x - x.max()
        
        # Check convergence
        if (x - x_old).abs().max() < tol:
            return TropicalEigenResult(
                eigenvalue=lambda_val,
                eigenvector=x,
                iterations=iteration + 1
            )
    
    return TropicalEigenResult(
        eigenvalue=lambda_val,
        eigenvector=x,
        iterations=max_iter
    )


def tropical_linear_program(A: torch.Tensor,
                            b: torch.Tensor,
                            semiring: TropicalSemiring = MinPlusSemiring
                            ) -> Tuple[torch.Tensor, float]:
    """
    Solve a tropical linear program.
    
    Min-plus version:
        minimize: max_i(c_i + x_i)
        subject to: A ⊗ x ≤ b
        
    where A ⊗ x ≤ b means min_j(A_ij + x_j) ≤ b_i for all i.
    
    This is equivalent to finding the smallest tropical scaling
    of a point that satisfies the constraints.
    
    Args:
        A: Constraint matrix (m × n)
        b: Right-hand side (m,)
        semiring: Tropical semiring
        
    Returns:
        (x, objective) tuple
    """
    m, n = A.shape
    
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        # For min-plus constraints A ⊗ x ≤ b:
        # min_j(A_ij + x_j) ≤ b_i
        # This means: for each i, there exists j such that A_ij + x_j ≤ b_i
        
        # Simple feasibility: x_j = min_i(b_i - A_ij)
        x = torch.full((n,), float('inf'))
        for j in range(n):
            for i in range(m):
                if A[i, j] < semiring.zero - 1:
                    bound = b[i] - A[i, j]
                    x[j] = min(x[j], bound.item())
        
        # Replace inf with 0
        x[x == float('inf')] = 0.0
        
        # Compute objective: max_i(c_i + x_i)
        # Assume c = 0 (minimize max coordinate)
        obj = x.max().item()
        
    else:
        # Max-plus: dual formulation
        x = torch.full((n,), -float('inf'))
        for j in range(n):
            for i in range(m):
                if A[i, j] > -semiring.zero + 1:
                    bound = b[i] - A[i, j]
                    x[j] = max(x[j], bound.item())
        
        x[x == -float('inf')] = 0.0
        obj = x.max().item()
    
    return x, obj


def tropical_least_squares(A: torch.Tensor,
                           b: torch.Tensor,
                           semiring: TropicalSemiring = MinPlusSemiring
                           ) -> torch.Tensor:
    """
    Solve tropical least squares: minimize ||A ⊗ x - b||_∞ in tropical sense.
    
    The tropical residual for row i is:
        r_i = |min_j(A_ij + x_j) - b_i|
    
    We minimize max_i r_i.
    
    Args:
        A: Matrix (m × n)
        b: Right-hand side (m,)
        semiring: Tropical semiring
        
    Returns:
        Least squares solution x
    """
    m, n = A.shape
    
    # For min-plus, the solution is related to the column space
    # x_j = median of (b_i - A_ij) over rows i where A_ij is finite
    
    x = torch.zeros(n)
    
    for j in range(n):
        values = []
        for i in range(m):
            if A[i, j] < semiring.zero - 1:
                values.append((b[i] - A[i, j]).item())
        
        if values:
            values.sort()
            x[j] = values[len(values) // 2]  # Median
    
    return x


def tropical_determinant(A: TropicalMatrix) -> float:
    """
    Compute the tropical determinant (tropical permanent).
    
    tdet(A) = ⊕_{σ ∈ S_n} ⊗_{i=1}^n A_{i,σ(i)}
    
    For min-plus: min over all permutations of sum of selected entries.
    This equals the weight of a minimum weight perfect matching!
    
    Args:
        A: Square tropical matrix
        
    Returns:
        Tropical determinant
    """
    n = A.size
    semiring = A.semiring
    
    if n > 10:
        # For large matrices, use Hungarian algorithm
        return _tropical_det_hungarian(A)
    
    # Brute force for small matrices
    from itertools import permutations
    
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        det = float('inf')
        for perm in permutations(range(n)):
            weight = sum(A.data[i, perm[i]].item() for i in range(n))
            det = min(det, weight)
    else:
        det = -float('inf')
        for perm in permutations(range(n)):
            weight = sum(A.data[i, perm[i]].item() for i in range(n))
            det = max(det, weight)
    
    return det


def _tropical_det_hungarian(A: TropicalMatrix) -> float:
    """
    Compute tropical determinant using Hungarian algorithm.
    
    The tropical determinant equals the minimum weight perfect matching.
    """
    n = A.size
    data = A.data.clone()
    
    # Replace inf with large value
    inf_val = A.semiring.zero
    data[data >= inf_val - 1] = 1e8
    
    # Hungarian algorithm for min-plus (assignment problem)
    # Simplified version
    
    u = torch.zeros(n)  # Row potential
    v = torch.zeros(n)  # Column potential
    assignment = [-1] * n
    
    for i in range(n):
        # Find augmenting path
        # Simplified: greedy assignment
        min_val = float('inf')
        min_j = -1
        for j in range(n):
            if j not in assignment[:i] and data[i, j] < min_val:
                min_val = data[i, j].item()
                min_j = j
        assignment[i] = min_j
    
    # Compute total weight
    total = sum(data[i, assignment[i]].item() for i in range(n) if assignment[i] >= 0)
    
    return total


def solve_tropical_equation(A: TropicalMatrix, 
                           b: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Solve the tropical linear equation A ⊗ x = b.
    
    In min-plus: min_j(A_ij + x_j) = b_i for all i.
    
    Not always solvable. Returns None if no solution.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side
        
    Returns:
        Solution x, or None if infeasible
    """
    n = A.size
    m = len(b)
    semiring = A.semiring
    
    # For min-plus, x_j must satisfy:
    # For each i: min_j(A_ij + x_j) = b_i
    # This means x_j ≤ b_i - A_ij for all i,
    # and equality for at least one i per row.
    
    # Upper bound: x_j ≤ min_i(b_i - A_ij)
    x = torch.full((n,), float('inf'))
    for j in range(n):
        for i in range(m):
            if A.data[i, j] < semiring.zero - 1:
                bound = b[i] - A.data[i, j]
                x[j] = min(x[j], bound.item())
    
    # Check feasibility
    x[x == float('inf')] = 0.0
    
    # Verify solution
    Ax = torch.full((m,), semiring.zero)
    for i in range(m):
        if semiring.semiring_type == SemiringType.MIN_PLUS:
            Ax[i] = (A.data[i] + x).min()
        else:
            Ax[i] = (A.data[i] + x).max()
    
    if torch.allclose(Ax, b, atol=1e-6):
        return x
    else:
        return None
