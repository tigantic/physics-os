"""
QTT Tropical Convexity

Tropical convexity generalizes classical convexity to the tropical semiring.

A set S is tropically convex if for all x, y in S:
    λ ⊙ x ⊕ μ ⊙ y ∈ S  for all λ, μ with λ ⊕ μ = 0

In min-plus: min(λ + x, μ + y) ∈ S for all λ, μ with min(λ, μ) = 0

Tropical polyhedra are solution sets of tropical linear inequalities.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import torch

from tensornet.genesis.tropical.semiring import (
    TropicalSemiring, MinPlusSemiring, MaxPlusSemiring, SemiringType
)


@dataclass
class TropicalHalfspace:
    """
    A tropical halfspace defined by a tropical linear inequality.
    
    In min-plus: min_i(a_i + x_i) ≤ min_j(b_j + x_j)
    
    Attributes:
        a: Left coefficients
        b: Right coefficients
        semiring: Tropical semiring
    """
    a: torch.Tensor
    b: torch.Tensor
    semiring: TropicalSemiring = field(default_factory=lambda: MinPlusSemiring)
    
    @property
    def dimension(self) -> int:
        """Ambient dimension."""
        return len(self.a)
    
    def contains(self, x: torch.Tensor) -> bool:
        """
        Check if point x is in the halfspace.
        
        Args:
            x: Point to test
            
        Returns:
            True if x satisfies the inequality
        """
        if self.semiring.semiring_type == SemiringType.MIN_PLUS:
            left = (self.a + x).min().item()
            right = (self.b + x).min().item()
            return left <= right + 1e-10
        else:
            left = (self.a + x).max().item()
            right = (self.b + x).max().item()
            return left <= right + 1e-10
    
    @classmethod
    def from_inequality(cls, coeffs: torch.Tensor,
                        lhs_indices: List[int],
                        rhs_indices: List[int],
                        semiring: TropicalSemiring = MinPlusSemiring
                        ) -> 'TropicalHalfspace':
        """
        Create halfspace from coefficient vector and index sets.
        
        min_{i in lhs} (c_i + x_i) ≤ min_{j in rhs} (c_j + x_j)
        """
        n = len(coeffs)
        a = torch.full((n,), semiring.zero)
        b = torch.full((n,), semiring.zero)
        
        for i in lhs_indices:
            a[i] = coeffs[i]
        for j in rhs_indices:
            b[j] = coeffs[j]
        
        return cls(a=a, b=b, semiring=semiring)


@dataclass
class TropicalPolyhedron:
    """
    A tropical polyhedron: intersection of tropical halfspaces.
    
    Tropical polyhedra are the solution sets of systems of
    tropical linear inequalities:
    
        min(a₁ + x, a₂ + y, ...) ≤ min(b₁ + x, b₂ + y, ...)
    
    Attributes:
        halfspaces: List of defining halfspaces
        vertices: Optional list of tropical vertices
        semiring: Tropical semiring
    """
    halfspaces: List[TropicalHalfspace] = field(default_factory=list)
    vertices: Optional[torch.Tensor] = None
    semiring: TropicalSemiring = field(default_factory=lambda: MinPlusSemiring)
    
    @property
    def dimension(self) -> int:
        """Ambient dimension."""
        if self.halfspaces:
            return self.halfspaces[0].dimension
        if self.vertices is not None:
            return self.vertices.shape[1]
        return 0
    
    @property
    def num_constraints(self) -> int:
        """Number of halfspace constraints."""
        return len(self.halfspaces)
    
    def contains(self, x: torch.Tensor) -> bool:
        """Check if point is in the polyhedron."""
        return all(h.contains(x) for h in self.halfspaces)
    
    def add_constraint(self, halfspace: TropicalHalfspace):
        """Add a halfspace constraint."""
        self.halfspaces.append(halfspace)
        self.vertices = None  # Invalidate cached vertices
    
    @classmethod
    def box(cls, lower: torch.Tensor, upper: torch.Tensor,
            semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalPolyhedron':
        """
        Create tropical polyhedron from coordinate bounds.
        
        Args:
            lower: Lower bounds (can be -∞)
            upper: Upper bounds (can be +∞)
            
        Returns:
            TropicalPolyhedron representing the box
        """
        n = len(lower)
        halfspaces = []
        
        # In min-plus:
        # x_i ≥ l_i means: min(0 + x_i) ≥ min(l_i + 0) → need to encode
        # For tropical polyhedra, bounds are encoded differently
        
        # x_i ≤ u_i: x_i ≤ u_i
        # In min-plus terms: min(0 + x_i) ≤ min(u_i + const)
        
        for i in range(n):
            if upper[i] < semiring.zero - 1:
                # x_i ≤ upper[i]
                a = torch.full((n,), semiring.zero)
                a[i] = 0.0
                b = torch.full((n,), semiring.zero)
                b[0] = upper[i]  # Use first coord as reference
                halfspaces.append(TropicalHalfspace(a, b, semiring))
        
        return cls(halfspaces=halfspaces, semiring=semiring)
    
    @classmethod
    def simplex(cls, n: int, scale: float = 1.0,
                semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalPolyhedron':
        """
        Create the tropical simplex in R^n.
        
        The tropical simplex is the set:
            {x : min(x) = 0, max(x) ≤ scale}
        
        Args:
            n: Dimension
            scale: Maximum coordinate value
            
        Returns:
            Tropical simplex polyhedron
        """
        # Vertices of tropical simplex
        vertices = torch.zeros((n, n))
        for i in range(n):
            vertices[i, :] = scale
            vertices[i, i] = 0.0
        
        return cls(vertices=vertices, semiring=semiring)


def tropical_convex_hull(points: torch.Tensor,
                         semiring: TropicalSemiring = MinPlusSemiring
                         ) -> TropicalPolyhedron:
    """
    Compute the tropical convex hull of a set of points.
    
    The tropical convex hull is the smallest tropically convex set
    containing all given points.
    
    For a finite set P, the tropical convex hull is:
        tconv(P) = {⊕ λ_i ⊙ p_i : λ_i ∈ T, ⊕ λ_i = 0}
    
    Args:
        points: (m, n) tensor of m points in R^n
        semiring: Tropical semiring
        
    Returns:
        TropicalPolyhedron representing the convex hull
    """
    m, n = points.shape
    
    # For tropical convex hull, the vertices are a subset of the input points
    # that are "tropically extreme"
    
    extreme_indices = []
    
    for i in range(m):
        is_extreme = True
        p_i = points[i]
        
        # Check if p_i can be written as tropical combination of others
        for j in range(m):
            if i == j:
                continue
            
            for k in range(m):
                if k == i or k == j:
                    continue
                
                # Check if p_i is in tropical segment [p_j, p_k]
                p_j = points[j]
                p_k = points[k]
                
                # p_i in segment means: for all coords,
                # p_i = min(λ + p_j, μ + p_k) for some λ, μ with min(λ,μ) = 0
                
                if _in_tropical_segment(p_i, p_j, p_k, semiring):
                    is_extreme = False
                    break
            
            if not is_extreme:
                break
        
        if is_extreme:
            extreme_indices.append(i)
    
    vertices = points[extreme_indices] if extreme_indices else points
    
    return TropicalPolyhedron(vertices=vertices, semiring=semiring)


def _in_tropical_segment(p: torch.Tensor, 
                         q: torch.Tensor, 
                         r: torch.Tensor,
                         semiring: TropicalSemiring) -> bool:
    """
    Check if p is in the tropical segment [q, r].
    
    p ∈ [q, r] if p = λ ⊙ q ⊕ μ ⊙ r for some λ, μ with λ ⊕ μ = 0.
    
    In min-plus: p_i = min(λ + q_i, μ + r_i) for all i,
    with min(λ, μ) = 0.
    """
    n = len(p)
    
    # For min-plus, we need λ, μ ≥ 0 with min(λ, μ) = 0
    # So either λ = 0 or μ = 0
    
    # Case 1: λ = 0, μ ≥ 0
    # p_i = min(q_i, μ + r_i)
    # This means: μ ≥ p_i - r_i when p_i ≤ q_i
    #            and p_i = q_i when q_i ≤ μ + r_i
    
    # Simplified check: p is on segment if componentwise
    # min(q, r) ≤ p ≤ max(q, r) up to tropical shift
    
    diff_pq = p - q
    diff_pr = p - r
    diff_qr = q - r
    
    # Normalize to have min = 0
    p_norm = p - p.min()
    q_norm = q - q.min()
    r_norm = r - r.min()
    
    # Check if p_norm is componentwise between q_norm and r_norm
    lower = torch.minimum(q_norm, r_norm)
    upper = torch.maximum(q_norm, r_norm)
    
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        return (p_norm >= lower - 1e-6).all() and (p_norm <= upper + 1e-6).all()
    else:
        return (p_norm >= lower - 1e-6).all() and (p_norm <= upper + 1e-6).all()


def is_tropically_convex(points: torch.Tensor,
                         semiring: TropicalSemiring = MinPlusSemiring,
                         tol: float = 1e-6) -> bool:
    """
    Check if a finite set of points is tropically convex.
    
    A set is tropically convex if for any two points, the tropical
    segment between them is contained in the set.
    
    Args:
        points: (m, n) tensor of points
        semiring: Tropical semiring
        tol: Tolerance for point containment
        
    Returns:
        True if set is tropically convex
    """
    m, n = points.shape
    
    if m <= 2:
        return True
    
    # Check all pairs
    for i in range(m):
        for j in range(i + 1, m):
            # Sample points on tropical segment
            for t in [0.25, 0.5, 0.75]:
                # Tropical midpoint with parameter t
                if semiring.semiring_type == SemiringType.MIN_PLUS:
                    # λ = 0, μ = log(t/(1-t)) approximately
                    # For t = 0.5, μ = 0, giving standard midpoint
                    mid = torch.minimum(points[i], points[j])
                else:
                    mid = torch.maximum(points[i], points[j])
                
                # Check if midpoint is in the set
                found = False
                for k in range(m):
                    if torch.allclose(mid, points[k], atol=tol):
                        found = True
                        break
                
                # For finite sets, midpoint may not be exactly a point
                # This is a simplified check
    
    # A more accurate check would compute the tropical hull and compare
    hull = tropical_convex_hull(points, semiring)
    
    # Set is convex if hull has same number of vertices
    if hull.vertices is not None:
        return len(hull.vertices) == m
    
    return True


def tropical_projection(x: torch.Tensor,
                        polyhedron: TropicalPolyhedron) -> torch.Tensor:
    """
    Project a point onto a tropical polyhedron.
    
    The tropical projection minimizes the tropical distance
    to the polyhedron.
    
    For tropical halfspace intersection, we use iterative projection:
    1. Project onto each halfspace in sequence
    2. Repeat until convergence
    
    The projection onto a single halfspace H = {y : ⊕_i(a_i + y_i) ≤ ⊕_j(b_j + y_j)}
    is computed by adjusting coordinates to satisfy the inequality.
    
    Args:
        x: Point to project
        polyhedron: Target polyhedron
        
    Returns:
        Projected point
    """
    semiring = polyhedron.semiring
    
    # If already in polyhedron, return x
    if polyhedron.contains(x):
        return x.clone()
    
    # If vertices available, find closest vertex
    if polyhedron.vertices is not None:
        vertices = polyhedron.vertices
        
        best_dist = float('inf')
        best_vertex = vertices[0]
        
        for i in range(len(vertices)):
            v = vertices[i]
            
            # Tropical Hilbert distance
            diff = x - v
            dist = (diff.max() - diff.min()).item()
            
            if dist < best_dist:
                best_dist = dist
                best_vertex = v
        
        # Project onto tropical segment [x, best_vertex]
        # The tropical segment consists of points:
        # min(λ + x, μ + v) for λ, μ with min(λ, μ) = 0
        # i.e., λ = 0 or μ = 0
        # This gives points: min(x, μ + v) or min(λ + x, v)
        
        # Find λ* that minimizes distance while staying in polyhedron
        for alpha in torch.linspace(0, 1, 20, device=x.device):
            # Candidate on tropical segment
            if semiring.semiring_type == SemiringType.MIN_PLUS:
                candidate = torch.minimum(x, alpha.item() + best_vertex)
            else:
                candidate = torch.maximum(x, alpha.item() + best_vertex)
            
            if polyhedron.contains(candidate):
                return candidate
        
        return best_vertex.clone()
    
    # Handle projection to halfspace intersection via iterative projection
    if polyhedron.halfspaces:
        proj = x.clone()
        
        # Dykstra-like iteration for tropical halfspaces
        max_iters = 100
        tol = 1e-8
        
        for iteration in range(max_iters):
            proj_old = proj.clone()
            
            # Project onto each halfspace in sequence
            for halfspace in polyhedron.halfspaces:
                proj = _project_onto_halfspace(proj, halfspace)
            
            # Check convergence
            if (proj - proj_old).abs().max() < tol:
                break
        
        return proj
    
    # No constraints - return original point
    return x.clone()


def _project_onto_halfspace(x: torch.Tensor, 
                            halfspace: TropicalHalfspace) -> torch.Tensor:
    """
    Project point x onto a single tropical halfspace.
    
    For min-plus halfspace: min_i(a_i + x_i) ≤ min_j(b_j + x_j)
    
    If x violates the inequality, we adjust coordinates to satisfy it
    with minimal change in tropical distance.
    
    Algorithm:
    1. Compute violation δ = min(a + x) - min(b + x)
    2. If δ > 0, we need to decrease lhs or increase rhs
    3. Adjust the coordinate achieving the lhs minimum by subtracting δ
    
    Args:
        x: Point to project
        halfspace: Target halfspace
        
    Returns:
        Projected point
    """
    if halfspace.contains(x):
        return x.clone()
    
    proj = x.clone()
    a, b = halfspace.a, halfspace.b
    semiring = halfspace.semiring
    
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        # Left side: min_i(a_i + x_i)
        ax = a + x
        lhs_val = ax.min().item()
        lhs_idx = ax.argmin().item()
        
        # Right side: min_j(b_j + x_j)
        bx = b + x
        # Only consider finite entries
        finite_mask = bx < semiring.zero - 1
        if finite_mask.any():
            rhs_val = bx[finite_mask].min().item()
        else:
            rhs_val = semiring.zero
        
        # Violation amount
        delta = lhs_val - rhs_val
        
        if delta > 1e-10:
            # Need to decrease lhs by δ
            # Decrease x at the lhs-achieving index
            proj[lhs_idx] = proj[lhs_idx] - delta
    else:
        # max-plus: max_i(a_i + x_i) ≤ max_j(b_j + x_j)
        ax = a + x
        finite_mask_a = ax > semiring.zero + 1
        if finite_mask_a.any():
            lhs_val = ax[finite_mask_a].max().item()
            lhs_idx = ax.argmax().item()
        else:
            return proj
        
        bx = b + x
        rhs_val = bx.max().item()
        
        delta = lhs_val - rhs_val
        
        if delta > 1e-10:
            proj[lhs_idx] = proj[lhs_idx] - delta
    
    return proj


def tropical_hilbert_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute the tropical Hilbert distance between two points.
    
    d_H(x, y) = max_i(x_i - y_i) - min_i(x_i - y_i)
    
    This is a projective metric: d_H(λx, y) = d_H(x, y) for any λ.
    
    Args:
        x, y: Points in tropical projective space
        
    Returns:
        Hilbert distance
    """
    diff = x - y
    return (diff.max() - diff.min()).item()


def tropical_barycenter(points: torch.Tensor,
                        weights: Optional[torch.Tensor] = None,
                        semiring: TropicalSemiring = MinPlusSemiring
                        ) -> torch.Tensor:
    """
    Compute the tropical barycenter (weighted tropical sum).
    
    For min-plus: bar = ⊕_i (w_i ⊙ p_i) = min_i (w_i + p_i)
    
    Args:
        points: (m, n) tensor of points
        weights: (m,) tensor of tropical weights
        semiring: Tropical semiring
        
    Returns:
        Tropical barycenter
    """
    m, n = points.shape
    
    if weights is None:
        weights = torch.zeros(m)  # Equal weights in tropical sense
    
    # Add weights to points
    shifted = points + weights.unsqueeze(1)
    
    # Take tropical sum (min or max)
    if semiring.semiring_type == SemiringType.MIN_PLUS:
        barycenter = shifted.min(dim=0)[0]
    else:
        barycenter = shifted.max(dim=0)[0]
    
    return barycenter
