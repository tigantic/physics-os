"""
Distance Metrics on Persistence Diagrams

Implements bottleneck distance, Wasserstein distance, and
persistence landscapes for comparing persistence diagrams.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import torch

from .persistence import PersistencePair, PersistenceDiagram


def _l_infinity_cost(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """L∞ distance between two points."""
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def _l_p_cost(p1: Tuple[float, float], p2: Tuple[float, float], p: float = 2.0) -> float:
    """Lp distance between two points."""
    return (abs(p1[0] - p2[0]) ** p + abs(p1[1] - p2[1]) ** p) ** (1/p)


def _diagonal_projection(point: Tuple[float, float]) -> Tuple[float, float]:
    """Project point to diagonal."""
    mid = (point[0] + point[1]) / 2
    return (mid, mid)


def _diagonal_cost(point: Tuple[float, float], metric: str = "inf", p: float = 2.0) -> float:
    """Cost of matching a point to its diagonal projection."""
    proj = _diagonal_projection(point)
    if metric == "inf":
        return _l_infinity_cost(point, proj)
    return _l_p_cost(point, proj, p)


def bottleneck_distance(diagram1: PersistenceDiagram,
                        diagram2: PersistenceDiagram,
                        dim: int = 0) -> float:
    """
    Compute bottleneck distance between persistence diagrams.
    
    The bottleneck distance is the infimum over all matchings of
    the maximum cost among matched pairs.
    
    d_B(D1, D2) = inf_γ sup_{p ∈ D1} ||p - γ(p)||_∞
    
    where γ can match points to points or to the diagonal.
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        dim: Homological dimension to compare
        
    Returns:
        Bottleneck distance
    """
    # Get points (excluding essential features for simplicity)
    pts1 = [(p.birth, p.death) for p in diagram1[dim] if not p.is_essential]
    pts2 = [(p.birth, p.death) for p in diagram2[dim] if not p.is_essential]
    
    if not pts1 and not pts2:
        return 0.0
    
    # Add diagonal points for unmatched points
    # Simple greedy approximation (exact requires Hungarian algorithm)
    
    # Build cost matrix: points vs points + diagonal options
    n1, n2 = len(pts1), len(pts2)
    
    if n1 == 0:
        # All pts2 go to diagonal
        return max(_diagonal_cost(p, "inf") for p in pts2)
    if n2 == 0:
        # All pts1 go to diagonal
        return max(_diagonal_cost(p, "inf") for p in pts1)
    
    # Create augmented cost matrix
    # Size: (n1 + n2) x (n1 + n2)
    # Top-left: cost between pts1 and pts2
    # Top-right: cost of pts1 to diagonal (only diagonal entries)
    # Bottom-left: cost of pts2 to diagonal (only diagonal entries)
    # Bottom-right: zeros (diagonal to diagonal)
    
    size = n1 + n2
    cost = torch.full((size, size), float('inf'))
    
    # Point-to-point costs
    for i, p1 in enumerate(pts1):
        for j, p2 in enumerate(pts2):
            cost[i, j] = _l_infinity_cost(p1, p2)
    
    # Point-to-diagonal costs
    for i, p1 in enumerate(pts1):
        cost[i, n2 + i] = _diagonal_cost(p1, "inf")
    
    for j, p2 in enumerate(pts2):
        cost[n1 + j, j] = _diagonal_cost(p2, "inf")
    
    # Diagonal-to-diagonal
    for i in range(n1):
        for j in range(n2):
            cost[n1 + j, n2 + i] = 0.0
    
    # Greedy matching (approximation)
    matched_rows = set()
    matched_cols = set()
    max_cost = 0.0
    
    while len(matched_rows) < size:
        # Find minimum unmatched entry
        min_val = float('inf')
        min_i, min_j = -1, -1
        
        for i in range(size):
            if i in matched_rows:
                continue
            for j in range(size):
                if j in matched_cols:
                    continue
                if cost[i, j] < min_val:
                    min_val = cost[i, j]
                    min_i, min_j = i, j
        
        if min_i == -1:
            break
        
        matched_rows.add(min_i)
        matched_cols.add(min_j)
        max_cost = max(max_cost, min_val)
    
    return max_cost


def wasserstein_distance_diagram(diagram1: PersistenceDiagram,
                                 diagram2: PersistenceDiagram,
                                 dim: int = 0,
                                 p: float = 2.0,
                                 q: float = 2.0) -> float:
    """
    Compute p-Wasserstein distance between persistence diagrams.
    
    W_p(D1, D2) = (inf_γ Σ_{x ∈ D1} ||x - γ(x)||_q^p)^{1/p}
    
    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        dim: Homological dimension
        p: Wasserstein exponent
        q: Ground metric exponent
        
    Returns:
        Wasserstein distance
    """
    pts1 = [(pair.birth, pair.death) for pair in diagram1[dim] if not pair.is_essential]
    pts2 = [(pair.birth, pair.death) for pair in diagram2[dim] if not pair.is_essential]
    
    if not pts1 and not pts2:
        return 0.0
    
    n1, n2 = len(pts1), len(pts2)
    
    if n1 == 0:
        return sum(_diagonal_cost(pt, "p", q) ** p for pt in pts2) ** (1/p)
    if n2 == 0:
        return sum(_diagonal_cost(pt, "p", q) ** p for pt in pts1) ** (1/p)
    
    # Build cost matrix with diagonal augmentation
    size = n1 + n2
    cost = torch.full((size, size), float('inf'))
    
    for i, p1 in enumerate(pts1):
        for j, p2 in enumerate(pts2):
            cost[i, j] = _l_p_cost(p1, p2, q) ** p
    
    for i, pt in enumerate(pts1):
        cost[i, n2 + i] = _diagonal_cost(pt, "p", q) ** p
    
    for j, pt in enumerate(pts2):
        cost[n1 + j, j] = _diagonal_cost(pt, "p", q) ** p
    
    for i in range(n1):
        for j in range(n2):
            cost[n1 + j, n2 + i] = 0.0
    
    # Greedy matching
    matched_rows = set()
    matched_cols = set()
    total_cost = 0.0
    
    # Sort by cost
    edges = []
    for i in range(size):
        for j in range(size):
            if cost[i, j] < float('inf'):
                edges.append((cost[i, j].item(), i, j))
    edges.sort()
    
    for c, i, j in edges:
        if i not in matched_rows and j not in matched_cols:
            matched_rows.add(i)
            matched_cols.add(j)
            total_cost += c
    
    return total_cost ** (1/p)


@dataclass
class PersistenceLandscape:
    """
    Persistence landscape representation.
    
    The k-th landscape function λ_k(t) is the k-th largest value
    of the "tent functions" centered at persistence points.
    
    Attributes:
        landscapes: List of landscape functions at sample points
        sample_points: Points where landscapes are evaluated
        dimension: Homological dimension
    """
    landscapes: List[torch.Tensor]  # List of λ_k evaluations
    sample_points: torch.Tensor
    dimension: int
    
    def __getitem__(self, k: int) -> torch.Tensor:
        """Get k-th landscape function values."""
        if k < len(self.landscapes):
            return self.landscapes[k]
        return torch.zeros_like(self.sample_points)
    
    @property
    def num_landscapes(self) -> int:
        """Number of non-zero landscape functions."""
        return len(self.landscapes)


def persistence_landscape(diagram: PersistenceDiagram,
                          dim: int = 0,
                          num_samples: int = 100,
                          num_landscapes: int = 10) -> PersistenceLandscape:
    """
    Compute persistence landscape from diagram.
    
    For each persistence point (b, d), define the tent function:
    Λ_{b,d}(t) = max(0, min(t-b, d-t))
    
    The k-th landscape is the k-th largest tent value at each t.
    
    Args:
        diagram: Persistence diagram
        dim: Homological dimension
        num_samples: Number of sample points
        num_landscapes: Number of landscapes to compute
        
    Returns:
        Persistence landscape
    """
    pairs = [p for p in diagram[dim] if not p.is_essential]
    
    if not pairs:
        samples = torch.linspace(0, 1, num_samples)
        return PersistenceLandscape(
            landscapes=[],
            sample_points=samples,
            dimension=dim
        )
    
    # Determine range
    min_birth = min(p.birth for p in pairs)
    max_death = max(p.death for p in pairs)
    
    samples = torch.linspace(min_birth, max_death, num_samples)
    
    # Compute tent functions
    tent_values = torch.zeros(len(pairs), num_samples)
    
    for i, pair in enumerate(pairs):
        b, d = pair.birth, pair.death
        for j, t in enumerate(samples):
            t_val = t.item()
            if b <= t_val <= d:
                tent_values[i, j] = min(t_val - b, d - t_val)
    
    # Sort at each sample point to get landscapes
    sorted_values, _ = torch.sort(tent_values, dim=0, descending=True)
    
    landscapes = []
    for k in range(min(num_landscapes, len(pairs))):
        landscapes.append(sorted_values[k])
    
    return PersistenceLandscape(
        landscapes=landscapes,
        sample_points=samples,
        dimension=dim
    )


def landscape_distance(landscape1: PersistenceLandscape,
                       landscape2: PersistenceLandscape,
                       p: float = 2.0) -> float:
    """
    Compute Lp distance between persistence landscapes.
    
    Args:
        landscape1: First landscape
        landscape2: Second landscape
        p: Exponent (1, 2, or inf)
        
    Returns:
        Distance
    """
    # Assume same sample points
    max_k = max(landscape1.num_landscapes, landscape2.num_landscapes)
    
    total = 0.0
    for k in range(max_k):
        diff = landscape1[k] - landscape2[k]
        
        if p == float('inf'):
            total = max(total, diff.abs().max().item())
        else:
            total += (diff.abs() ** p).sum().item()
    
    if p == float('inf'):
        return total
    return total ** (1/p)


def persistence_image(diagram: PersistenceDiagram,
                      dim: int = 0,
                      resolution: int = 20,
                      sigma: float = 0.1) -> torch.Tensor:
    """
    Compute persistence image from diagram.
    
    Converts persistence diagram to a fixed-size vector by:
    1. Transforming to birth-persistence coordinates
    2. Placing Gaussian at each point
    3. Discretizing on a grid
    4. Weighting by persistence
    
    Args:
        diagram: Persistence diagram
        dim: Homological dimension
        resolution: Grid resolution
        sigma: Gaussian bandwidth
        
    Returns:
        Persistence image, shape (resolution, resolution)
    """
    pairs = [p for p in diagram[dim] if not p.is_essential]
    
    if not pairs:
        return torch.zeros(resolution, resolution)
    
    # Transform to (birth, persistence) coordinates
    points = [(p.birth, p.persistence) for p in pairs]
    
    # Determine range
    births = [p[0] for p in points]
    perss = [p[1] for p in points]
    
    b_min, b_max = min(births), max(births)
    p_min, p_max = 0, max(perss)  # Persistence starts at 0
    
    # Add padding
    b_range = b_max - b_min + 1e-6
    p_range = p_max - p_min + 1e-6
    
    # Create grid
    b_grid = torch.linspace(b_min - 0.1 * b_range, b_max + 0.1 * b_range, resolution)
    p_grid = torch.linspace(p_min, p_max + 0.1 * p_range, resolution)
    
    image = torch.zeros(resolution, resolution)
    
    for birth, pers in points:
        # Weight by persistence
        weight = pers
        
        # Add Gaussian
        for i, bi in enumerate(b_grid):
            for j, pj in enumerate(p_grid):
                dist_sq = (bi - birth) ** 2 + (pj - pers) ** 2
                image[i, j] += weight * torch.exp(-dist_sq / (2 * sigma ** 2))
    
    return image


def diagram_entropy(diagram: PersistenceDiagram, dim: int = 0) -> float:
    """
    Compute persistent entropy of a diagram.
    
    H = -Σ (p_i / L) log(p_i / L)
    
    where p_i is persistence and L is total persistence.
    
    Args:
        diagram: Persistence diagram
        dim: Homological dimension
        
    Returns:
        Persistent entropy
    """
    pairs = [p for p in diagram[dim] if not p.is_essential and p.persistence > 0]
    
    if not pairs:
        return 0.0
    
    persistences = [p.persistence for p in pairs]
    total = sum(persistences)
    
    if total == 0:
        return 0.0
    
    probs = [p / total for p in persistences]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    
    return entropy


def silhouette(diagram: PersistenceDiagram,
               dim: int = 0,
               power: float = 1.0,
               num_samples: int = 100) -> torch.Tensor:
    """
    Compute persistence silhouette.
    
    Weighted average of landscape functions.
    
    Args:
        diagram: Persistence diagram
        dim: Homological dimension
        power: Weighting power
        num_samples: Number of sample points
        
    Returns:
        Silhouette function values
    """
    pairs = [p for p in diagram[dim] if not p.is_essential]
    
    if not pairs:
        return torch.zeros(num_samples)
    
    min_birth = min(p.birth for p in pairs)
    max_death = max(p.death for p in pairs)
    
    samples = torch.linspace(min_birth, max_death, num_samples)
    silhouette_vals = torch.zeros(num_samples)
    
    total_weight = 0.0
    
    for pair in pairs:
        b, d = pair.birth, pair.death
        weight = (d - b) ** power
        total_weight += weight
        
        for j, t in enumerate(samples):
            t_val = t.item()
            if b <= t_val <= d:
                tent = min(t_val - b, d - t_val)
                silhouette_vals[j] += weight * tent
    
    if total_weight > 0:
        silhouette_vals /= total_weight
    
    return silhouette_vals
