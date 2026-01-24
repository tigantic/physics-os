"""
Persistent Homology Computation

Implements persistence computation via boundary matrix reduction.

The standard algorithm reduces the boundary matrix to identify
persistence pairs (birth, death) for topological features.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import torch

from .simplicial import Simplex, SimplicialComplex
from .boundary import boundary_matrix


@dataclass(frozen=True)
class PersistencePair:
    """
    A persistence pair representing a topological feature.
    
    Attributes:
        birth: Filtration value when feature appears
        death: Filtration value when feature disappears (inf for essential)
        dimension: Homological dimension (0=component, 1=loop, 2=void)
        birth_simplex: Simplex creating the feature
        death_simplex: Simplex destroying the feature (None if essential)
    """
    birth: float
    death: float
    dimension: int
    birth_simplex: Optional[Simplex] = None
    death_simplex: Optional[Simplex] = None
    
    @property
    def persistence(self) -> float:
        """Lifetime of the feature."""
        if math.isinf(self.death):
            return float('inf')
        return self.death - self.birth
    
    @property
    def is_essential(self) -> bool:
        """Whether this is an essential (never-dying) feature."""
        return math.isinf(self.death)
    
    @property
    def midpoint(self) -> float:
        """Midpoint of the feature lifetime."""
        if math.isinf(self.death):
            return float('inf')
        return (self.birth + self.death) / 2
    
    def to_tuple(self) -> Tuple[float, float]:
        """Return (birth, death) tuple."""
        return (self.birth, self.death)


@dataclass
class PersistenceDiagram:
    """
    Persistence diagram: collection of persistence pairs.
    
    Attributes:
        pairs: Dictionary mapping dimension to list of persistence pairs
        complex: The underlying simplicial complex
    """
    pairs: Dict[int, List[PersistencePair]] = field(default_factory=dict)
    complex: Optional[SimplicialComplex] = None
    
    def __getitem__(self, dim: int) -> List[PersistencePair]:
        """Get pairs in dimension dim."""
        return self.pairs.get(dim, [])
    
    def total_persistence(self, dim: Optional[int] = None, p: float = 1.0) -> float:
        """
        Compute total persistence.
        
        Σ persistence^p
        
        Args:
            dim: Specific dimension (None for all)
            p: Power (default 1)
            
        Returns:
            Total persistence
        """
        total = 0.0
        dims = [dim] if dim is not None else list(self.pairs.keys())
        
        for d in dims:
            for pair in self.pairs.get(d, []):
                if not pair.is_essential:
                    total += pair.persistence ** p
        
        return total
    
    def betti_numbers(self) -> List[int]:
        """
        Get Betti numbers (count of essential features).
        """
        if not self.pairs:
            return []
        
        max_dim = max(self.pairs.keys())
        return [
            sum(1 for p in self.pairs.get(d, []) if p.is_essential)
            for d in range(max_dim + 1)
        ]
    
    def filter_by_persistence(self, min_persistence: float) -> 'PersistenceDiagram':
        """
        Filter pairs by minimum persistence.
        
        Args:
            min_persistence: Minimum lifetime threshold
            
        Returns:
            Filtered diagram
        """
        new_pairs = {}
        for dim, pairs in self.pairs.items():
            new_pairs[dim] = [p for p in pairs if p.persistence >= min_persistence]
        return PersistenceDiagram(pairs=new_pairs, complex=self.complex)
    
    def to_array(self, dim: int) -> torch.Tensor:
        """
        Convert to array of (birth, death) points.
        
        Args:
            dim: Dimension
            
        Returns:
            Array shape (n, 2)
        """
        pairs = self[dim]
        if not pairs:
            return torch.zeros(0, 2)
        
        points = []
        for p in pairs:
            if not p.is_essential:
                points.append([p.birth, p.death])
        
        if not points:
            return torch.zeros(0, 2)
        return torch.tensor(points)


def reduce_boundary_matrix(D: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Standard boundary matrix reduction algorithm.
    
    Reduces the boundary matrix to identify pivot columns,
    which correspond to persistence pairs.
    
    Args:
        D: Boundary matrix over Z/2Z (using +/- 1)
        
    Returns:
        (reduced_matrix, pivots) where pivots[j] = i means column j
        has pivot in row i
    """
    R = D.clone()
    m, n = R.shape
    
    pivots = {}  # column -> row of its pivot
    
    for j in range(n):
        # Find pivot (lowest non-zero entry)
        while True:
            # Get lowest non-zero row
            col = R[:, j]
            nonzero = (col.abs() > 1e-10).nonzero(as_tuple=True)[0]
            
            if len(nonzero) == 0:
                # Column is zero
                break
            
            low = nonzero[-1].item()  # Lowest non-zero row
            
            # Check if this row is already a pivot
            found_match = False
            for k in range(j):
                if k in pivots and pivots[k] == low:
                    # Add column k to column j to eliminate this pivot
                    R[:, j] = R[:, j] + R[:, k]
                    # Work over Z/2Z: take mod 2
                    R[:, j] = torch.sign(R[:, j]) * (R[:, j].abs() % 2)
                    found_match = True
                    break
            
            if not found_match:
                # This is a new pivot
                pivots[j] = low
                break
    
    return R, pivots


def persistence_pairs(complex: SimplicialComplex) -> List[PersistencePair]:
    """
    Compute persistence pairs from a filtered simplicial complex.
    
    Uses the standard persistence algorithm.
    
    Args:
        complex: Filtered simplicial complex
        
    Returns:
        List of persistence pairs
    """
    # Sort simplices by filtration value
    sorted_simplices = complex.sorted_simplices()
    n = len(sorted_simplices)
    
    if n == 0:
        return []
    
    # Create index mapping
    simplex_to_idx = {s: i for i, s in enumerate(sorted_simplices)}
    
    # Build full boundary matrix
    D = torch.zeros(n, n)
    
    for j, sigma in enumerate(sorted_simplices):
        for face in sigma.faces():
            if face in simplex_to_idx:
                i = simplex_to_idx[face]
                coeff = sigma.boundary_coefficient(face)
                D[i, j] = coeff
    
    # Reduce the matrix
    R, pivots = reduce_boundary_matrix(D)
    
    # Extract pairs
    pairs = []
    paired = set()
    
    for j, i in pivots.items():
        if j not in paired and i not in paired:
            birth_simplex = sorted_simplices[i]
            death_simplex = sorted_simplices[j]
            
            birth = complex.filtration_values.get(birth_simplex, 0.0)
            death = complex.filtration_values.get(death_simplex, 0.0)
            dim = birth_simplex.dimension
            
            pairs.append(PersistencePair(
                birth=birth,
                death=death,
                dimension=dim,
                birth_simplex=birth_simplex,
                death_simplex=death_simplex
            ))
            
            paired.add(i)
            paired.add(j)
    
    # Essential features (unpaired simplices)
    for i, sigma in enumerate(sorted_simplices):
        if i not in paired:
            birth = complex.filtration_values.get(sigma, 0.0)
            dim = sigma.dimension
            
            # Check if this could be essential
            # (vertices are always essential in H_0 if they don't get paired)
            # Higher dim simplices that are cycles but don't kill anything
            
            pairs.append(PersistencePair(
                birth=birth,
                death=float('inf'),
                dimension=dim,
                birth_simplex=sigma,
                death_simplex=None
            ))
    
    return pairs


def compute_persistence(complex: SimplicialComplex) -> PersistenceDiagram:
    """
    Compute persistence diagram from a filtered simplicial complex.
    
    Args:
        complex: Filtered simplicial complex
        
    Returns:
        Persistence diagram
    """
    pairs = persistence_pairs(complex)
    
    # Organize by dimension
    pairs_by_dim: Dict[int, List[PersistencePair]] = {}
    
    for pair in pairs:
        dim = pair.dimension
        if dim not in pairs_by_dim:
            pairs_by_dim[dim] = []
        pairs_by_dim[dim].append(pair)
    
    return PersistenceDiagram(pairs=pairs_by_dim, complex=complex)


def compute_betti_curve(complex: SimplicialComplex,
                        filtration_values: List[float]) -> Dict[int, List[int]]:
    """
    Compute Betti numbers at each filtration value.
    
    Args:
        complex: Filtered simplicial complex
        filtration_values: Values at which to compute Betti numbers
        
    Returns:
        {dim: [betti_0, betti_1, ...]} for each filtration value
    """
    diagram = compute_persistence(complex)
    
    betti_curves: Dict[int, List[int]] = {}
    
    for dim in diagram.pairs.keys():
        curve = []
        for t in filtration_values:
            # Count features alive at time t
            count = sum(
                1 for p in diagram[dim]
                if p.birth <= t < p.death
            )
            curve.append(count)
        betti_curves[dim] = curve
    
    return betti_curves


def euler_curve(complex: SimplicialComplex,
                filtration_values: List[float]) -> List[int]:
    """
    Compute Euler characteristic at each filtration value.
    
    χ(t) = Σ_k (-1)^k β_k(t)
    
    Args:
        complex: Filtered simplicial complex
        filtration_values: Values at which to compute
        
    Returns:
        Euler characteristic curve
    """
    betti_curves = compute_betti_curve(complex, filtration_values)
    
    euler = []
    for i, t in enumerate(filtration_values):
        chi = 0
        for dim, curve in betti_curves.items():
            chi += ((-1) ** dim) * curve[i]
        euler.append(chi)
    
    return euler


class VineyardAlgorithm:
    """
    Vineyard algorithm for tracking persistence across a family of filtrations.
    
    This is useful when the filtration changes continuously.
    """
    
    def __init__(self, initial_complex: SimplicialComplex):
        self.complex = initial_complex
        self.diagram = compute_persistence(initial_complex)
    
    def update_filtration(self, new_values: Dict[Simplex, float]) -> PersistenceDiagram:
        """
        Update filtration values and recompute persistence.
        
        For now, this does full recomputation. A true vineyard
        implementation would track swaps incrementally.
        
        Args:
            new_values: New filtration values
            
        Returns:
            Updated persistence diagram
        """
        self.complex.filtration_values.update(new_values)
        self.diagram = compute_persistence(self.complex)
        return self.diagram
