"""
Simplicial Complex Construction

Implements simplicial complexes and standard constructions
(Rips complex, Vietoris-Rips, Čech complex).

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, FrozenSet, Iterator
import torch
from itertools import combinations


@dataclass(frozen=True)
class Simplex:
    """
    A k-simplex represented by a sorted tuple of vertices.
    
    A k-simplex is the convex hull of k+1 affinely independent points.
    - 0-simplex: vertex
    - 1-simplex: edge
    - 2-simplex: triangle
    - 3-simplex: tetrahedron
    
    Attributes:
        vertices: Sorted tuple of vertex indices
    """
    vertices: Tuple[int, ...]
    
    def __post_init__(self):
        # Ensure vertices are sorted (for canonical form)
        object.__setattr__(self, 'vertices', tuple(sorted(self.vertices)))
    
    @property
    def dimension(self) -> int:
        """Dimension of the simplex (k for k-simplex)."""
        return len(self.vertices) - 1
    
    def faces(self) -> Iterator['Simplex']:
        """
        Generate all (k-1)-dimensional faces.
        
        Each face is obtained by removing one vertex.
        
        Yields:
            Face simplices
        """
        for i in range(len(self.vertices)):
            face_vertices = self.vertices[:i] + self.vertices[i+1:]
            if face_vertices:
                yield Simplex(face_vertices)
    
    def boundary_coefficient(self, face: 'Simplex') -> int:
        """
        Compute the boundary coefficient for a face.
        
        The boundary of a k-simplex [v_0, ..., v_k] is:
        ∂[v_0, ..., v_k] = Σ (-1)^i [v_0, ..., v̂_i, ..., v_k]
        
        Args:
            face: A (k-1)-face of this simplex
            
        Returns:
            +1 or -1 coefficient
        """
        # Find which vertex was removed
        for i, v in enumerate(self.vertices):
            if v not in face.vertices:
                return (-1) ** i
        return 0
    
    def __hash__(self) -> int:
        return hash(self.vertices)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Simplex):
            return False
        return self.vertices == other.vertices
    
    def __lt__(self, other: 'Simplex') -> bool:
        """Order by dimension, then lexicographically."""
        if self.dimension != other.dimension:
            return self.dimension < other.dimension
        return self.vertices < other.vertices


@dataclass
class SimplicialComplex:
    """
    A simplicial complex is a collection of simplices closed under
    taking faces.
    
    Attributes:
        simplices: Set of all simplices
        max_dim: Maximum dimension
        filtration_values: Optional mapping from simplex to filtration value
    """
    simplices: Set[Simplex] = field(default_factory=set)
    max_dim: int = 0
    filtration_values: Dict[Simplex, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.simplices:
            self.max_dim = max(s.dimension for s in self.simplices)
    
    @classmethod
    def from_edges(cls, vertices: int, edges: List[Tuple[int, int]]) -> 'SimplicialComplex':
        """
        Construct 1-dimensional complex from vertices and edges.
        
        Args:
            vertices: Number of vertices
            edges: List of (i, j) edges
            
        Returns:
            Simplicial complex with 0-simplices and 1-simplices
        """
        simplices = set()
        
        # Add vertices
        for v in range(vertices):
            simplices.add(Simplex((v,)))
        
        # Add edges
        for i, j in edges:
            simplices.add(Simplex((i, j)))
        
        return cls(simplices=simplices, max_dim=1)
    
    def add_simplex(self, simplex: Simplex, 
                    filtration_value: Optional[float] = None,
                    add_faces: bool = True):
        """
        Add a simplex and optionally all its faces.
        
        Args:
            simplex: Simplex to add
            filtration_value: Optional filtration value
            add_faces: Whether to add all faces
        """
        self.simplices.add(simplex)
        if filtration_value is not None:
            self.filtration_values[simplex] = filtration_value
        
        self.max_dim = max(self.max_dim, simplex.dimension)
        
        if add_faces:
            for face in simplex.faces():
                self.add_simplex(face, filtration_value, add_faces=True)
    
    def simplices_of_dim(self, k: int) -> List[Simplex]:
        """Get all k-dimensional simplices."""
        return sorted([s for s in self.simplices if s.dimension == k])
    
    def num_simplices(self, k: Optional[int] = None) -> int:
        """Count simplices (of dimension k if specified)."""
        if k is None:
            return len(self.simplices)
        return len(self.simplices_of_dim(k))
    
    def euler_characteristic(self) -> int:
        """
        Compute Euler characteristic χ = Σ (-1)^k |S_k|.
        """
        chi = 0
        for k in range(self.max_dim + 1):
            chi += ((-1) ** k) * self.num_simplices(k)
        return chi
    
    def skeleton(self, k: int) -> 'SimplicialComplex':
        """Get k-skeleton (simplices of dimension ≤ k)."""
        skel_simplices = {s for s in self.simplices if s.dimension <= k}
        filt = {s: v for s, v in self.filtration_values.items() if s.dimension <= k}
        return SimplicialComplex(
            simplices=skel_simplices,
            max_dim=min(k, self.max_dim),
            filtration_values=filt
        )
    
    def is_valid(self) -> bool:
        """Check that complex is closed under taking faces."""
        for simplex in self.simplices:
            for face in simplex.faces():
                if face not in self.simplices:
                    return False
        return True
    
    def sorted_simplices(self) -> List[Simplex]:
        """
        Get simplices sorted by filtration value, then dimension.
        """
        def sort_key(s: Simplex) -> Tuple[float, int, Tuple[int, ...]]:
            filt = self.filtration_values.get(s, 0.0)
            return (filt, s.dimension, s.vertices)
        
        return sorted(self.simplices, key=sort_key)


def pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distance matrix.
    
    Args:
        points: Point cloud, shape (n, d)
        
    Returns:
        Distance matrix, shape (n, n)
    """
    return torch.cdist(points, points)


@dataclass
class RipsComplex(SimplicialComplex):
    """
    Vietoris-Rips complex construction.
    
    A simplex is included if all pairwise distances are ≤ radius.
    
    Attributes:
        points: Original point cloud
        max_radius: Maximum filtration radius
    """
    points: Optional[torch.Tensor] = None
    max_radius: float = 1.0
    
    @classmethod
    def from_points(cls, points: torch.Tensor,
                    max_radius: float = 1.0,
                    max_dim: int = 2) -> 'RipsComplex':
        """
        Build Rips complex from point cloud.
        
        Args:
            points: Point cloud, shape (n, d)
            max_radius: Maximum edge length
            max_dim: Maximum simplex dimension
            
        Returns:
            Rips complex
        """
        n = points.shape[0]
        dist = pairwise_distances(points)
        
        complex = cls(points=points, max_radius=max_radius)
        
        # Add vertices at filtration value 0
        for i in range(n):
            simplex = Simplex((i,))
            complex.add_simplex(simplex, filtration_value=0.0, add_faces=False)
        
        # Add edges and higher simplices
        for k in range(1, max_dim + 1):
            for vertices in combinations(range(n), k + 1):
                # Check if all pairwise distances are within threshold
                max_dist = 0.0
                valid = True
                
                for i, j in combinations(vertices, 2):
                    d = dist[i, j].item()
                    if d > max_radius:
                        valid = False
                        break
                    max_dist = max(max_dist, d)
                
                if valid:
                    simplex = Simplex(vertices)
                    complex.add_simplex(simplex, filtration_value=max_dist, add_faces=False)
        
        complex.max_dim = min(max_dim, n - 1)
        return complex


# Alias for compatibility
VietorisRips = RipsComplex


@dataclass
class CechComplex(SimplicialComplex):
    """
    Čech complex construction.
    
    A simplex is included if the balls of radius r centered at the
    vertices have non-empty intersection.
    
    For points in Euclidean space, this is the nerve of the ball cover.
    """
    points: Optional[torch.Tensor] = None
    max_radius: float = 1.0
    
    @classmethod
    def from_points(cls, points: torch.Tensor,
                    max_radius: float = 1.0,
                    max_dim: int = 2) -> 'CechComplex':
        """
        Build Čech complex from point cloud.
        
        Uses miniball computation for exact Čech (simplified to 
        circumradius for triangles, etc.)
        
        Args:
            points: Point cloud, shape (n, d)
            max_radius: Maximum filtration radius
            max_dim: Maximum simplex dimension
            
        Returns:
            Čech complex
        """
        n = points.shape[0]
        
        complex = cls(points=points, max_radius=max_radius)
        
        # Add vertices
        for i in range(n):
            simplex = Simplex((i,))
            complex.add_simplex(simplex, filtration_value=0.0, add_faces=False)
        
        # Add edges (radius = distance/2)
        dist = pairwise_distances(points)
        for i, j in combinations(range(n), 2):
            r = dist[i, j].item() / 2
            if r <= max_radius:
                simplex = Simplex((i, j))
                complex.add_simplex(simplex, filtration_value=r, add_faces=False)
        
        # Add higher simplices using circumradius approximation
        for k in range(2, max_dim + 1):
            for vertices in combinations(range(n), k + 1):
                # Approximate: use maximum pairwise distance / 2
                # (True Čech uses smallest enclosing ball)
                max_dist = 0.0
                for i, j in combinations(vertices, 2):
                    max_dist = max(max_dist, dist[vertices[i - vertices[0]], 
                                                  vertices[j - vertices[0]]].item() 
                                   if i < len(vertices) and j < len(vertices) else dist[i, j].item())
                
                # Recompute properly
                simplex_dists = []
                for vi, vj in combinations(vertices, 2):
                    simplex_dists.append(dist[vi, vj].item())
                
                r = max(simplex_dists) / 2 if simplex_dists else 0.0
                
                if r <= max_radius:
                    simplex = Simplex(vertices)
                    complex.add_simplex(simplex, filtration_value=r, add_faces=False)
        
        complex.max_dim = min(max_dim, n - 1)
        return complex


def alpha_complex_2d(points: torch.Tensor) -> SimplicialComplex:
    """
    Alpha complex for 2D point cloud (simplified).
    
    The alpha complex is a subcomplex of the Delaunay triangulation.
    
    Args:
        points: 2D point cloud, shape (n, 2)
        
    Returns:
        Alpha complex (simplified construction)
    """
    # For a proper implementation, use Delaunay triangulation
    # Here we use Rips as approximation
    
    # Estimate appropriate radius from point density
    n = points.shape[0]
    dist = pairwise_distances(points)
    
    # Use median nearest neighbor distance
    nn_dist = dist.topk(2, dim=1, largest=False)[0][:, 1]
    radius = nn_dist.median().item() * 2
    
    return RipsComplex.from_points(points, max_radius=radius, max_dim=2)
