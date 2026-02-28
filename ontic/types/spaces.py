"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                        S P A C E S   M O D U L E                                        ║
║                                                                                          ║
║     Type-safe representations of mathematical spaces.                                   ║
║     These are the "where" of geometric computation.                                     ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Spaces form the foundation of the geometric type system. Every field, measure, and
operator is defined over a space, and the type system ensures operations are
compatible between spaces.

Hierarchy:
    Space (abstract)
    ├── EuclideanSpace[dim]  →  R1, R2, R3, R4
    ├── Sphere[dim]          →  S1, S2, S3
    ├── Torus[dim]           →  T2, T3
    ├── Manifold[dim, metric, topology]
    ├── TangentSpace[base_space, point]
    └── CotangentSpace[base_space, point]
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, ClassVar, Optional, Tuple, Dict, Any,
    Type, Union, Callable, List, Literal, Protocol, runtime_checkable
)
from enum import Enum, auto
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

Dim = TypeVar("Dim", bound=int)
S = TypeVar("S", bound="Space")
M = TypeVar("M", bound="Manifold")


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class Topology(Enum):
    """Topological properties of a space."""
    EUCLIDEAN = auto()      # R^n topology
    COMPACT = auto()        # Closed and bounded
    CONNECTED = auto()      # Path-connected
    SIMPLY_CONNECTED = auto()  # No holes
    ORIENTABLE = auto()     # Has consistent orientation
    PERIODIC = auto()       # Toroidal/periodic boundaries


class Boundary(Enum):
    """Boundary conditions for a space."""
    NONE = auto()           # Open space, no boundary
    DIRICHLET = auto()      # Fixed values at boundary
    NEUMANN = auto()        # Fixed derivatives at boundary
    PERIODIC = auto()       # Wrap-around
    REFLECTING = auto()     # Mirror at boundary
    ABSORBING = auto()      # Zero at boundary


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MetricSignature:
    """
    Signature of a pseudo-Riemannian metric.
    
    (p, q) means p positive eigenvalues, q negative eigenvalues.
    - (n, 0): Riemannian (positive definite)
    - (3, 1): Lorentzian (Minkowski spacetime)
    - (p, q): General pseudo-Riemannian
    """
    positive: int
    negative: int
    
    @property
    def dimension(self) -> int:
        return self.positive + self.negative
    
    @property
    def is_riemannian(self) -> bool:
        return self.negative == 0
    
    @property
    def is_lorentzian(self) -> bool:
        return self.negative == 1
    
    def __str__(self) -> str:
        return f"({self.positive}, {self.negative})"


# Standard signatures
EUCLIDEAN_1D = MetricSignature(1, 0)
EUCLIDEAN_2D = MetricSignature(2, 0)
EUCLIDEAN_3D = MetricSignature(3, 0)
EUCLIDEAN_4D = MetricSignature(4, 0)
MINKOWSKI = MetricSignature(3, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE SPACE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Space(ABC):
    """
    Abstract base class for all mathematical spaces.
    
    A Space defines:
    - Dimension
    - Metric structure
    - Topological properties
    - Coordinate charts
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimension of this space."""
        ...
    
    @property
    @abstractmethod
    def metric_signature(self) -> MetricSignature:
        """The metric signature (p, q) of this space."""
        ...
    
    @property
    def is_riemannian(self) -> bool:
        """True if the metric is positive definite."""
        return self.metric_signature.is_riemannian
    
    @property
    def is_lorentzian(self) -> bool:
        """True if the metric has signature (n-1, 1)."""
        return self.metric_signature.is_lorentzian
    
    @abstractmethod
    def metric_at(self, point: Tensor) -> Tensor:
        """
        Compute the metric tensor at a point.
        
        Args:
            point: Coordinates of the point, shape [..., dim]
            
        Returns:
            Metric tensor, shape [..., dim, dim]
        """
        ...
    
    @abstractmethod
    def christoffel_at(self, point: Tensor) -> Tensor:
        """
        Compute Christoffel symbols at a point.
        
        Args:
            point: Coordinates, shape [..., dim]
            
        Returns:
            Christoffel symbols Γ^i_jk, shape [..., dim, dim, dim]
        """
        ...
    
    @abstractmethod
    def volume_element(self, point: Tensor) -> Tensor:
        """
        Compute the volume element sqrt(|det g|) at a point.
        
        Args:
            point: Coordinates, shape [..., dim]
            
        Returns:
            Volume element, shape [...]
        """
        ...
    
    def geodesic_distance(self, p1: Tensor, p2: Tensor) -> Tensor:
        """
        Compute geodesic distance between two points.
        
        Default implementation uses Euclidean distance.
        Override for curved spaces.
        """
        return torch.norm(p2 - p1, dim=-1)
    
    @abstractmethod
    def is_inside(self, point: Tensor) -> Tensor:
        """
        Check if a point is inside this space.
        
        Returns:
            Boolean tensor, shape [...]
        """
        ...
    
    def tangent_space_at(self, point: Tensor) -> "TangentSpace":
        """Get the tangent space at a point."""
        return TangentSpace(self, point)
    
    def cotangent_space_at(self, point: Tensor) -> "CotangentSpace":
        """Get the cotangent space at a point."""
        return CotangentSpace(self, point)


# ═══════════════════════════════════════════════════════════════════════════════
# EUCLIDEAN SPACES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EuclideanSpace(Space):
    """
    Euclidean space R^n with flat metric.
    
    The metric is the identity matrix everywhere:
        g_ij = δ_ij
    
    All Christoffel symbols vanish.
    """
    
    dim: int
    bounds: Optional[Tuple[Tensor, Tensor]] = None  # (lower, upper) bounds
    boundary: Boundary = Boundary.NONE
    
    @property
    def dimension(self) -> int:
        return self.dim
    
    @property
    def metric_signature(self) -> MetricSignature:
        return MetricSignature(self.dim, 0)
    
    def metric_at(self, point: Tensor) -> Tensor:
        """Metric is identity everywhere."""
        batch_shape = point.shape[:-1]
        eye = torch.eye(self.dim, device=point.device, dtype=point.dtype)
        return eye.expand(*batch_shape, self.dim, self.dim)
    
    def christoffel_at(self, point: Tensor) -> Tensor:
        """Christoffel symbols vanish in Euclidean space."""
        batch_shape = point.shape[:-1]
        return torch.zeros(*batch_shape, self.dim, self.dim, self.dim,
                          device=point.device, dtype=point.dtype)
    
    def volume_element(self, point: Tensor) -> Tensor:
        """Volume element is 1 everywhere in Euclidean space."""
        return torch.ones(point.shape[:-1], device=point.device, dtype=point.dtype)
    
    def is_inside(self, point: Tensor) -> Tensor:
        """Check if point is within bounds."""
        if self.bounds is None:
            return torch.ones(point.shape[:-1], dtype=torch.bool, device=point.device)
        lower, upper = self.bounds
        inside = (point >= lower) & (point <= upper)
        return inside.all(dim=-1)
    
    def __repr__(self) -> str:
        return f"R{self.dim}"


# Canonical Euclidean spaces
R1 = EuclideanSpace(1)
R2 = EuclideanSpace(2)
R3 = EuclideanSpace(3)
R4 = EuclideanSpace(4)


# ═══════════════════════════════════════════════════════════════════════════════
# SPHERES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Sphere(Space):
    """
    n-dimensional sphere S^n embedded in R^(n+1).
    
    The sphere of radius r is the set {x ∈ R^(n+1) : |x| = r}.
    It has dimension n and is compact.
    
    We use stereographic coordinates for charts.
    """
    
    dim: int
    radius: float = 1.0
    
    @property
    def dimension(self) -> int:
        return self.dim
    
    @property
    def metric_signature(self) -> MetricSignature:
        return MetricSignature(self.dim, 0)
    
    def embedding_dimension(self) -> int:
        """Dimension of the ambient Euclidean space."""
        return self.dim + 1
    
    def metric_at(self, point: Tensor) -> Tensor:
        """
        Metric in spherical coordinates.
        For S^2 in (θ, φ): g = r² diag(1, sin²θ)
        """
        batch_shape = point.shape[:-1]
        
        if self.dim == 1:
            # S^1: metric is r²
            return (self.radius ** 2) * torch.ones(*batch_shape, 1, 1,
                                                   device=point.device, dtype=point.dtype)
        elif self.dim == 2:
            # S^2: θ ∈ [0, π], φ ∈ [0, 2π)
            theta = point[..., 0]
            g = torch.zeros(*batch_shape, 2, 2, device=point.device, dtype=point.dtype)
            g[..., 0, 0] = self.radius ** 2
            g[..., 1, 1] = (self.radius * torch.sin(theta)) ** 2
            return g
        else:
            # General case: would need full spherical coordinate metric
            raise NotImplementedError(f"Metric for S^{self.dim} not yet implemented")
    
    def christoffel_at(self, point: Tensor) -> Tensor:
        """Christoffel symbols for the sphere."""
        batch_shape = point.shape[:-1]
        gamma = torch.zeros(*batch_shape, self.dim, self.dim, self.dim,
                           device=point.device, dtype=point.dtype)
        
        if self.dim == 2:
            # S^2 Christoffel symbols
            theta = point[..., 0]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            
            # Γ^θ_φφ = -sin(θ)cos(θ)
            gamma[..., 0, 1, 1] = -sin_theta * cos_theta
            # Γ^φ_θφ = Γ^φ_φθ = cot(θ)
            cot_theta = cos_theta / (sin_theta + 1e-8)
            gamma[..., 1, 0, 1] = cot_theta
            gamma[..., 1, 1, 0] = cot_theta
        
        return gamma
    
    def volume_element(self, point: Tensor) -> Tensor:
        """Volume element for the sphere."""
        if self.dim == 1:
            return self.radius * torch.ones(point.shape[:-1], device=point.device)
        elif self.dim == 2:
            theta = point[..., 0]
            return (self.radius ** 2) * torch.sin(theta)
        else:
            raise NotImplementedError(f"Volume element for S^{self.dim}")
    
    def is_inside(self, point: Tensor) -> Tensor:
        """
        For spherical coordinates, check that angles are in valid range.
        """
        # Simplified: assume point is in embedding coordinates
        r = torch.norm(point, dim=-1)
        return torch.abs(r - self.radius) < 1e-6
    
    def to_embedding(self, spherical: Tensor) -> Tensor:
        """Convert spherical coordinates to Cartesian embedding."""
        if self.dim == 1:
            theta = spherical[..., 0]
            x = self.radius * torch.cos(theta)
            y = self.radius * torch.sin(theta)
            return torch.stack([x, y], dim=-1)
        elif self.dim == 2:
            theta = spherical[..., 0]  # polar angle
            phi = spherical[..., 1]    # azimuthal angle
            x = self.radius * torch.sin(theta) * torch.cos(phi)
            y = self.radius * torch.sin(theta) * torch.sin(phi)
            z = self.radius * torch.cos(theta)
            return torch.stack([x, y, z], dim=-1)
        else:
            raise NotImplementedError(f"Embedding for S^{self.dim}")
    
    def from_embedding(self, cartesian: Tensor) -> Tensor:
        """Convert Cartesian embedding to spherical coordinates."""
        if self.dim == 1:
            x, y = cartesian[..., 0], cartesian[..., 1]
            theta = torch.atan2(y, x)
            return theta.unsqueeze(-1)
        elif self.dim == 2:
            x, y, z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
            r = torch.norm(cartesian, dim=-1)
            theta = torch.acos(z / (r + 1e-8))
            phi = torch.atan2(y, x)
            return torch.stack([theta, phi], dim=-1)
        else:
            raise NotImplementedError(f"From embedding for S^{self.dim}")
    
    def __repr__(self) -> str:
        if self.radius == 1.0:
            return f"S{self.dim}"
        return f"S{self.dim}(r={self.radius})"


# Canonical spheres
S1 = Sphere(1)  # Circle
S2 = Sphere(2)  # 2-sphere
S3 = Sphere(3)  # 3-sphere


# ═══════════════════════════════════════════════════════════════════════════════
# TORUS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Torus(Space):
    """
    n-dimensional torus T^n = S^1 × S^1 × ... × S^1.
    
    The torus is a flat space with periodic boundary conditions.
    """
    
    dim: int
    periods: Optional[Tuple[float, ...]] = None  # Period in each dimension
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = tuple(2 * math.pi for _ in range(self.dim))
    
    @property
    def dimension(self) -> int:
        return self.dim
    
    @property
    def metric_signature(self) -> MetricSignature:
        return MetricSignature(self.dim, 0)
    
    def metric_at(self, point: Tensor) -> Tensor:
        """Flat metric on torus."""
        batch_shape = point.shape[:-1]
        eye = torch.eye(self.dim, device=point.device, dtype=point.dtype)
        return eye.expand(*batch_shape, self.dim, self.dim)
    
    def christoffel_at(self, point: Tensor) -> Tensor:
        """Christoffel symbols vanish on flat torus."""
        batch_shape = point.shape[:-1]
        return torch.zeros(*batch_shape, self.dim, self.dim, self.dim,
                          device=point.device, dtype=point.dtype)
    
    def volume_element(self, point: Tensor) -> Tensor:
        """Volume element is 1 for flat torus."""
        return torch.ones(point.shape[:-1], device=point.device, dtype=point.dtype)
    
    def is_inside(self, point: Tensor) -> Tensor:
        """All points are inside the torus (periodic)."""
        return torch.ones(point.shape[:-1], dtype=torch.bool, device=point.device)
    
    def wrap(self, point: Tensor) -> Tensor:
        """Wrap coordinates to fundamental domain [0, period)."""
        periods = torch.tensor(self.periods, device=point.device, dtype=point.dtype)
        return torch.remainder(point, periods)
    
    def __repr__(self) -> str:
        return f"T{self.dim}"


# Canonical tori
T2 = Torus(2)  # 2-torus
T3 = Torus(3)  # 3-torus


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL MANIFOLD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Manifold(Space):
    """
    General Riemannian or pseudo-Riemannian manifold.
    
    Defined by a metric tensor field g_ij(x).
    """
    
    dim: int
    signature: MetricSignature
    metric_fn: Callable[[Tensor], Tensor]  # x → g_ij(x)
    name: str = "M"
    
    @property
    def dimension(self) -> int:
        return self.dim
    
    @property
    def metric_signature(self) -> MetricSignature:
        return self.signature
    
    def metric_at(self, point: Tensor) -> Tensor:
        return self.metric_fn(point)
    
    def christoffel_at(self, point: Tensor) -> Tensor:
        """
        Compute Christoffel symbols numerically from the metric.
        
        Γ^k_ij = (1/2) g^kl (∂_i g_lj + ∂_j g_il - ∂_l g_ij)
        """
        eps = 1e-5
        batch_shape = point.shape[:-1]
        d = self.dim
        
        # Get metric and its inverse
        g = self.metric_at(point)
        g_inv = torch.linalg.inv(g)
        
        # Compute metric derivatives numerically
        dg = torch.zeros(*batch_shape, d, d, d, device=point.device, dtype=point.dtype)
        
        for l in range(d):
            delta = torch.zeros_like(point)
            delta[..., l] = eps
            g_plus = self.metric_at(point + delta)
            g_minus = self.metric_at(point - delta)
            dg[..., l, :, :] = (g_plus - g_minus) / (2 * eps)
        
        # Compute Christoffel symbols
        gamma = torch.zeros(*batch_shape, d, d, d, device=point.device, dtype=point.dtype)
        
        for k in range(d):
            for i in range(d):
                for j in range(d):
                    for l in range(d):
                        gamma[..., k, i, j] += 0.5 * g_inv[..., k, l] * (
                            dg[..., i, l, j] + dg[..., j, i, l] - dg[..., l, i, j]
                        )
        
        return gamma
    
    def volume_element(self, point: Tensor) -> Tensor:
        g = self.metric_at(point)
        return torch.sqrt(torch.abs(torch.linalg.det(g)))
    
    def is_inside(self, point: Tensor) -> Tensor:
        """Default: everywhere is inside."""
        return torch.ones(point.shape[:-1], dtype=torch.bool, device=point.device)
    
    def riemann_tensor_at(self, point: Tensor) -> Tensor:
        """
        Compute Riemann curvature tensor R^l_ijk.
        
        R^l_ijk = ∂_j Γ^l_ik - ∂_k Γ^l_ij + Γ^l_jm Γ^m_ik - Γ^l_km Γ^m_ij
        """
        eps = 1e-5
        d = self.dim
        batch_shape = point.shape[:-1]
        
        gamma = self.christoffel_at(point)
        
        # Compute derivatives of Christoffel symbols
        dgamma = torch.zeros(*batch_shape, d, d, d, d, device=point.device, dtype=point.dtype)
        for m in range(d):
            delta = torch.zeros_like(point)
            delta[..., m] = eps
            gamma_plus = self.christoffel_at(point + delta)
            gamma_minus = self.christoffel_at(point - delta)
            dgamma[..., m, :, :, :] = (gamma_plus - gamma_minus) / (2 * eps)
        
        # Build Riemann tensor
        R = torch.zeros(*batch_shape, d, d, d, d, device=point.device, dtype=point.dtype)
        
        for l in range(d):
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        R[..., l, i, j, k] = dgamma[..., j, l, i, k] - dgamma[..., k, l, i, j]
                        for m in range(d):
                            R[..., l, i, j, k] += (
                                gamma[..., l, j, m] * gamma[..., m, i, k] -
                                gamma[..., l, k, m] * gamma[..., m, i, j]
                            )
        
        return R
    
    def ricci_tensor_at(self, point: Tensor) -> Tensor:
        """Ricci tensor R_ij = R^k_ikj."""
        R = self.riemann_tensor_at(point)
        return torch.einsum("...kikj->...ij", R)
    
    def ricci_scalar_at(self, point: Tensor) -> Tensor:
        """Ricci scalar R = g^ij R_ij."""
        Ric = self.ricci_tensor_at(point)
        g = self.metric_at(point)
        g_inv = torch.linalg.inv(g)
        return torch.einsum("...ij,...ij->...", g_inv, Ric)
    
    def __repr__(self) -> str:
        return f"{self.name}^{self.dim}"


# ═══════════════════════════════════════════════════════════════════════════════
# TANGENT AND COTANGENT SPACES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TangentSpace(Space):
    """
    Tangent space T_p M at a point p of a manifold M.
    
    This is always a vector space (Euclidean) of dimension dim(M).
    """
    
    base_space: Space
    point: Tensor
    
    @property
    def dimension(self) -> int:
        return self.base_space.dimension
    
    @property
    def metric_signature(self) -> MetricSignature:
        return self.base_space.metric_signature
    
    def metric_at(self, vector: Tensor) -> Tensor:
        """Inner product inherited from base manifold metric at the base point."""
        return self.base_space.metric_at(self.point)
    
    def christoffel_at(self, vector: Tensor) -> Tensor:
        """Tangent space is flat - no Christoffel symbols."""
        batch_shape = vector.shape[:-1]
        d = self.dimension
        return torch.zeros(*batch_shape, d, d, d, device=vector.device, dtype=vector.dtype)
    
    def volume_element(self, vector: Tensor) -> Tensor:
        return self.base_space.volume_element(self.point)
    
    def is_inside(self, vector: Tensor) -> Tensor:
        """All vectors are in the tangent space."""
        return torch.ones(vector.shape[:-1], dtype=torch.bool, device=vector.device)
    
    def inner_product(self, v1: Tensor, v2: Tensor) -> Tensor:
        """Compute <v1, v2> using the base metric at this point."""
        g = self.base_space.metric_at(self.point)
        return torch.einsum("...i,...ij,...j->...", v1, g, v2)
    
    def norm(self, v: Tensor) -> Tensor:
        """Compute ||v|| using the base metric."""
        return torch.sqrt(torch.abs(self.inner_product(v, v)))
    
    def __repr__(self) -> str:
        return f"T_{self.point}({self.base_space})"


@dataclass
class CotangentSpace(Space):
    """
    Cotangent space T*_p M at a point p of a manifold M.
    
    Elements are linear functionals on the tangent space (1-forms at p).
    """
    
    base_space: Space
    point: Tensor
    
    @property
    def dimension(self) -> int:
        return self.base_space.dimension
    
    @property
    def metric_signature(self) -> MetricSignature:
        return self.base_space.metric_signature
    
    def metric_at(self, covector: Tensor) -> Tensor:
        """Inverse metric for cotangent space."""
        g = self.base_space.metric_at(self.point)
        return torch.linalg.inv(g)
    
    def christoffel_at(self, covector: Tensor) -> Tensor:
        batch_shape = covector.shape[:-1]
        d = self.dimension
        return torch.zeros(*batch_shape, d, d, d, device=covector.device, dtype=covector.dtype)
    
    def volume_element(self, covector: Tensor) -> Tensor:
        return 1.0 / self.base_space.volume_element(self.point)
    
    def is_inside(self, covector: Tensor) -> Tensor:
        return torch.ones(covector.shape[:-1], dtype=torch.bool, device=covector.device)
    
    def pair(self, covector: Tensor, vector: Tensor) -> Tensor:
        """Natural pairing <ω, v> between covector and vector."""
        return torch.einsum("...i,...i->...", covector, vector)
    
    def musical_sharp(self, covector: Tensor) -> Tensor:
        """Raise index: ω_i → g^ij ω_j (covector to vector)."""
        g_inv = self.metric_at(covector)
        return torch.einsum("...ij,...j->...i", g_inv, covector)
    
    def __repr__(self) -> str:
        return f"T*_{self.point}({self.base_space})"


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT SPACES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProductSpace(Space):
    """
    Cartesian product of spaces M × N.
    """
    
    space1: Space
    space2: Space
    
    @property
    def dimension(self) -> int:
        return self.space1.dimension + self.space2.dimension
    
    @property
    def metric_signature(self) -> MetricSignature:
        s1 = self.space1.metric_signature
        s2 = self.space2.metric_signature
        return MetricSignature(s1.positive + s2.positive, s1.negative + s2.negative)
    
    def metric_at(self, point: Tensor) -> Tensor:
        d1 = self.space1.dimension
        d2 = self.space2.dimension
        
        p1 = point[..., :d1]
        p2 = point[..., d1:]
        
        g1 = self.space1.metric_at(p1)
        g2 = self.space2.metric_at(p2)
        
        batch_shape = point.shape[:-1]
        g = torch.zeros(*batch_shape, self.dimension, self.dimension,
                       device=point.device, dtype=point.dtype)
        g[..., :d1, :d1] = g1
        g[..., d1:, d1:] = g2
        
        return g
    
    def christoffel_at(self, point: Tensor) -> Tensor:
        d1 = self.space1.dimension
        d2 = self.space2.dimension
        d = d1 + d2
        
        p1 = point[..., :d1]
        p2 = point[..., d1:]
        
        gamma1 = self.space1.christoffel_at(p1)
        gamma2 = self.space2.christoffel_at(p2)
        
        batch_shape = point.shape[:-1]
        gamma = torch.zeros(*batch_shape, d, d, d, device=point.device, dtype=point.dtype)
        gamma[..., :d1, :d1, :d1] = gamma1
        gamma[..., d1:, d1:, d1:] = gamma2
        
        return gamma
    
    def volume_element(self, point: Tensor) -> Tensor:
        d1 = self.space1.dimension
        p1 = point[..., :d1]
        p2 = point[..., d1:]
        return self.space1.volume_element(p1) * self.space2.volume_element(p2)
    
    def is_inside(self, point: Tensor) -> Tensor:
        d1 = self.space1.dimension
        p1 = point[..., :d1]
        p2 = point[..., d1:]
        return self.space1.is_inside(p1) & self.space2.is_inside(p2)
    
    def project1(self, point: Tensor) -> Tensor:
        """Project to first factor."""
        return point[..., :self.space1.dimension]
    
    def project2(self, point: Tensor) -> Tensor:
        """Project to second factor."""
        return point[..., self.space1.dimension:]
    
    def __repr__(self) -> str:
        return f"{self.space1} × {self.space2}"


# ═══════════════════════════════════════════════════════════════════════════════
# MINKOWSKI SPACETIME
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MinkowskiSpacetime(Space):
    """
    Minkowski spacetime R^{3,1} with metric η = diag(-1, 1, 1, 1).
    
    Convention: (t, x, y, z) with c = 1.
    """
    
    @property
    def dimension(self) -> int:
        return 4
    
    @property
    def metric_signature(self) -> MetricSignature:
        return MINKOWSKI
    
    def metric_at(self, point: Tensor) -> Tensor:
        batch_shape = point.shape[:-1]
        eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0],
                                       device=point.device, dtype=point.dtype))
        return eta.expand(*batch_shape, 4, 4)
    
    def christoffel_at(self, point: Tensor) -> Tensor:
        """Flat spacetime - no Christoffel symbols."""
        batch_shape = point.shape[:-1]
        return torch.zeros(*batch_shape, 4, 4, 4, device=point.device, dtype=point.dtype)
    
    def volume_element(self, point: Tensor) -> Tensor:
        return torch.ones(point.shape[:-1], device=point.device, dtype=point.dtype)
    
    def is_inside(self, point: Tensor) -> Tensor:
        return torch.ones(point.shape[:-1], dtype=torch.bool, device=point.device)
    
    def lorentz_factor(self, velocity: Tensor) -> Tensor:
        """Compute γ = 1/√(1 - v²/c²) for 3-velocity v."""
        v_squared = torch.sum(velocity ** 2, dim=-1)
        return 1.0 / torch.sqrt(1.0 - v_squared)
    
    def proper_time(self, worldline: Tensor) -> Tensor:
        """
        Compute proper time along a worldline.
        
        worldline: shape [..., N, 4] - N points in spacetime
        """
        dt = worldline[..., 1:, 0] - worldline[..., :-1, 0]
        dx = worldline[..., 1:, 1:] - worldline[..., :-1, 1:]
        ds_squared = dt ** 2 - torch.sum(dx ** 2, dim=-1)
        return torch.sum(torch.sqrt(torch.abs(ds_squared)), dim=-1)
    
    def __repr__(self) -> str:
        return "M^{3,1}"


# Canonical Minkowski spacetime
M31 = MinkowskiSpacetime()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "Space", "Topology", "Boundary", "MetricSignature",
    # Euclidean
    "EuclideanSpace", "R1", "R2", "R3", "R4",
    # Spheres
    "Sphere", "S1", "S2", "S3",
    # Tori
    "Torus", "T2", "T3",
    # General
    "Manifold", "ProductSpace",
    # Tangent/Cotangent
    "TangentSpace", "CotangentSpace",
    # Spacetime
    "MinkowskiSpacetime", "M31", "MINKOWSKI",
    # Signatures
    "EUCLIDEAN_1D", "EUCLIDEAN_2D", "EUCLIDEAN_3D", "EUCLIDEAN_4D",
]
