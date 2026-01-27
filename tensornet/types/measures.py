"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                       M E A S U R E S   M O D U L E                                     ║
║                                                                                          ║
║     Type-safe measure theory for probability and integration.                           ║
║     These are the "distributions" in geometric computation.                             ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Measures formalize probability distributions and integration:
    μ: Σ → [0, ∞]  (σ-algebra to extended reals)

Types of measures:
    Measure[M]              - General measure on space M
    ProbabilityMeasure[M]   - Normalized: μ(M) = 1
    LebesgueMeasure[M]      - Standard volume measure
    DiracMeasure[M]         - Point mass δ_x
    GaussianMeasure[M]      - Gaussian/normal distribution
    HaarMeasure[G]          - Invariant measure on Lie group

Key operations:
    - Integration: ∫ f dμ
    - Pushforward: φ_* μ (transport through map φ)
    - Optimal transport: W_p(μ, ν)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Optional, Tuple, Dict, Any,
    Type, Union, Callable, List
)
import torch
from torch import Tensor

from tensornet.types.spaces import Space, EuclideanSpace, R1, R3
from tensornet.types.constraints import Constraint, Probability, Normalized, Positive


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

S = TypeVar("S", bound=Space)
M = TypeVar("M", bound="Measure")


# ═══════════════════════════════════════════════════════════════════════════════
# BASE MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Measure(ABC, Generic[S]):
    """
    Abstract base class for measures on a space.
    
    A measure μ assigns a non-negative "size" to subsets of a space.
    
    Representations:
    - Density function ρ(x) w.r.t. Lebesgue: dμ = ρ dV
    - Discrete: sum of weighted Dirac masses
    - Implicit: defined by sampling or moments
    """
    
    space: S
    
    @abstractmethod
    def density(self, point: Tensor) -> Tensor:
        """
        Evaluate the density function at a point.
        
        For absolutely continuous measures: dμ/dλ where λ is Lebesgue.
        """
        ...
    
    @abstractmethod
    def sample(self, n: int) -> Tensor:
        """
        Draw n samples from this measure.
        
        Returns:
            Tensor of shape [n, dim]
        """
        ...
    
    @abstractmethod
    def total_mass(self) -> float:
        """
        Compute the total mass μ(space).
        
        For probability measures, this should be 1.
        """
        ...
    
    def integrate(self, f: Callable[[Tensor], Tensor], n_samples: int = 10000) -> Tensor:
        """
        Compute ∫ f dμ using Monte Carlo integration.
        """
        samples = self.sample(n_samples)
        f_values = f(samples)
        return f_values.mean() * self.total_mass()
    
    def expectation(self, f: Callable[[Tensor], Tensor], n_samples: int = 10000) -> Tensor:
        """
        Compute E_μ[f] = ∫ f dμ / μ(space).
        
        For probability measures, this equals ∫ f dμ.
        """
        return self.integrate(f, n_samples) / self.total_mass()
    
    def mean(self, n_samples: int = 10000) -> Tensor:
        """Compute the mean (center of mass)."""
        samples = self.sample(n_samples)
        return samples.mean(dim=0)
    
    def variance(self, n_samples: int = 10000) -> Tensor:
        """Compute the variance."""
        samples = self.sample(n_samples)
        mean = samples.mean(dim=0, keepdim=True)
        return ((samples - mean) ** 2).mean(dim=0)
    
    def covariance(self, n_samples: int = 10000) -> Tensor:
        """Compute the covariance matrix."""
        samples = self.sample(n_samples)
        mean = samples.mean(dim=0, keepdim=True)
        centered = samples - mean
        return (centered.T @ centered) / (n_samples - 1)
    
    def pushforward(self, phi: Callable[[Tensor], Tensor]) -> "PushforwardMeasure[S]":
        """
        Compute pushforward measure φ_* μ.
        
        (φ_* μ)(A) = μ(φ^{-1}(A))
        
        Sampling: sample from μ, then apply φ.
        """
        return PushforwardMeasure(base_measure=self, map_fn=phi)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProbabilityMeasure(Measure[S]):
    """
    Probability measure: μ(space) = 1.
    
    This is the foundational constraint for probabilistic reasoning.
    All probability measures must satisfy normalization.
    """
    
    _density_fn: Optional[Callable[[Tensor], Tensor]] = None
    _sampler: Optional[Callable[[int], Tensor]] = None
    
    def density(self, point: Tensor) -> Tensor:
        if self._density_fn is not None:
            return self._density_fn(point)
        raise NotImplementedError("Density not defined")
    
    def sample(self, n: int) -> Tensor:
        if self._sampler is not None:
            return self._sampler(n)
        raise NotImplementedError("Sampler not defined")
    
    def total_mass(self) -> float:
        return 1.0  # By definition
    
    def entropy(self, n_samples: int = 10000) -> Tensor:
        """
        Compute differential entropy H[μ] = -∫ ρ log ρ dV.
        """
        samples = self.sample(n_samples)
        log_p = torch.log(self.density(samples) + 1e-10)
        return -log_p.mean()
    
    def kl_divergence(self, other: "ProbabilityMeasure[S]", n_samples: int = 10000) -> Tensor:
        """
        Compute KL divergence D_KL(self || other) = ∫ ρ log(ρ/σ) dV.
        """
        samples = self.sample(n_samples)
        log_p = torch.log(self.density(samples) + 1e-10)
        log_q = torch.log(other.density(samples) + 1e-10)
        return (log_p - log_q).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# LEBESGUE MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LebesgueMeasure(Measure[S]):
    """
    Lebesgue measure (standard volume measure).
    
    For R^n: dλ = dx_1 dx_2 ... dx_n
    
    This is the reference measure for defining densities.
    """
    
    bounds: Optional[Tuple[Tensor, Tensor]] = None  # (lower, upper)
    
    def density(self, point: Tensor) -> Tensor:
        """Lebesgue density is 1 everywhere (or 1/volume for bounded)."""
        if self.bounds is None:
            return torch.ones(point.shape[:-1], device=point.device)
        else:
            # Uniform density on bounded region
            volume = (self.bounds[1] - self.bounds[0]).prod()
            return torch.ones(point.shape[:-1], device=point.device) / volume
    
    def sample(self, n: int) -> Tensor:
        """Sample uniformly from bounded region."""
        if self.bounds is None:
            raise ValueError("Cannot sample from unbounded Lebesgue measure")
        
        lower, upper = self.bounds
        dim = lower.shape[0]
        samples = torch.rand(n, dim, device=lower.device)
        return lower + samples * (upper - lower)
    
    def total_mass(self) -> float:
        if self.bounds is None:
            return float('inf')
        return (self.bounds[1] - self.bounds[0]).prod().item()


# ═══════════════════════════════════════════════════════════════════════════════
# DIRAC MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiracMeasure(Measure[S]):
    """
    Dirac delta measure δ_x.
    
    Concentrated at a single point x.
    δ_x(A) = 1 if x ∈ A, else 0.
    ∫ f dδ_x = f(x)
    """
    
    point: Tensor
    weight: float = 1.0
    
    def density(self, x: Tensor) -> Tensor:
        """
        Dirac delta is a distribution, not a function.
        Returns a very peaked Gaussian approximation.
        """
        eps = 1e-6
        dist_sq = ((x - self.point) ** 2).sum(dim=-1)
        return self.weight * torch.exp(-dist_sq / (2 * eps ** 2)) / (eps * math.sqrt(2 * math.pi))
    
    def sample(self, n: int) -> Tensor:
        """All samples are at the point."""
        return self.point.unsqueeze(0).expand(n, -1).clone()
    
    def total_mass(self) -> float:
        return self.weight
    
    def integrate(self, f: Callable[[Tensor], Tensor], n_samples: int = 1) -> Tensor:
        """∫ f dδ_x = f(x)."""
        return self.weight * f(self.point.unsqueeze(0)).squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GaussianMeasure(ProbabilityMeasure[S]):
    """
    Gaussian (normal) probability measure.
    
    ρ(x) = (2π)^{-d/2} |Σ|^{-1/2} exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
    """
    
    mean_vec: Tensor = field(default_factory=lambda: torch.zeros(1))
    covariance_matrix: Tensor = field(default_factory=lambda: torch.eye(1))
    
    def __post_init__(self):
        super().__init__(space=self.space)
        # Precompute for efficiency
        self._dim = self.mean_vec.shape[0]
        self._cov_inv = torch.linalg.inv(self.covariance_matrix)
        self._cov_det = torch.linalg.det(self.covariance_matrix)
        self._norm_const = (2 * math.pi) ** (self._dim / 2) * torch.sqrt(self._cov_det)
    
    def density(self, point: Tensor) -> Tensor:
        """Evaluate Gaussian density."""
        centered = point - self.mean_vec
        quad_form = torch.einsum("...i,ij,...j->...", centered, self._cov_inv, centered)
        return torch.exp(-0.5 * quad_form) / self._norm_const
    
    def sample(self, n: int) -> Tensor:
        """Sample from multivariate Gaussian."""
        # Cholesky decomposition: Σ = LLᵀ
        L = torch.linalg.cholesky(self.covariance_matrix)
        z = torch.randn(n, self._dim, device=self.mean_vec.device)
        return self.mean_vec + z @ L.T
    
    def entropy(self, n_samples: int = 0) -> Tensor:
        """
        Entropy of Gaussian: H = ½ log((2πe)^d |Σ|)
        
        Exact formula, no sampling needed.
        """
        return 0.5 * (self._dim * (1 + math.log(2 * math.pi)) + 
                     torch.log(self._cov_det))
    
    def kl_divergence(self, other: "GaussianMeasure[S]", n_samples: int = 0) -> Tensor:
        """
        KL divergence between Gaussians.
        
        D_KL(p || q) = ½[tr(Σ_q⁻¹Σ_p) + (μ_q-μ_p)ᵀΣ_q⁻¹(μ_q-μ_p) - d + log(|Σ_q|/|Σ_p|)]
        """
        d = self._dim
        
        mu_diff = other.mean_vec - self.mean_vec
        
        trace_term = torch.trace(other._cov_inv @ self.covariance_matrix)
        quad_term = mu_diff @ other._cov_inv @ mu_diff
        log_det_term = torch.log(other._cov_det / self._cov_det)
        
        return 0.5 * (trace_term + quad_term - d + log_det_term)
    
    @classmethod
    def standard(cls, dim: int, space: S) -> "GaussianMeasure[S]":
        """Create standard Gaussian N(0, I)."""
        return cls(
            space=space,
            mean_vec=torch.zeros(dim),
            covariance_matrix=torch.eye(dim)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HAAR MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HaarMeasure(Measure[S]):
    """
    Haar measure on a Lie group.
    
    The (left) Haar measure is the unique (up to scaling) left-invariant measure:
    μ(gA) = μ(A) for all g ∈ G.
    
    Examples:
        - SO(n): uniform on rotation matrices
        - U(n): uniform on unitary matrices
        - R^n: Lebesgue measure
    """
    
    group_type: str = "SO3"  # SO2, SO3, U1, SU2, etc.
    
    def density(self, point: Tensor) -> Tensor:
        """
        Haar density in given coordinates.
        
        For compact groups with appropriate normalization.
        """
        # For rotation matrices, density is uniform
        return torch.ones(point.shape[:-2], device=point.device)
    
    def sample(self, n: int) -> Tensor:
        """Sample uniformly from the Lie group."""
        if self.group_type == "SO2":
            return self._sample_so2(n)
        elif self.group_type == "SO3":
            return self._sample_so3(n)
        elif self.group_type == "U1":
            return self._sample_u1(n)
        else:
            raise NotImplementedError(f"Sampling for {self.group_type}")
    
    def _sample_so2(self, n: int) -> Tensor:
        """Sample from SO(2) - 2D rotations."""
        theta = 2 * math.pi * torch.rand(n)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1)
        ], dim=-2)
        return R
    
    def _sample_so3(self, n: int) -> Tensor:
        """Sample from SO(3) - 3D rotations using quaternions."""
        # Sample uniform quaternion, convert to rotation matrix
        u = torch.rand(n, 3)
        q = torch.stack([
            torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
            torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
            torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
            torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2])
        ], dim=-1)
        
        # Quaternion to rotation matrix
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
            torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
            torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
        ], dim=-2)
        return R
    
    def _sample_u1(self, n: int) -> Tensor:
        """Sample from U(1) - phases."""
        theta = 2 * math.pi * torch.rand(n)
        return torch.exp(1j * theta)
    
    def total_mass(self) -> float:
        """
        Total mass depends on normalization.
        
        For probability Haar measure, this is 1.
        """
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRETE MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiscreteMeasure(Measure[S]):
    """
    Discrete measure: sum of weighted Dirac masses.
    
    μ = Σᵢ wᵢ δ_{xᵢ}
    """
    
    points: Tensor  # [n, dim]
    weights: Tensor  # [n]
    
    def __post_init__(self):
        super().__init__(space=self.space)
        assert len(self.points) == len(self.weights)
        assert (self.weights >= 0).all(), "Weights must be non-negative"
    
    def density(self, point: Tensor) -> Tensor:
        """
        Discrete measure doesn't have a density in the usual sense.
        Returns sum of Gaussian approximations.
        """
        eps = 1e-4
        dists_sq = ((point.unsqueeze(-2) - self.points) ** 2).sum(dim=-1)
        gaussians = torch.exp(-dists_sq / (2 * eps ** 2)) / (eps * math.sqrt(2 * math.pi))
        return (gaussians * self.weights).sum(dim=-1)
    
    def sample(self, n: int) -> Tensor:
        """Sample from discrete distribution."""
        # Normalize weights to probabilities
        probs = self.weights / self.weights.sum()
        
        # Categorical sampling
        indices = torch.multinomial(probs, n, replacement=True)
        return self.points[indices]
    
    def total_mass(self) -> float:
        return self.weights.sum().item()
    
    def support(self) -> Tensor:
        """Return the support (points with positive weight)."""
        mask = self.weights > 0
        return self.points[mask]
    
    def normalize(self) -> "DiscreteMeasure[S]":
        """Return normalized version (probability measure)."""
        total = self.weights.sum()
        return DiscreteMeasure(
            space=self.space,
            points=self.points,
            weights=self.weights / total
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PUSHFORWARD MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PushforwardMeasure(Measure[S]):
    """
    Pushforward of a measure through a map.
    
    (φ_* μ)(A) = μ(φ^{-1}(A))
    
    If μ has density ρ and φ is a diffeomorphism:
    (φ_* μ) has density ρ ∘ φ^{-1} / |det Dφ|
    """
    
    base_measure: Measure
    map_fn: Callable[[Tensor], Tensor]
    inverse_fn: Optional[Callable[[Tensor], Tensor]] = None
    jacobian_fn: Optional[Callable[[Tensor], Tensor]] = None
    
    def density(self, point: Tensor) -> Tensor:
        """Compute density of pushforward."""
        if self.inverse_fn is None or self.jacobian_fn is None:
            raise NotImplementedError("Need inverse and Jacobian for density")
        
        # ρ_push(y) = ρ(φ^{-1}(y)) / |det Dφ(φ^{-1}(y))|
        x = self.inverse_fn(point)
        base_density = self.base_measure.density(x)
        jac = self.jacobian_fn(x)
        det_jac = torch.linalg.det(jac).abs()
        
        return base_density / det_jac
    
    def sample(self, n: int) -> Tensor:
        """Sample from pushforward: sample from base, apply map."""
        base_samples = self.base_measure.sample(n)
        return self.map_fn(base_samples)
    
    def total_mass(self) -> float:
        """Pushforward preserves total mass."""
        return self.base_measure.total_mass()


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT MEASURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProductMeasure(Measure):
    """
    Product measure μ × ν on M × N.
    
    (μ × ν)(A × B) = μ(A) · ν(B)
    """
    
    measure1: Measure
    measure2: Measure
    
    def density(self, point: Tensor) -> Tensor:
        """Density is product of marginal densities."""
        d1 = self.measure1.space.dimension
        p1 = point[..., :d1]
        p2 = point[..., d1:]
        return self.measure1.density(p1) * self.measure2.density(p2)
    
    def sample(self, n: int) -> Tensor:
        """Sample independently from each marginal."""
        s1 = self.measure1.sample(n)
        s2 = self.measure2.sample(n)
        return torch.cat([s1, s2], dim=-1)
    
    def total_mass(self) -> float:
        return self.measure1.total_mass() * self.measure2.total_mass()


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def uniform(space: S, bounds: Tuple[Tensor, Tensor]) -> ProbabilityMeasure[S]:
    """Create uniform distribution on bounded region."""
    lebesgue = LebesgueMeasure(space=space, bounds=bounds)
    volume = (bounds[1] - bounds[0]).prod().item()
    
    return ProbabilityMeasure(
        space=space,
        _density_fn=lambda x: torch.ones(x.shape[:-1], device=x.device) / volume,
        _sampler=lambda n: lebesgue.sample(n)
    )


def gaussian(
    space: S,
    mean: Tensor,
    covariance: Optional[Tensor] = None,
    std: Optional[float] = None
) -> GaussianMeasure[S]:
    """Create Gaussian distribution."""
    dim = mean.shape[0]
    
    if covariance is not None:
        cov = covariance
    elif std is not None:
        cov = std ** 2 * torch.eye(dim, device=mean.device)
    else:
        cov = torch.eye(dim, device=mean.device)
    
    return GaussianMeasure(
        space=space,
        mean_vec=mean,
        covariance_matrix=cov
    )


def point_mass(space: S, point: Tensor, weight: float = 1.0) -> DiracMeasure[S]:
    """Create Dirac delta at a point."""
    return DiracMeasure(space=space, point=point, weight=weight)


def empirical(space: S, samples: Tensor) -> DiscreteMeasure[S]:
    """Create empirical measure from samples."""
    n = samples.shape[0]
    return DiscreteMeasure(
        space=space,
        points=samples,
        weights=torch.ones(n, device=samples.device) / n
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "Measure",
    # Probability
    "ProbabilityMeasure",
    # Standard measures
    "LebesgueMeasure", "DiracMeasure", "GaussianMeasure", "HaarMeasure",
    # Discrete
    "DiscreteMeasure",
    # Transforms
    "PushforwardMeasure", "ProductMeasure",
    # Factories
    "uniform", "gaussian", "point_mass", "empirical",
]
