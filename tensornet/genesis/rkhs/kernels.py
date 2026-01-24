"""
QTT Kernel Functions

Implements common kernel functions for RKHS-based methods.

All kernels are positive definite and can be used to construct
QTT kernel matrices.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
import torch


class Kernel(ABC):
    """
    Abstract base class for kernel functions.
    
    A kernel k(x, y) must be symmetric and positive definite.
    """
    
    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate kernel between x and y.
        
        Args:
            x: First input, shape (n, d) or (d,)
            y: Second input, shape (m, d) or (d,)
            
        Returns:
            Kernel values, shape (n, m) or scalar
        """
        pass
    
    @abstractmethod
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal k(x_i, x_i) efficiently.
        
        Args:
            x: Input, shape (n, d)
            
        Returns:
            Diagonal values, shape (n,)
        """
        pass
    
    def matrix(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute full kernel matrix.
        
        Args:
            x: First input, shape (n, d)
            y: Second input, shape (m, d), or None for x
            
        Returns:
            Kernel matrix, shape (n, m)
        """
        if y is None:
            y = x
        
        # Ensure 2D
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        n, m = x.shape[0], y.shape[0]
        K = torch.zeros(n, m)
        
        for i in range(n):
            K[i] = self(x[i:i+1], y).squeeze(0)
        
        return K
    
    def __add__(self, other: 'Kernel') -> 'SumKernel':
        """Sum of two kernels."""
        return SumKernel(self, other)
    
    def __mul__(self, other: Union['Kernel', float]) -> 'Kernel':
        """Product of two kernels or scalar multiplication."""
        if isinstance(other, (int, float)):
            return ScaledKernel(self, other)
        return ProductKernel(self, other)
    
    def __rmul__(self, other: float) -> 'ScaledKernel':
        """Scalar multiplication."""
        return ScaledKernel(self, other)


@dataclass
class RBFKernel(Kernel):
    """
    Radial Basis Function (Gaussian) kernel.
    
    k(x, y) = σ² exp(-||x - y||² / (2ℓ²))
    
    Also known as squared exponential or Gaussian kernel.
    
    Attributes:
        length_scale: Characteristic length scale ℓ
        variance: Output variance σ²
    """
    length_scale: float = 1.0
    variance: float = 1.0
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        # Compute squared distances
        # ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (n, 1)
        y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # (m, 1)
        
        dist_sq = x_sq + y_sq.T - 2 * x @ y.T  # (n, m)
        dist_sq = torch.clamp(dist_sq, min=0)  # Numerical stability
        
        return self.variance * torch.exp(-dist_sq / (2 * self.length_scale ** 2))
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        # k(x, x) = σ² for all x
        return torch.full((x.shape[0],), self.variance)


@dataclass
class MaternKernel(Kernel):
    """
    Matérn kernel with smoothness parameter ν.
    
    For ν = 1/2: Ornstein-Uhlenbeck (exponential)
    For ν = 3/2: Once differentiable
    For ν = 5/2: Twice differentiable
    For ν → ∞: Approaches RBF
    
    Attributes:
        nu: Smoothness parameter (1/2, 3/2, or 5/2)
        length_scale: Characteristic length scale
        variance: Output variance
    """
    nu: float = 2.5
    length_scale: float = 1.0
    variance: float = 1.0
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        # Compute distances
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_sq = (y ** 2).sum(dim=-1, keepdim=True)
        dist_sq = x_sq + y_sq.T - 2 * x @ y.T
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
        
        r = dist / self.length_scale
        
        if abs(self.nu - 0.5) < 1e-6:
            # ν = 1/2: Exponential
            K = torch.exp(-r)
        elif abs(self.nu - 1.5) < 1e-6:
            # ν = 3/2
            sqrt3 = math.sqrt(3)
            K = (1 + sqrt3 * r) * torch.exp(-sqrt3 * r)
        elif abs(self.nu - 2.5) < 1e-6:
            # ν = 5/2
            sqrt5 = math.sqrt(5)
            K = (1 + sqrt5 * r + 5 * r**2 / 3) * torch.exp(-sqrt5 * r)
        else:
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {self.nu}")
        
        return self.variance * K
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.variance)


@dataclass
class PolynomialKernel(Kernel):
    """
    Polynomial kernel.
    
    k(x, y) = (σ² ⟨x, y⟩ + c)^d
    
    Attributes:
        degree: Polynomial degree d
        constant: Constant term c
        variance: Scaling factor σ²
    """
    degree: int = 3
    constant: float = 1.0
    variance: float = 1.0
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        inner = x @ y.T
        return (self.variance * inner + self.constant) ** self.degree
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        inner_diag = (x ** 2).sum(dim=-1)
        return (self.variance * inner_diag + self.constant) ** self.degree


@dataclass
class LinearKernel(Kernel):
    """
    Linear kernel (inner product).
    
    k(x, y) = σ² ⟨x, y⟩ + c
    
    Attributes:
        variance: Scaling factor σ²
        constant: Bias term c
    """
    variance: float = 1.0
    constant: float = 0.0
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        return self.variance * (x @ y.T) + self.constant
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.variance * (x ** 2).sum(dim=-1) + self.constant


@dataclass
class PeriodicKernel(Kernel):
    """
    Periodic kernel for cyclic patterns.
    
    k(x, y) = σ² exp(-2 sin²(π|x-y|/p) / ℓ²)
    
    Attributes:
        period: Period p
        length_scale: Length scale ℓ
        variance: Output variance σ²
    """
    period: float = 1.0
    length_scale: float = 1.0
    variance: float = 1.0
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        # Compute |x - y| for 1D
        if x.shape[-1] == 1 and y.shape[-1] == 1:
            diff = x - y.T
        else:
            # For multi-D, use Euclidean distance
            x_sq = (x ** 2).sum(dim=-1, keepdim=True)
            y_sq = (y ** 2).sum(dim=-1, keepdim=True)
            dist_sq = x_sq + y_sq.T - 2 * x @ y.T
            diff = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        sin_term = torch.sin(math.pi * torch.abs(diff) / self.period)
        
        return self.variance * torch.exp(-2 * sin_term ** 2 / self.length_scale ** 2)
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.variance)


@dataclass
class ScaledKernel(Kernel):
    """Kernel multiplied by a scalar."""
    base_kernel: Kernel
    scale: float = 1.0
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.scale * self.base_kernel(x, y)
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.base_kernel.diagonal(x)


@dataclass
class CompositeKernel(Kernel):
    """Base class for composite kernels."""
    kernel1: Kernel
    kernel2: Kernel


@dataclass
class SumKernel(CompositeKernel):
    """
    Sum of two kernels.
    
    k(x, y) = k1(x, y) + k2(x, y)
    """
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel1(x, y) + self.kernel2(x, y)
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel1.diagonal(x) + self.kernel2.diagonal(x)


@dataclass
class ProductKernel(CompositeKernel):
    """
    Product of two kernels.
    
    k(x, y) = k1(x, y) × k2(x, y)
    """
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel1(x, y) * self.kernel2(x, y)
    
    def diagonal(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel1.diagonal(x) * self.kernel2.diagonal(x)


def verify_kernel_properties(kernel: Kernel, 
                             points: torch.Tensor,
                             tol: float = 1e-4) -> Tuple[bool, str]:
    """
    Verify that a kernel satisfies key properties.
    
    Checks:
    1. Symmetry: k(x, y) = k(y, x)
    2. Positive semi-definiteness: eigenvalues ≥ 0
    
    Args:
        kernel: Kernel to verify
        points: Test points
        tol: Tolerance (default 1e-4 for numerical stability)
        
    Returns:
        (passed, message) tuple
    """
    K = kernel.matrix(points)
    n = K.shape[0]
    
    issues = []
    
    # Symmetry
    sym_err = (K - K.T).abs().max().item()
    if sym_err > tol:
        issues.append(f"Not symmetric: max error {sym_err:.2e}")
    
    # Positive semi-definiteness (allow small negative due to numerics)
    eigenvalues = torch.linalg.eigvalsh(K)
    min_eig = eigenvalues.min().item()
    if min_eig < -tol:
        issues.append(f"Not PSD: min eigenvalue {min_eig:.2e}")
    
    if issues:
        return False, "; ".join(issues)
    return True, f"Symmetric (err={sym_err:.2e}), PSD (λ_min={min_eig:.2e})"
