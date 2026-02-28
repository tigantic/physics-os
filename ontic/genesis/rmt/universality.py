"""
Universality Laws in Random Matrix Theory

Provides theoretical spectral densities for comparison:
- Wigner Semicircle Law (GOE/GUE)
- Marchenko-Pastur Law (Wishart)

Also includes verification tools to test if empirical densities
match theoretical predictions.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch

from ontic.genesis.rmt.ensembles import QTTEnsemble
from ontic.genesis.rmt.spectral_density import spectral_density, density_from_eigenvalues


@dataclass
class WignerSemicircle:
    """
    Wigner semicircle law for GOE/GUE matrices.
    
    For N × N random symmetric matrix with i.i.d. entries (variance 1/N),
    the eigenvalue density converges to:
    
        ρ(λ) = (1/2π) √(4 - λ²)  for |λ| ≤ 2
             = 0                  for |λ| > 2
    
    Attributes:
        radius: Semicircle radius (default 2 for normalized matrices)
    """
    radius: float = 2.0
    
    def __call__(self, lambdas: torch.Tensor) -> torch.Tensor:
        """Evaluate semicircle density."""
        return self.evaluate(lambdas)
    
    def evaluate(self, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Wigner semicircle at given points.
        
        Args:
            lambdas: Evaluation points
            
        Returns:
            Density values
        """
        r = self.radius
        
        # ρ(λ) = (1/2πr²) √(4r² - λ²) for standard normalization
        # With r=2: ρ(λ) = (1/2π) √(4 - λ²)
        
        rho = torch.zeros_like(lambdas)
        mask = (lambdas.abs() <= r)
        
        rho[mask] = torch.sqrt(r**2 - lambdas[mask]**2) / (math.pi * r**2 / 2)
        
        return rho
    
    def cdf(self, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Cumulative distribution function.
        
        F(λ) = ∫_{-r}^λ ρ(x) dx
        
        For semicircle with radius r:
        F(λ) = 1/2 + λ√(r² - λ²)/(πr²) + arcsin(λ/r)/π
        
        Args:
            lambdas: Evaluation points
            
        Returns:
            CDF values
        """
        r = self.radius
        cdf = torch.zeros_like(lambdas)
        
        # Below support
        cdf[lambdas < -r] = 0.0
        
        # Above support
        cdf[lambdas > r] = 1.0
        
        # Inside support
        mask = (lambdas >= -r) & (lambdas <= r)
        x = lambdas[mask]
        cdf[mask] = 0.5 + x * torch.sqrt(r**2 - x**2) / (math.pi * r**2) + torch.arcsin(x / r) / math.pi
        
        return cdf
    
    def moments(self, max_order: int = 10) -> torch.Tensor:
        """
        Compute moments of the semicircle distribution.
        
        For the semicircle:
        - Odd moments = 0
        - Even moments: μ_{2n} = C_n (Catalan numbers) for r=2
        
        Args:
            max_order: Maximum moment order
            
        Returns:
            Array of moments
        """
        moments = torch.zeros(max_order + 1)
        
        for n in range(max_order + 1):
            if n % 2 == 1:
                moments[n] = 0
            else:
                k = n // 2
                # Catalan number C_k = (2k)! / ((k+1)! k!)
                catalan = math.factorial(2*k) / (math.factorial(k+1) * math.factorial(k))
                moments[n] = catalan * (self.radius / 2) ** n
        
        return moments


@dataclass
class MarchenkoPastur:
    """
    Marchenko-Pastur law for Wishart matrices.
    
    For sample covariance W = (1/n) X^T X where X is n × p,
    with aspect ratio γ = p/n < 1, the eigenvalue density is:
    
        ρ(λ) = (1/2πγλ) √((λ_+ - λ)(λ - λ_-))  for λ_- ≤ λ ≤ λ_+
        
    where λ_± = (1 ± √γ)².
    
    Attributes:
        gamma: Aspect ratio p/n (must be < 1)
    """
    gamma: float = 0.5
    
    def __post_init__(self):
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        
        self.lambda_minus = (1 - math.sqrt(self.gamma)) ** 2
        self.lambda_plus = (1 + math.sqrt(self.gamma)) ** 2
    
    def __call__(self, lambdas: torch.Tensor) -> torch.Tensor:
        """Evaluate MP density."""
        return self.evaluate(lambdas)
    
    def evaluate(self, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Marchenko-Pastur density at given points.
        
        Args:
            lambdas: Evaluation points
            
        Returns:
            Density values
        """
        rho = torch.zeros_like(lambdas)
        
        mask = (lambdas >= self.lambda_minus) & (lambdas <= self.lambda_plus)
        
        lam = lambdas[mask]
        numerator = torch.sqrt((self.lambda_plus - lam) * (lam - self.lambda_minus))
        denominator = 2 * math.pi * self.gamma * lam
        
        rho[mask] = numerator / denominator
        
        return rho
    
    def support(self) -> Tuple[float, float]:
        """Return eigenvalue support [λ_-, λ_+]."""
        return (self.lambda_minus, self.lambda_plus)
    
    def mean(self) -> float:
        """Mean of MP distribution = 1."""
        return 1.0
    
    def variance(self) -> float:
        """Variance of MP distribution = γ."""
        return self.gamma


def wigner_semicircle(lambdas: torch.Tensor,
                      radius: float = 2.0) -> torch.Tensor:
    """
    Evaluate Wigner semicircle density.
    
    Args:
        lambdas: Evaluation points
        radius: Semicircle radius (default 2)
        
    Returns:
        Density values
    """
    return WignerSemicircle(radius=radius).evaluate(lambdas)


def marchenko_pastur(lambdas: torch.Tensor,
                     gamma: float = 0.5) -> torch.Tensor:
    """
    Evaluate Marchenko-Pastur density.
    
    Args:
        lambdas: Evaluation points
        gamma: Aspect ratio (0 < γ < 1)
        
    Returns:
        Density values
    """
    return MarchenkoPastur(gamma=gamma).evaluate(lambdas)


@dataclass
class UniversalityResult:
    """Result of universality verification."""
    law: str  # 'wigner' or 'marchenko_pastur'
    ks_statistic: float
    l2_error: float
    linf_error: float
    passed: bool
    threshold: float = 0.1


def verify_universality(ensemble: QTTEnsemble,
                        law: str = 'wigner',
                        num_points: int = 100,
                        eta: float = 0.05,
                        threshold: float = 0.2) -> UniversalityResult:
    """
    Verify if matrix follows expected universality law.
    
    Compares empirical spectral density against theoretical prediction.
    
    Args:
        ensemble: QTT matrix to test
        law: 'wigner' or 'marchenko_pastur'
        num_points: Points for density estimation
        eta: Broadening parameter
        threshold: L2 error threshold for passing
        
    Returns:
        UniversalityResult with statistics
    """
    # Compute empirical density
    lambdas, rho_empirical = spectral_density(
        ensemble, num_points=num_points, eta=eta
    )
    
    # Get theoretical density
    if law == 'wigner':
        rho_theory = wigner_semicircle(lambdas)
    elif law == 'marchenko_pastur':
        gamma = getattr(ensemble, 'aspect_ratio', 0.5)
        rho_theory = marchenko_pastur(lambdas, gamma=gamma)
    else:
        raise ValueError(f"Unknown law: {law}")
    
    # Normalize both to integrate to 1
    dx = (lambdas[-1] - lambdas[0]) / (len(lambdas) - 1)
    rho_empirical = rho_empirical / (rho_empirical.sum() * dx + 1e-10)
    rho_theory = rho_theory / (rho_theory.sum() * dx + 1e-10)
    
    # Compute errors
    l2_error = torch.sqrt(((rho_empirical - rho_theory) ** 2).sum() * dx).item()
    linf_error = (rho_empirical - rho_theory).abs().max().item()
    
    # KS statistic (max difference in CDFs)
    cdf_empirical = torch.cumsum(rho_empirical * dx, dim=0)
    cdf_theory = torch.cumsum(rho_theory * dx, dim=0)
    ks_statistic = (cdf_empirical - cdf_theory).abs().max().item()
    
    passed = l2_error < threshold
    
    return UniversalityResult(
        law=law,
        ks_statistic=ks_statistic,
        l2_error=l2_error,
        linf_error=linf_error,
        passed=passed,
        threshold=threshold
    )


def verify_with_dense(ensemble: QTTEnsemble,
                      law: str = 'wigner') -> UniversalityResult:
    """
    Verify universality using dense eigendecomposition.
    
    Only for small matrices (validation purposes).
    
    Args:
        ensemble: QTT matrix (size ≤ 2^12)
        law: 'wigner' or 'marchenko_pastur'
        
    Returns:
        UniversalityResult
    """
    if ensemble.size > 2**12:
        raise ValueError(f"Matrix too large for dense: {ensemble.size}")
    
    # Get actual eigenvalues
    H = ensemble.to_dense()
    eigenvalues = torch.linalg.eigvalsh(H)
    
    # Normalize by sqrt(N) for Wigner normalization
    eigenvalues = eigenvalues / math.sqrt(ensemble.size)
    
    # KDE density
    lambdas, rho_empirical = density_from_eigenvalues(eigenvalues, num_points=100)
    
    # Theoretical
    if law == 'wigner':
        rho_theory = wigner_semicircle(lambdas)
    elif law == 'marchenko_pastur':
        gamma = getattr(ensemble, 'aspect_ratio', 0.5)
        rho_theory = marchenko_pastur(lambdas, gamma=gamma)
    else:
        raise ValueError(f"Unknown law: {law}")
    
    # Normalize
    dx = (lambdas[-1] - lambdas[0]) / (len(lambdas) - 1)
    rho_empirical = rho_empirical / (rho_empirical.sum() * dx + 1e-10)
    rho_theory = rho_theory / (rho_theory.sum() * dx + 1e-10)
    
    # Errors
    l2_error = torch.sqrt(((rho_empirical - rho_theory) ** 2).sum() * dx).item()
    linf_error = (rho_empirical - rho_theory).abs().max().item()
    
    # KS
    cdf_empirical = torch.cumsum(rho_empirical * dx, dim=0)
    cdf_theory = torch.cumsum(rho_theory * dx, dim=0)
    ks_statistic = (cdf_empirical - cdf_theory).abs().max().item()
    
    # For random matrices, convergence to limiting law requires large N
    # Use generous threshold
    passed = l2_error < 0.5
    
    return UniversalityResult(
        law=law,
        ks_statistic=ks_statistic,
        l2_error=l2_error,
        linf_error=linf_error,
        passed=passed,
        threshold=0.5
    )
