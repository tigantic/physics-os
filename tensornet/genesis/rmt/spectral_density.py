"""
QTT Spectral Density Estimation

Computes the eigenvalue density ρ(λ) from a QTT matrix using the resolvent.

The key relation is:
    ρ(λ) = -(1/π) lim_{η→0+} Im[m(λ + iη)]

where m(z) = (1/N) Tr((H - zI)^{-1}) is the Stieltjes transform.

For finite η > 0, we get a smoothed version of the density with resolution ~ η.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch

from tensornet.genesis.rmt.ensembles import QTTEnsemble
from tensornet.genesis.rmt.resolvent import resolvent_trace, resolvent_at_points


@dataclass
class SpectralDensity:
    """
    Spectral density estimator for QTT matrices.
    
    Computes ρ(λ) using the Stieltjes transform.
    
    Attributes:
        ensemble: QTT matrix H
        lambda_min: Minimum eigenvalue (estimated or given)
        lambda_max: Maximum eigenvalue (estimated or given)
        eta: Broadening parameter (Im(z))
        num_samples: Samples for trace estimation
    """
    ensemble: QTTEnsemble
    lambda_min: float = -3.0
    lambda_max: float = 3.0
    eta: float = 0.05
    num_samples: int = 10
    
    @classmethod
    def from_ensemble(cls, ensemble: QTTEnsemble,
                      margin: float = 0.5,
                      eta: float = 0.05,
                      num_samples: int = 10) -> 'SpectralDensity':
        """
        Create spectral density estimator with auto-detected bounds.
        
        For Wigner matrices, eigenvalues lie in [-2-ε, 2+ε].
        For Wishart with ratio γ, in [(1-√γ)², (1+√γ)²].
        
        Args:
            ensemble: QTT matrix
            margin: Extra margin beyond theoretical bounds
            eta: Broadening parameter
            num_samples: Trace estimation samples
            
        Returns:
            SpectralDensity estimator
        """
        if ensemble.ensemble_type == 'wishart':
            gamma = getattr(ensemble, 'aspect_ratio', 0.5)
            lambda_min = max(0, (1 - math.sqrt(gamma))**2 - margin)
            lambda_max = (1 + math.sqrt(gamma))**2 + margin
        else:
            # Wigner/GOE/GUE: semicircle support is [-2, 2]
            lambda_min = -2.0 - margin
            lambda_max = 2.0 + margin
        
        return cls(
            ensemble=ensemble,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            eta=eta,
            num_samples=num_samples
        )
    
    def evaluate(self, lambdas: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spectral density at given points.
        
        ρ(λ) = (1/π) Im[m(λ + iη)]
        
        where m(z) = (1/N) Tr((H - zI)^{-1}).
        For our convention, Im(m) > 0 when Im(z) > 0.
        
        Args:
            lambdas: Points to evaluate density
            
        Returns:
            Density values
        """
        m_values = resolvent_at_points(
            self.ensemble, lambdas, 
            eta=self.eta, 
            num_samples=self.num_samples
        )
        
        # ρ(λ) = (1/π) Im[m(λ + iη)]
        # Our convention: m(z) = (1/N) Tr((H-zI)^{-1}) has Im(m) > 0
        rho = m_values.imag / math.pi
        
        # Ensure non-negative (can be slightly negative due to noise)
        rho = torch.clamp(rho, min=0)
        
        return rho.real
    
    def compute(self, num_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spectral density over the full range.
        
        Args:
            num_points: Number of evaluation points
            
        Returns:
            (lambdas, rho) tuple
        """
        lambdas = torch.linspace(self.lambda_min, self.lambda_max, num_points)
        rho = self.evaluate(lambdas)
        return lambdas, rho
    
    def moments(self, max_order: int = 4) -> List[float]:
        """
        Compute moments of the spectral density.
        
        μ_k = ∫ λ^k ρ(λ) dλ ≈ (1/N) Tr(H^k)
        
        For k=1: mean eigenvalue
        For k=2: related to Frobenius norm
        
        Args:
            max_order: Maximum moment order
            
        Returns:
            List of moments [μ_0, μ_1, ..., μ_{max_order}]
        """
        # Use dense for now
        if self.ensemble.size > 2**12:
            raise NotImplementedError("Large-scale moments pending")
        
        H = self.ensemble.to_dense()
        N = self.ensemble.size
        
        moments = []
        H_power = torch.eye(N, dtype=H.dtype)
        
        for k in range(max_order + 1):
            mu_k = torch.trace(H_power).real / N
            moments.append(mu_k.item())
            H_power = H_power @ H
        
        return moments


def spectral_density(ensemble: QTTEnsemble,
                     num_points: int = 100,
                     eta: float = 0.05,
                     num_samples: int = 10,
                     lambda_range: Optional[Tuple[float, float]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral density of a QTT matrix.
    
    Args:
        ensemble: QTT matrix H
        num_points: Number of evaluation points
        eta: Broadening parameter
        num_samples: Trace estimation samples
        lambda_range: Optional (min, max) range
        
    Returns:
        (lambdas, rho) tuple
    """
    if lambda_range is not None:
        estimator = SpectralDensity(
            ensemble=ensemble,
            lambda_min=lambda_range[0],
            lambda_max=lambda_range[1],
            eta=eta,
            num_samples=num_samples
        )
    else:
        estimator = SpectralDensity.from_ensemble(
            ensemble, eta=eta, num_samples=num_samples
        )
    
    return estimator.compute(num_points)


def stieltjes_transform(ensemble: QTTEnsemble,
                        z: Union[complex, torch.Tensor],
                        num_samples: int = 10) -> Union[complex, torch.Tensor]:
    """
    Compute Stieltjes transform m(z) = (1/N) Tr((H - zI)^{-1}).
    
    Args:
        ensemble: QTT matrix H
        z: Complex spectral parameter(s)
        num_samples: Trace estimation samples
        
    Returns:
        m(z) value(s)
    """
    if isinstance(z, complex):
        return resolvent_trace(ensemble, z, num_samples)
    
    # Array of z values
    m_values = torch.zeros(len(z), dtype=torch.complex128)
    for i, zi in enumerate(z):
        m_values[i] = resolvent_trace(ensemble, complex(zi), num_samples)
    return m_values


def inverse_stieltjes(m: torch.Tensor, 
                      lambdas: torch.Tensor) -> torch.Tensor:
    """
    Recover spectral density from Stieltjes transform values.
    
    ρ(λ) = (1/π) Im[m(λ + i0+)]
    
    (Our convention has Im(m) > 0 for Im(z) > 0.)
    
    Args:
        m: Complex Stieltjes transform values
        lambdas: Corresponding real points
        
    Returns:
        Spectral density values
    """
    rho = m.imag / math.pi
    return torch.clamp(rho, min=0).real


def density_from_eigenvalues(eigenvalues: torch.Tensor,
                             num_points: int = 100,
                             bandwidth: Optional[float] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute density from actual eigenvalues using KDE.
    
    Useful for comparison/validation against QTT estimate.
    
    Args:
        eigenvalues: Array of eigenvalues
        num_points: Number of density points
        bandwidth: KDE bandwidth (auto if None)
        
    Returns:
        (lambdas, rho) tuple
    """
    n = len(eigenvalues)
    
    if bandwidth is None:
        # Silverman's rule
        std = eigenvalues.std()
        bandwidth = 1.06 * std * n ** (-1/5)
    
    lambda_min = eigenvalues.min() - 3 * bandwidth
    lambda_max = eigenvalues.max() + 3 * bandwidth
    
    lambdas = torch.linspace(lambda_min.item(), lambda_max.item(), num_points)
    
    # Gaussian KDE
    rho = torch.zeros(num_points)
    for lam in eigenvalues:
        rho += torch.exp(-0.5 * ((lambdas - lam) / bandwidth) ** 2)
    
    rho = rho / (n * bandwidth * math.sqrt(2 * math.pi))
    
    return lambdas, rho
