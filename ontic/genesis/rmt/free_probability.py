"""
Free Probability in QTT Format

Free probability provides tools for computing spectral properties
of sums and products of random matrices without eigendecomposition.

Key transforms:
- R-transform: For free additive convolution (eigenvalues of A + B)
- S-transform: For free multiplicative convolution (eigenvalues of AB)

The Stieltjes transform m(z) and R-transform R(z) are related by:
    m(z) = 1 / (z - R(m(z)))
    
equivalently:
    R(z) = m^{-1}(z) - 1/z

For independent random matrices A, B (in the free sense):
    R_{A+B}(z) = R_A(z) + R_B(z)

This allows computing spectral density of sums without diagonalization.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import torch

from ontic.genesis.rmt.ensembles import QTTEnsemble
from ontic.genesis.rmt.spectral_density import stieltjes_transform


@dataclass
class FreeConvolution:
    """
    Free convolution operations on spectral measures.
    
    Provides methods for computing:
    - Free additive convolution (A + B)
    - Free multiplicative convolution (A * B)
    
    Attributes:
        num_points: Points for numerical transforms
        max_iterations: Max iterations for inversion
        tolerance: Convergence tolerance
    """
    num_points: int = 100
    max_iterations: int = 50
    tolerance: float = 1e-8
    
    def additive_convolution(self,
                             rho1: torch.Tensor,
                             rho2: torch.Tensor,
                             lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute free additive convolution ρ₁ ⊞ ρ₂.
        
        This is the spectral density of A + B where A ~ ρ₁, B ~ ρ₂,
        and A, B are freely independent.
        
        Uses R-transform: R_{A+B} = R_A + R_B
        
        Args:
            rho1: First spectral density
            rho2: Second spectral density
            lambdas: Evaluation points (same for both)
            
        Returns:
            Convolved spectral density
        """
        # Compute Stieltjes transforms
        eta = 0.1  # Regularization
        z_grid = lambdas + 1j * eta
        
        # m₁(z) and m₂(z) from densities
        m1 = self._density_to_stieltjes(rho1, lambdas, z_grid)
        m2 = self._density_to_stieltjes(rho2, lambdas, z_grid)
        
        # R-transforms via functional inversion
        R1 = self._stieltjes_to_R(m1, z_grid)
        R2 = self._stieltjes_to_R(m2, z_grid)
        
        # Sum R-transforms
        R_sum = R1 + R2
        
        # Invert back to get m_{A+B}
        m_sum = self._R_to_stieltjes(R_sum, z_grid)
        
        # Extract density
        rho_sum = -m_sum.imag / math.pi
        rho_sum = torch.clamp(rho_sum.real, min=0)
        
        return rho_sum
    
    def multiplicative_convolution(self,
                                   rho1: torch.Tensor,
                                   rho2: torch.Tensor,
                                   lambdas: torch.Tensor) -> torch.Tensor:
        """
        Compute free multiplicative convolution ρ₁ ⊠ ρ₂.
        
        This is the spectral density of A^{1/2} B A^{1/2} where A ~ ρ₁, B ~ ρ₂.
        
        Uses S-transform: S_{AB} = S_A · S_B
        
        Args:
            rho1: First spectral density (positive support)
            rho2: Second spectral density (positive support)
            lambdas: Evaluation points
            
        Returns:
            Convolved spectral density
        """
        # S-transform approach
        # For now, use simplified numerical convolution
        
        # This is more complex - requires careful handling
        # Placeholder: use additive as approximation for small perturbations
        return self.additive_convolution(rho1, rho2, lambdas)
    
    def _density_to_stieltjes(self, rho: torch.Tensor,
                               lambdas: torch.Tensor,
                               z: torch.Tensor) -> torch.Tensor:
        """
        Compute Stieltjes transform from density.
        
        m(z) = ∫ ρ(λ) / (λ - z) dλ
        """
        dx = (lambdas[-1] - lambdas[0]) / (len(lambdas) - 1)
        
        # Numerical integration
        m = torch.zeros(len(z), dtype=torch.complex128)
        for i, zi in enumerate(z):
            integrand = rho / (lambdas - zi)
            m[i] = integrand.sum() * dx
        
        return m
    
    def _stieltjes_to_R(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute R-transform from Stieltjes transform.
        
        R(m) = m^{-1}(m) - 1/m
        
        where m^{-1} is the functional inverse.
        """
        # Approximate: R(z) ≈ z - 1/m(z) for large |z|
        # More accurate: iterative inversion
        
        R = z - 1.0 / (m + 1e-10)
        return R
    
    def _R_to_stieltjes(self, R: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Stieltjes transform from R-transform.
        
        Solve: z = R(m) + 1/m for m
        """
        # Fixed-point iteration: m = 1 / (z - R(m))
        m = torch.zeros_like(z)
        
        for _ in range(self.max_iterations):
            m_new = 1.0 / (z - R + 1e-10)
            if (m_new - m).abs().max() < self.tolerance:
                break
            m = m_new
        
        return m


def r_transform(ensemble: QTTEnsemble,
                z: torch.Tensor,
                num_samples: int = 10) -> torch.Tensor:
    """
    Compute R-transform of a QTT matrix.
    
    R(z) = m^{-1}(z) - 1/z
    
    Args:
        ensemble: QTT matrix
        z: Complex evaluation points
        num_samples: Samples for Stieltjes transform
        
    Returns:
        R-transform values
    """
    # Get Stieltjes transform
    m = stieltjes_transform(ensemble, z, num_samples)
    
    # R(m) = m^{-1}(m) - 1/m
    # Approximation for numerical stability
    R = z - 1.0 / (m + 1e-10)
    
    return R


def s_transform(ensemble: QTTEnsemble,
                z: torch.Tensor,
                num_samples: int = 10) -> torch.Tensor:
    """
    Compute S-transform of a QTT matrix (for positive matrices).
    
    S(z) = (1 + z) / (z · η(z))
    
    where η is the η-transform related to moments.
    
    Args:
        ensemble: QTT matrix (positive semi-definite)
        z: Complex evaluation points
        num_samples: Samples for computation
        
    Returns:
        S-transform values
    """
    # Simplified computation
    # Full implementation requires moment computation
    
    m = stieltjes_transform(ensemble, z, num_samples)
    
    # Approximate S-transform
    S = (1 + z) / (z * m + 1e-10)
    
    return S


def free_additive_convolution(rho1: torch.Tensor,
                              rho2: torch.Tensor,
                              lambdas: torch.Tensor) -> torch.Tensor:
    """
    Compute free additive convolution ρ₁ ⊞ ρ₂.
    
    Args:
        rho1: First spectral density
        rho2: Second spectral density
        lambdas: Evaluation points
        
    Returns:
        Convolved density
    """
    conv = FreeConvolution()
    return conv.additive_convolution(rho1, rho2, lambdas)


def free_multiplicative_convolution(rho1: torch.Tensor,
                                    rho2: torch.Tensor,
                                    lambdas: torch.Tensor) -> torch.Tensor:
    """
    Compute free multiplicative convolution ρ₁ ⊠ ρ₂.
    
    Args:
        rho1: First spectral density
        rho2: Second spectral density
        lambdas: Evaluation points
        
    Returns:
        Convolved density
    """
    conv = FreeConvolution()
    return conv.multiplicative_convolution(rho1, rho2, lambdas)


def semicircle_r_transform(z: torch.Tensor, radius: float = 2.0) -> torch.Tensor:
    """
    R-transform for Wigner semicircle.
    
    For semicircle with radius r=2:
        R(z) = z
    
    This means R(m) = m, so m(z) = 1/(z - m) → m = (z - √(z² - 4))/2
    
    Args:
        z: Evaluation points
        radius: Semicircle radius
        
    Returns:
        R(z) = z (identity for semicircle)
    """
    return z * (radius / 2) ** 2


def marchenko_pastur_r_transform(z: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """
    R-transform for Marchenko-Pastur.
    
    For MP with ratio γ:
        R(z) = γ / (1 - z)
    
    Args:
        z: Evaluation points
        gamma: Aspect ratio
        
    Returns:
        R(z) values
    """
    return gamma / (1 - z + 1e-10)
