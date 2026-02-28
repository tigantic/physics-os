"""
Chebyshev Polynomial Approximation — Layer 21 Component

Fast approximation of matrix functions g(L) using Chebyshev polynomials.

Key insight: Instead of computing eigenvectors of L (O(N³)), we approximate
g(x) with Chebyshev polynomials and use the recurrence relation:
    T₀(x) = 1
    T₁(x) = x
    T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)

For QTT matrices, each T_k(L) application is O(r³ log N).

References:
    - Hammond et al. (2011) "Wavelets on Graphs"
    - Defferrard et al. (2016) "ChebNet"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import torch
import math

from .graph_signals import QTTSignal
from .laplacian import QTTLaplacian


def chebyshev_coefficients(
    func: Callable[[float], float],
    order: int,
    num_samples: int = 1000
) -> torch.Tensor:
    """
    Compute Chebyshev coefficients for a function on [-1, 1].
    
    The function is approximated as:
        g(x) ≈ Σ_{k=0}^{K} c_k T_k(x)
    
    where c_k are computed via discrete cosine transform.
    
    Args:
        func: Function to approximate, defined on [-1, 1]
        order: Number of Chebyshev terms (K+1)
        num_samples: Number of sample points for DCT
        
    Returns:
        Tensor of Chebyshev coefficients [c_0, c_1, ..., c_K]
    """
    # Chebyshev nodes
    k = torch.arange(num_samples, dtype=torch.float64)
    x = torch.cos(math.pi * (k + 0.5) / num_samples)
    
    # Sample function at nodes
    f_vals = torch.tensor([func(xi.item()) for xi in x], dtype=torch.float64)
    
    # Compute coefficients via DCT
    coeffs = torch.zeros(order, dtype=torch.float64)
    
    for n in range(order):
        # c_n = (2/M) * Σ f(x_k) T_n(x_k)
        Tn_vals = torch.cos(n * math.pi * (k + 0.5) / num_samples)
        coeffs[n] = (2.0 / num_samples) * (f_vals * Tn_vals).sum()
    
    # First coefficient has factor 1/2
    coeffs[0] = coeffs[0] / 2.0
    
    return coeffs


def chebyshev_approximation(
    coeffs: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate Chebyshev approximation at points x.
    
    Uses Clenshaw recurrence for numerical stability.
    
    Args:
        coeffs: Chebyshev coefficients [c_0, ..., c_K]
        x: Points in [-1, 1] to evaluate at
        
    Returns:
        Approximation values at x
    """
    K = len(coeffs)
    
    if K == 1:
        return coeffs[0] * torch.ones_like(x)
    
    # Clenshaw recurrence (backwards)
    b_kp2 = torch.zeros_like(x)
    b_kp1 = torch.zeros_like(x)
    
    for k in range(K - 1, 0, -1):
        b_k = 2 * x * b_kp1 - b_kp2 + coeffs[k]
        b_kp2 = b_kp1
        b_kp1 = b_k
    
    # Final step
    return x * b_kp1 - b_kp2 + coeffs[0]


@dataclass
class ChebyshevApproximator:
    """
    Chebyshev polynomial approximator for matrix functions.
    
    Given a function g and a normalized Laplacian L̃ (spectrum in [-1, 1]),
    computes g(L̃)x for signal x using Chebyshev recurrence.
    
    Attributes:
        coefficients: Chebyshev coefficients
        order: Number of polynomial terms
        laplacian: Normalized QTT Laplacian
    """
    coefficients: torch.Tensor
    order: int
    laplacian: QTTLaplacian
    
    @classmethod
    def from_function(
        cls,
        func: Callable[[float], float],
        laplacian: QTTLaplacian,
        order: int = 30,
        num_samples: int = 1000
    ) -> 'ChebyshevApproximator':
        """
        Create approximator from a function.
        
        Args:
            func: Function g(λ) to approximate, defined for λ ∈ [0, λ_max]
            laplacian: QTT Laplacian (will be normalized internally)
            order: Chebyshev polynomial order
            num_samples: Samples for coefficient computation
        """
        # Transform function to normalized domain [-1, 1]
        # x = 2λ/λ_max - 1, so λ = λ_max * (x + 1) / 2
        lambda_max = laplacian.max_eigenvalue
        
        def normalized_func(x: float) -> float:
            lam = lambda_max * (x + 1) / 2
            return func(lam)
        
        coeffs = chebyshev_coefficients(normalized_func, order, num_samples)
        
        return cls(
            coefficients=coeffs,
            order=order,
            laplacian=laplacian
        )
    
    def apply(self, signal: QTTSignal, round_tol: float = 1e-10) -> QTTSignal:
        """
        Apply g(L) to a signal using Chebyshev recurrence.
        
        Computes: y = g(L)x = Σ_k c_k T_k(L̃)x
        
        where L̃ = 2L/λ_max - I is the normalized Laplacian.
        
        Args:
            signal: Input signal x
            round_tol: TT-rounding tolerance after each iteration
            
        Returns:
            Output signal g(L)x
        """
        if len(self.coefficients) == 0:
            return QTTSignal.zeros(signal.num_nodes, dtype=signal.dtype)
        
        # Compute normalized Laplacian application: L̃x = 2Lx/λ_max - x
        def apply_normalized_laplacian(x: QTTSignal) -> QTTSignal:
            Lx = self.laplacian.matvec(x)
            scaled = Lx.scale(2.0 / self.laplacian.max_eigenvalue)
            result = scaled.add(x.scale(-1.0))
            return result.round(tol=round_tol)
        
        # Initialize Chebyshev recurrence
        # T_0(L̃)x = x
        T_0 = signal
        
        if self.order == 1:
            return T_0.scale(self.coefficients[0].item())
        
        # T_1(L̃)x = L̃x
        T_1 = apply_normalized_laplacian(signal)
        
        # Accumulate: result = c_0 * T_0 + c_1 * T_1
        result = T_0.scale(self.coefficients[0].item())
        result = result.add(T_1.scale(self.coefficients[1].item()))
        result = result.round(tol=round_tol)
        
        # Chebyshev recurrence: T_{k+1} = 2 L̃ T_k - T_{k-1}
        T_km1 = T_0  # T_{k-1}
        T_k = T_1    # T_k
        
        for k in range(2, self.order):
            # T_{k+1} = 2 L̃ T_k - T_{k-1}
            LT_k = apply_normalized_laplacian(T_k)
            T_kp1 = LT_k.scale(2.0).add(T_km1.scale(-1.0))
            T_kp1 = T_kp1.round(tol=round_tol)
            
            # Accumulate
            result = result.add(T_kp1.scale(self.coefficients[k].item()))
            result = result.round(tol=round_tol)
            
            # Update for next iteration
            T_km1 = T_k
            T_k = T_kp1
        
        return result
    
    def approximation_error(
        self,
        func: Callable[[float], float],
        num_test_points: int = 100
    ) -> Tuple[float, float]:
        """
        Estimate approximation error.
        
        Args:
            func: Original function
            num_test_points: Number of test points
            
        Returns:
            (max_error, mean_error)
        """
        lambda_max = self.laplacian.max_eigenvalue
        
        # Test points in [0, λ_max]
        lam = torch.linspace(0, lambda_max, num_test_points, dtype=torch.float64)
        x = 2 * lam / lambda_max - 1  # Normalized to [-1, 1]
        
        # True values
        true_vals = torch.tensor([func(l.item()) for l in lam], dtype=torch.float64)
        
        # Chebyshev approximation
        approx_vals = chebyshev_approximation(self.coefficients, x)
        
        errors = torch.abs(true_vals - approx_vals)
        
        return errors.max().item(), errors.mean().item()


# Convenience functions for common kernels

def mexican_hat_chebyshev(
    laplacian: QTTLaplacian,
    scale: float = 1.0,
    order: int = 30
) -> ChebyshevApproximator:
    """
    Create Chebyshev approximator for Mexican hat wavelet.
    
    g(λ) = λ * exp(-λ/scale)
    """
    def mexican_hat(lam: float) -> float:
        return lam * math.exp(-lam / scale) if lam >= 0 else 0.0
    
    return ChebyshevApproximator.from_function(mexican_hat, laplacian, order)


def heat_kernel_chebyshev(
    laplacian: QTTLaplacian,
    scale: float = 1.0,
    order: int = 30
) -> ChebyshevApproximator:
    """
    Create Chebyshev approximator for heat kernel.
    
    g(λ) = exp(-scale * λ)
    """
    def heat_kernel(lam: float) -> float:
        return math.exp(-scale * lam)
    
    return ChebyshevApproximator.from_function(heat_kernel, laplacian, order)


def lowpass_chebyshev(
    laplacian: QTTLaplacian,
    cutoff: float = 0.5,
    order: int = 30
) -> ChebyshevApproximator:
    """
    Create Chebyshev approximator for low-pass filter.
    
    g(λ) = exp(-λ²/(2*cutoff²))
    """
    lambda_max = laplacian.max_eigenvalue
    sigma = cutoff * lambda_max
    
    def lowpass(lam: float) -> float:
        return math.exp(-lam**2 / (2 * sigma**2))
    
    return ChebyshevApproximator.from_function(lowpass, laplacian, order)


def highpass_chebyshev(
    laplacian: QTTLaplacian,
    cutoff: float = 0.5,
    order: int = 30
) -> ChebyshevApproximator:
    """
    Create Chebyshev approximator for high-pass filter.
    
    g(λ) = 1 - exp(-λ²/(2*cutoff²))
    """
    lambda_max = laplacian.max_eigenvalue
    sigma = cutoff * lambda_max
    
    def highpass(lam: float) -> float:
        return 1.0 - math.exp(-lam**2 / (2 * sigma**2))
    
    return ChebyshevApproximator.from_function(highpass, laplacian, order)
