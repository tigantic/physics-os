"""
Graph Filters — Layer 21 Component

Low-pass, high-pass, and band-pass filters for graph signals.

Graph filtering in the spectral domain:
    y = h(L) x
    
where h is the filter response function and L is the graph Laplacian.

Key filters:
    - Low-pass: Smooth/denoise signals
    - High-pass: Edge detection, anomaly detection
    - Band-pass: Isolate specific frequency components
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import torch
import math

from .laplacian import QTTLaplacian
from .graph_signals import QTTSignal
from .chebyshev import ChebyshevApproximator


@dataclass
class GraphFilter(ABC):
    """
    Abstract base class for graph filters.
    
    A graph filter applies a function h(L) to a signal:
        y = h(L) x
    
    Implemented via Chebyshev polynomial approximation.
    """
    laplacian: QTTLaplacian
    chebyshev_order: int = 30
    approximator: Optional[ChebyshevApproximator] = None
    
    def __post_init__(self):
        """Build Chebyshev approximator."""
        if self.approximator is None:
            self._build_approximator()
    
    @abstractmethod
    def response(self, lam: float) -> float:
        """
        Filter frequency response h(λ).
        
        Args:
            lam: Eigenvalue (frequency)
            
        Returns:
            Filter response at this frequency
        """
        pass
    
    def _build_approximator(self):
        """Build Chebyshev approximator for filter response."""
        self.approximator = ChebyshevApproximator.from_function(
            self.response,
            self.laplacian,
            order=self.chebyshev_order
        )
    
    def apply(self, signal: QTTSignal, round_tol: float = 1e-10) -> QTTSignal:
        """
        Apply filter to signal.
        
        Args:
            signal: Input signal
            round_tol: TT-rounding tolerance
            
        Returns:
            Filtered signal
        """
        if self.approximator is None:
            self._build_approximator()
        
        return self.approximator.apply(signal, round_tol=round_tol)
    
    def frequency_response(self, num_points: int = 100) -> tuple:
        """
        Compute filter frequency response curve.
        
        Returns:
            (frequencies, responses) tuple
        """
        lam = torch.linspace(0, self.laplacian.max_eigenvalue, num_points)
        response = torch.tensor([self.response(l.item()) for l in lam])
        return lam, response


@dataclass
class LowPassFilter(GraphFilter):
    """
    Low-pass filter for graph signals.
    
    h(λ) = exp(-λ² / (2σ²))
    
    where σ = cutoff * λ_max controls the cutoff frequency.
    
    Applications:
        - Denoising
        - Smoothing
        - Community detection preprocessing
    """
    cutoff: float = 0.5  # Normalized cutoff in [0, 1]
    
    def response(self, lam: float) -> float:
        """Gaussian low-pass response."""
        sigma = self.cutoff * self.laplacian.max_eigenvalue
        if sigma <= 0:
            return 1.0 if lam == 0 else 0.0
        return math.exp(-lam**2 / (2 * sigma**2))


@dataclass
class HighPassFilter(GraphFilter):
    """
    High-pass filter for graph signals.
    
    h(λ) = 1 - exp(-λ² / (2σ²))
    
    Applications:
        - Edge detection
        - Anomaly detection
        - Feature extraction
    """
    cutoff: float = 0.5  # Normalized cutoff in [0, 1]
    
    def response(self, lam: float) -> float:
        """Gaussian high-pass response."""
        sigma = self.cutoff * self.laplacian.max_eigenvalue
        if sigma <= 0:
            return 0.0 if lam == 0 else 1.0
        return 1.0 - math.exp(-lam**2 / (2 * sigma**2))


@dataclass
class BandPassFilter(GraphFilter):
    """
    Band-pass filter for graph signals.
    
    h(λ) = exp(-((λ - μ) / σ)²)
    
    where μ is center frequency and σ controls bandwidth.
    
    Applications:
        - Isolating specific frequency components
        - Multi-scale analysis
        - Pattern matching
    """
    low: float = 0.3   # Lower cutoff (normalized)
    high: float = 0.7  # Upper cutoff (normalized)
    
    def response(self, lam: float) -> float:
        """Band-pass response (product of low and high pass)."""
        lambda_max = self.laplacian.max_eigenvalue
        low_sigma = self.low * lambda_max
        high_sigma = self.high * lambda_max
        
        # Low-pass at high cutoff
        if high_sigma <= 0:
            lp = 1.0 if lam == 0 else 0.0
        else:
            lp = math.exp(-lam**2 / (2 * high_sigma**2))
        
        # High-pass at low cutoff
        if low_sigma <= 0:
            hp = 0.0 if lam == 0 else 1.0
        else:
            hp = 1.0 - math.exp(-lam**2 / (2 * low_sigma**2))
        
        return lp * hp


@dataclass
class IdealLowPassFilter(GraphFilter):
    """
    Ideal low-pass filter (sharp cutoff).
    
    h(λ) = 1 if λ ≤ λ_c, else 0
    
    Note: Requires high Chebyshev order for good approximation.
    """
    cutoff: float = 0.5
    
    def response(self, lam: float) -> float:
        """Sharp cutoff response."""
        lambda_c = self.cutoff * self.laplacian.max_eigenvalue
        return 1.0 if lam <= lambda_c else 0.0


@dataclass
class HeatFilter(GraphFilter):
    """
    Heat diffusion filter.
    
    h(λ) = exp(-t * λ)
    
    Corresponds to heat equation solution at time t.
    """
    time: float = 1.0
    
    def response(self, lam: float) -> float:
        """Heat kernel response."""
        return math.exp(-self.time * lam)


@dataclass
class InverseLaplacianFilter(GraphFilter):
    """
    Inverse Laplacian filter (regularized).
    
    h(λ) = 1 / (λ + ε)
    
    Used for graph signal smoothing and solving Poisson equation.
    """
    regularization: float = 0.1
    
    def response(self, lam: float) -> float:
        """Regularized inverse response."""
        return 1.0 / (lam + self.regularization)


@dataclass  
class CustomFilter(GraphFilter):
    """
    Custom filter with user-defined response function.
    
    Example:
        >>> def my_response(lam):
        ...     return lam**2 * math.exp(-lam)
        >>> filt = CustomFilter(L, response_func=my_response)
    """
    response_func: Callable[[float], float] = None
    
    def response(self, lam: float) -> float:
        """User-defined response."""
        if self.response_func is None:
            return 1.0
        return self.response_func(lam)


# Convenience constructors

def low_pass(laplacian: QTTLaplacian, cutoff: float = 0.5, 
             order: int = 30) -> LowPassFilter:
    """Create low-pass filter."""
    return LowPassFilter(laplacian=laplacian, cutoff=cutoff, chebyshev_order=order)


def high_pass(laplacian: QTTLaplacian, cutoff: float = 0.5,
              order: int = 30) -> HighPassFilter:
    """Create high-pass filter."""
    return HighPassFilter(laplacian=laplacian, cutoff=cutoff, chebyshev_order=order)


def band_pass(laplacian: QTTLaplacian, low: float = 0.3, high: float = 0.7,
              order: int = 30) -> BandPassFilter:
    """Create band-pass filter."""
    return BandPassFilter(laplacian=laplacian, low=low, high=high, chebyshev_order=order)


def heat_diffusion(laplacian: QTTLaplacian, time: float = 1.0,
                   order: int = 30) -> HeatFilter:
    """Create heat diffusion filter."""
    return HeatFilter(laplacian=laplacian, time=time, chebyshev_order=order)
