"""
QTT Spectral Graph Wavelets — Layer 21 Component

Multi-scale wavelet transforms on graphs using QTT compression.

The spectral graph wavelet at scale s centered at node n is:
    ψ_{s,n} = g(sL)δ_n
    
where g is the wavelet generating kernel and L is the graph Laplacian.

For a signal f, the wavelet coefficients are:
    W_f(s, n) = ⟨ψ_{s,n}, f⟩ = (g(sL)f)(n)

Key wavelets implemented:
    - Mexican hat: g(λ) = λ exp(-λ) — edge detection
    - Heat kernel: g(λ) = exp(-sλ) — diffusion/smoothing
    - Meyer: Compactly supported in frequency
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Literal
import torch
import math

from .laplacian import QTTLaplacian
from .graph_signals import QTTSignal
from .chebyshev import ChebyshevApproximator, chebyshev_coefficients


# Wavelet kernel functions

def mexican_hat_kernel(lam: float, scale: float = 1.0) -> float:
    """
    Mexican hat (Difference of Gaussians) wavelet.
    
    g(λ) = sλ * exp(-sλ)
    
    Good for edge detection and localization.
    """
    x = scale * lam
    return x * math.exp(-x) if x >= 0 else 0.0


def heat_kernel(lam: float, scale: float = 1.0) -> float:
    """
    Heat/diffusion kernel.
    
    g(λ) = exp(-s*λ)
    
    Smoothing/diffusion at scale s.
    """
    return math.exp(-scale * lam)


def meyer_kernel(lam: float, scale: float = 1.0, 
                 lambda_max: float = 4.0) -> float:
    """
    Meyer wavelet kernel (compactly supported).
    
    Based on the Meyer wavelet with smooth transitions.
    """
    # Normalized frequency
    x = lam / lambda_max * scale
    
    if x < 1.0/3:
        return 0.0
    elif x < 2.0/3:
        # Rising edge
        t = 3 * x - 1
        # Smooth step function
        return math.sin(0.5 * math.pi * _meyer_aux(t))
    elif x < 4.0/3:
        # Falling edge
        t = 1.5 * x - 1
        return math.cos(0.5 * math.pi * _meyer_aux(t))
    else:
        return 0.0


def _meyer_aux(t: float) -> float:
    """Auxiliary function for Meyer wavelet."""
    if t <= 0:
        return 0.0
    elif t >= 1:
        return 1.0
    else:
        return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)


def abspline_kernel(lam: float, scale: float = 1.0, alpha: int = 2) -> float:
    """
    Abspline wavelet kernel.
    
    g(λ) = (sλ)^α for polynomial growth.
    """
    x = scale * lam
    if x <= 0:
        return 0.0
    return x ** alpha * math.exp(-x)


@dataclass
class WaveletResult:
    """
    Result of spectral graph wavelet transform.
    
    Attributes:
        coefficients: List of wavelet coefficients for each scale
        scales: Wavelet scales used
        scaling_coefficients: Optional low-pass (scaling function) output
    """
    coefficients: List[QTTSignal]
    scales: List[float]
    scaling_coefficients: Optional[QTTSignal] = None
    
    def energy_per_scale(self) -> List[float]:
        """Compute energy at each scale."""
        return [c.norm() ** 2 for c in self.coefficients]
    
    def total_energy(self) -> float:
        """Total energy across all scales."""
        energy = sum(self.energy_per_scale())
        if self.scaling_coefficients is not None:
            energy += self.scaling_coefficients.norm() ** 2
        return energy


@dataclass
class QTTGraphWavelet:
    """
    Spectral graph wavelet transform in QTT format.
    
    Computes multi-scale wavelet decomposition of signals on graphs
    using Chebyshev polynomial approximation.
    
    Attributes:
        laplacian: QTT graph Laplacian
        scales: List of wavelet scales
        kernel_type: Wavelet kernel ('mexican_hat', 'heat', 'meyer', 'abspline')
        chebyshev_order: Polynomial approximation order
        approximators: Pre-computed Chebyshev approximators for each scale
    """
    laplacian: QTTLaplacian
    scales: List[float]
    kernel_type: str
    chebyshev_order: int
    approximators: List[ChebyshevApproximator] = field(default_factory=list)
    scaling_approximator: Optional[ChebyshevApproximator] = None
    
    def __post_init__(self):
        """Pre-compute Chebyshev approximators."""
        if not self.approximators:
            self._build_approximators()
    
    @classmethod
    def create(
        cls,
        laplacian: QTTLaplacian,
        scales: Optional[List[float]] = None,
        kernel: Literal['mexican_hat', 'heat', 'meyer', 'abspline'] = 'mexican_hat',
        chebyshev_order: int = 30,
        include_scaling: bool = True
    ) -> 'QTTGraphWavelet':
        """
        Create wavelet transform.
        
        Args:
            laplacian: QTT graph Laplacian
            scales: Wavelet scales (default: [1, 2, 4, 8, 16])
            kernel: Wavelet kernel type
            chebyshev_order: Polynomial order for approximation
            include_scaling: Include low-pass scaling function
        """
        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0, 16.0]
        
        wavelet = cls(
            laplacian=laplacian,
            scales=list(scales),
            kernel_type=kernel,
            chebyshev_order=chebyshev_order
        )
        
        return wavelet
    
    def _build_approximators(self):
        """Build Chebyshev approximators for all scales."""
        lambda_max = self.laplacian.max_eigenvalue
        
        for scale in self.scales:
            # Get kernel function for this scale
            kernel_func = self._get_scaled_kernel(scale)
            
            # Build approximator
            approx = ChebyshevApproximator.from_function(
                kernel_func,
                self.laplacian,
                order=self.chebyshev_order
            )
            self.approximators.append(approx)
        
        # Build scaling function (low-pass)
        def scaling_func(lam: float) -> float:
            # Low-pass that captures what wavelets miss
            return math.exp(-lam / (self.scales[0] * 2))
        
        self.scaling_approximator = ChebyshevApproximator.from_function(
            scaling_func,
            self.laplacian,
            order=self.chebyshev_order
        )
    
    def _get_scaled_kernel(self, scale: float) -> Callable[[float], float]:
        """Get wavelet kernel function for a specific scale."""
        if self.kernel_type == 'mexican_hat':
            return lambda lam: mexican_hat_kernel(lam, scale)
        elif self.kernel_type == 'heat':
            return lambda lam: heat_kernel(lam, scale)
        elif self.kernel_type == 'meyer':
            return lambda lam: meyer_kernel(lam, scale, self.laplacian.max_eigenvalue)
        elif self.kernel_type == 'abspline':
            return lambda lam: abspline_kernel(lam, scale)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def transform(
        self,
        signal: QTTSignal,
        round_tol: float = 1e-10
    ) -> WaveletResult:
        """
        Apply wavelet transform to signal.
        
        Computes wavelet coefficients at each scale:
            W_f(s) = g_s(L) f
        
        Args:
            signal: Input signal on graph
            round_tol: TT-rounding tolerance
            
        Returns:
            WaveletResult with coefficients at each scale
        """
        if not self.approximators:
            self._build_approximators()
        
        coefficients = []
        
        for approx in self.approximators:
            # Apply wavelet at this scale
            coef = approx.apply(signal, round_tol=round_tol)
            coefficients.append(coef)
        
        # Scaling function (low-pass)
        scaling_coef = None
        if self.scaling_approximator is not None:
            scaling_coef = self.scaling_approximator.apply(signal, round_tol=round_tol)
        
        return WaveletResult(
            coefficients=coefficients,
            scales=self.scales.copy(),
            scaling_coefficients=scaling_coef
        )
    
    def inverse(
        self,
        wavelet_result: WaveletResult,
        round_tol: float = 1e-10
    ) -> QTTSignal:
        """
        Reconstruct signal from wavelet coefficients.
        
        Uses the frame property of spectral wavelets for reconstruction.
        For tight frames: f = Σ_s W_s^T W_s f
        
        Args:
            wavelet_result: Wavelet transform output
            round_tol: TT-rounding tolerance
            
        Returns:
            Reconstructed signal
        """
        if not self.approximators:
            self._build_approximators()
        
        # Initialize with zeros
        result = QTTSignal.zeros(self.laplacian.num_nodes, 
                                 dtype=wavelet_result.coefficients[0].dtype)
        
        # Accumulate wavelet reconstructions
        for approx, coef in zip(self.approximators, wavelet_result.coefficients):
            # Apply adjoint (same as forward for symmetric kernel)
            contribution = approx.apply(coef, round_tol=round_tol)
            result = result.add(contribution)
            result = result.round(tol=round_tol)
        
        # Add scaling function contribution
        if wavelet_result.scaling_coefficients is not None and \
           self.scaling_approximator is not None:
            scaling_contribution = self.scaling_approximator.apply(
                wavelet_result.scaling_coefficients, round_tol=round_tol
            )
            result = result.add(scaling_contribution)
            result = result.round(tol=round_tol)
        
        return result
    
    def localization_at_node(
        self,
        node_index: int,
        scale_index: int = 0,
        round_tol: float = 1e-10
    ) -> QTTSignal:
        """
        Compute wavelet localized at a specific node.
        
        ψ_{s,n} = g_s(L) δ_n
        
        Args:
            node_index: Node to center wavelet at
            scale_index: Which scale to use
            round_tol: TT-rounding tolerance
            
        Returns:
            Localized wavelet as QTTSignal
        """
        if not self.approximators:
            self._build_approximators()
        
        # Create delta at node
        delta = QTTSignal.delta(self.laplacian.num_nodes, node_index)
        
        # Apply wavelet kernel
        approx = self.approximators[scale_index]
        return approx.apply(delta, round_tol=round_tol)
    
    def energy_spectrum(self, signal: QTTSignal) -> torch.Tensor:
        """
        Compute energy distribution across scales.
        
        Returns:
            Tensor of energies [E_1, E_2, ..., E_J] for J scales
        """
        result = self.transform(signal)
        return torch.tensor(result.energy_per_scale(), dtype=torch.float64)
    
    def __repr__(self) -> str:
        return (f"QTTGraphWavelet(scales={self.scales}, "
                f"kernel='{self.kernel_type}', order={self.chebyshev_order})")
