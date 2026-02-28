"""
QTT-Spectral Graph Wavelets (QTT-SGW) — Layer 21

Multi-scale graph signal analysis using spectral wavelets in QTT format.

The key insight: For structured graphs (grids, lattices), the graph Laplacian L
has low TT rank. Matrix functions g(sL) are computed via Chebyshev polynomial
approximation, where each polynomial term is an MPO operation.

Key Components:
    - QTTLaplacian: Graph Laplacian as QTT-MPO
    - QTTSignal: Signals on graphs in QTT format
    - QTTGraphWavelet: Spectral wavelet transform
    - GraphFilter: Low-pass, high-pass, band-pass filters

Mathematical Foundation:
    Spectral Graph Wavelet: ψ_s = g(sL) where g is wavelet kernel
    For signal f: W_f(s,n) = ⟨ψ_{s,n}, f⟩ = Σ_k g(sλ_k) f̂(k) χ_k(n)
    
    Chebyshev approximation: g(sL) ≈ Σ_{k=0}^{K} c_k T_k(L̃)
    where L̃ = 2L/λ_max - I maps spectrum to [-1, 1]

Why QTT:
    - Grid Laplacian has TT rank 3 (tridiagonal structure)
    - Chebyshev recurrence: T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)
    - Each iteration is MPO×MPS with bounded rank growth
    - Full transform: O(r³ K log N) vs O(N³) for eigendecomposition

References:
    - Hammond, Vandergheynst, Gribonval (2011) "Wavelets on Graphs"
    - Shuman et al. (2013) "The Emerging Field of Signal Processing on Graphs"

Example:
    >>> from ontic.genesis.sgw import QTTLaplacian, QTTGraphWavelet, QTTSignal
    >>> 
    >>> # Create 1D grid Laplacian (10^12 nodes)
    >>> L = QTTLaplacian.grid_1d(2**40)
    >>> 
    >>> # Multi-scale wavelet transform
    >>> wavelet = QTTGraphWavelet(L, scales=[1, 2, 4, 8])
    >>> signal = QTTSignal.random(L.num_nodes)
    >>> coefficients = wavelet.transform(signal)
    >>>
    >>> # Analyze energy at each scale
    >>> for s, coef in zip(wavelet.scales, coefficients):
    ...     print(f"Scale {s}: Energy = {coef.norm()**2:.4f}")

TENSOR GENESIS Protocol — Layer 21
"""

from .laplacian import QTTLaplacian, grid_laplacian_1d, grid_laplacian_2d, grid_laplacian_3d
from .graph_signals import QTTSignal
from .chebyshev import chebyshev_coefficients, chebyshev_approximation, ChebyshevApproximator
from .wavelets import QTTGraphWavelet, WaveletResult, mexican_hat_kernel, heat_kernel, meyer_kernel
from .filters import GraphFilter, LowPassFilter, HighPassFilter, BandPassFilter

__all__ = [
    # Laplacian
    'QTTLaplacian',
    'grid_laplacian_1d',
    'grid_laplacian_2d', 
    'grid_laplacian_3d',
    
    # Signals
    'QTTSignal',
    
    # Chebyshev
    'chebyshev_coefficients',
    'chebyshev_approximation',
    'ChebyshevApproximator',
    
    # Wavelets
    'QTTGraphWavelet',
    'WaveletResult',
    'mexican_hat_kernel',
    'heat_kernel',
    'meyer_kernel',
    
    # Filters
    'GraphFilter',
    'LowPassFilter',
    'HighPassFilter',
    'BandPassFilter',
]

__version__ = '1.0.0'
__layer__ = 21
__primitive__ = 'QTT-SGW'
