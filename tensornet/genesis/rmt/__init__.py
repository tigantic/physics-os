"""
QTT-RMT: Random Matrix Theory in Quantized Tensor Train Format

Layer 22 of the TENSOR GENESIS Protocol.

This module provides QTT-compressed random matrix theory capabilities,
enabling eigenvalue statistics and spectral analysis at scales impossible
with traditional O(N³) eigendecomposition.

Key Insight:
    For structured random matrices (banded, Toeplitz-like), the resolvent
    G(z) = (H - zI)^{-1} can be computed in TT format. Spectral density
    is extracted via the Stieltjes transform:
    
        ρ(λ) = -(1/π) lim_{η→0+} Im[Tr(G(λ + iη))]
    
    Complexity: O(r³ log N) vs O(N³) eigendecomposition.

Classes:
    QTTEnsemble: Random matrix ensembles in QTT format
    QTTResolvent: Resolvent computation G(z) = (H - zI)^{-1}
    SpectralDensity: Eigenvalue distribution estimation
    WignerSemicircle: Wigner semicircle law verification
    MarchenkoPastur: Marchenko-Pastur law for sample covariance

Functions:
    spectral_density: Compute eigenvalue density from QTT matrix
    wigner_semicircle: Theoretical Wigner semicircle
    marchenko_pastur: Theoretical Marchenko-Pastur density
    free_convolution: Free additive convolution via R-transform

Example:
    >>> from tensornet.genesis.rmt import QTTEnsemble, spectral_density
    >>> 
    >>> # Create GOE (Gaussian Orthogonal Ensemble) matrix
    >>> H = QTTEnsemble.goe(size=2**20, rank=10, seed=42)
    >>> 
    >>> # Compute spectral density
    >>> lambdas, rho = spectral_density(H, num_points=1000)
    >>> 
    >>> # Compare to Wigner semicircle
    >>> from tensornet.genesis.rmt import wigner_semicircle
    >>> rho_theory = wigner_semicircle(lambdas)

Constitutional Reference: TENSOR_GENESIS.md, Article II Section 2.1

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

__version__ = "1.0.0"
__author__ = "Bradly Biron Baker Adams"
__layer__ = 22

# Ensembles
from tensornet.genesis.rmt.ensembles import (
    QTTEnsemble,
    goe_matrix,
    gue_matrix,
    wishart_matrix,
    wigner_matrix,
)

# Resolvent
from tensornet.genesis.rmt.resolvent import (
    QTTResolvent,
    compute_resolvent,
    resolvent_trace,
)

# Spectral Density
from tensornet.genesis.rmt.spectral_density import (
    SpectralDensity,
    spectral_density,
    stieltjes_transform,
    inverse_stieltjes,
)

# Universality Laws
from tensornet.genesis.rmt.universality import (
    WignerSemicircle,
    MarchenkoPastur,
    wigner_semicircle,
    marchenko_pastur,
    verify_universality,
)

# Free Probability
from tensornet.genesis.rmt.free_probability import (
    FreeConvolution,
    r_transform,
    s_transform,
    free_additive_convolution,
    free_multiplicative_convolution,
)

__all__ = [
    # Ensembles
    "QTTEnsemble",
    "goe_matrix",
    "gue_matrix", 
    "wishart_matrix",
    "wigner_matrix",
    # Resolvent
    "QTTResolvent",
    "compute_resolvent",
    "resolvent_trace",
    # Spectral Density
    "SpectralDensity",
    "spectral_density",
    "stieltjes_transform",
    "inverse_stieltjes",
    # Universality
    "WignerSemicircle",
    "MarchenkoPastur",
    "wigner_semicircle",
    "marchenko_pastur",
    "verify_universality",
    # Free Probability
    "FreeConvolution",
    "r_transform",
    "s_transform",
    "free_additive_convolution",
    "free_multiplicative_convolution",
]
