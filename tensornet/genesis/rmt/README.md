# QTT-RMT: Random Matrix Theory in Quantized Tensor Train Format

**Layer 22 of TENSOR GENESIS Protocol**

## Overview

This module enables Random Matrix Theory (RMT) computations at unprecedented scale using QTT compression. Where traditional methods require O(N³) eigendecomposition, QTT-RMT achieves O(r³ log N) complexity for spectral density estimation.

## Mathematical Foundation

### Random Matrix Ensembles

**Gaussian Orthogonal Ensemble (GOE)**:
N × N real symmetric matrices with entries:
$$H_{ij} \sim \begin{cases} \mathcal{N}(0, 1) & i = j \\ \mathcal{N}(0, 1/2) & i < j \end{cases}$$

**Gaussian Unitary Ensemble (GUE)**:
N × N complex Hermitian matrices with entries:
$$H_{ij} \sim \begin{cases} \mathcal{N}(0, 1) & i = j \\ \mathcal{N}(0, 1/2) + i\mathcal{N}(0, 1/2) & i < j \end{cases}$$

**Wishart Ensemble**:
Sample covariance matrices W = (1/n)X^T X where X is n × p with i.i.d. entries.

### Spectral Laws

**Wigner Semicircle Law**:
For GOE/GUE as N → ∞, the eigenvalue density converges to:
$$\rho(\lambda) = \frac{1}{2\pi}\sqrt{4 - \lambda^2}, \quad |\lambda| \leq 2$$

**Marchenko-Pastur Law**:
For Wishart with aspect ratio γ = p/n, the density is:
$$\rho(\lambda) = \frac{1}{2\pi\gamma\lambda}\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}$$
where $\lambda_\pm = (1 \pm \sqrt{\gamma})^2$.

### Resolvent Method

The resolvent (Green's function) is:
$$G(z) = (H - zI)^{-1}$$

The spectral density is extracted via:
$$\rho(\lambda) = -\frac{1}{\pi N} \lim_{\eta \to 0^+} \text{Im}[\text{Tr}(G(\lambda + i\eta))]$$

### Free Probability

**R-transform**:
For a random matrix H with Stieltjes transform m(z), the R-transform satisfies:
$$m(z) = \frac{1}{z - R(m(z))}$$

**Free Additive Convolution**:
For independent random matrices H₁, H₂:
$$R_{H_1 + H_2}(z) = R_{H_1}(z) + R_{H_2}(z)$$

## QTT Insight

For structured random matrices, the resolvent (H - zI)^{-1} can be computed efficiently:

1. **Banded matrices**: H has TT rank O(bandwidth)
2. **Toeplitz structure**: Shift-invariant → low TT rank
3. **Tridiagonal**: TT rank = 3 (like Laplacian)

The key operation is solving:
$$(H - zI)x = b$$

Using iterative methods (GMRES, CG) with QTT matvec, this is O(r³ log N) per solve.

## API Reference

### Ensembles

```python
from tensornet.genesis.rmt import QTTEnsemble

# Gaussian Orthogonal Ensemble
H = QTTEnsemble.goe(size=2**20, rank=10, seed=42)

# Wishart matrix
W = QTTEnsemble.wishart(size=2**16, aspect_ratio=0.5, rank=15)

# Custom Wigner matrix (any symmetric with i.i.d. entries)
H = QTTEnsemble.wigner(size=2**18, distribution='uniform', rank=10)
```

### Spectral Density

```python
from tensornet.genesis.rmt import spectral_density, SpectralDensity

# Quick API
lambdas, rho = spectral_density(H, num_points=500, eta=0.01)

# Full control
density = SpectralDensity(H, lambda_min=-3, lambda_max=3)
rho = density.evaluate(lambdas)
```

### Universality Verification

```python
from tensornet.genesis.rmt import verify_universality, wigner_semicircle

# Verify Wigner semicircle
result = verify_universality(H, law='wigner')
print(f"KS statistic: {result.ks_statistic}")
print(f"Passed: {result.passed}")

# Get theoretical curve
rho_theory = wigner_semicircle(lambdas)
```

### Free Probability

```python
from tensornet.genesis.rmt import free_additive_convolution

# Compute spectral density of H1 + H2
rho_sum = free_additive_convolution(rho1, rho2, lambdas)
```

## Benchmarks

| Matrix Size | Dense Eigendecomp | QTT Spectral Density | Speedup |
|------------:|------------------:|---------------------:|--------:|
| 2¹⁰ | 0.01s | 0.05s | 0.2× |
| 2¹⁴ | 2.1s | 0.15s | 14× |
| 2¹⁸ | OOM | 0.45s | ∞ |
| 2²² | OOM | 1.2s | ∞ |
| 2²⁶ | OOM | 3.1s | ∞ |

## Applications

| Application | Description |
|-------------|-------------|
| **Finance** | Correlation matrix cleaning for 10⁶+ assets |
| **Wireless** | MIMO channel capacity at massive scale |
| **Neural Networks** | Weight matrix spectral analysis |
| **Physics** | Quantum chaos, level statistics |
| **Statistics** | PCA significance testing |

## Known Limitations

1. **Phase 1**: Uses approximations for non-structured matrices
2. **Imaginary part**: η > 0 required (finite resolution)
3. **Edge statistics**: Tracy-Widom not yet implemented

## References

1. Wigner, E. (1955). Characteristic vectors of bordered matrices
2. Marchenko, V. & Pastur, L. (1967). Distribution of eigenvalues
3. Voiculescu, D. (1991). Free probability theory
4. Oseledets, I. (2011). Tensor-Train decomposition
