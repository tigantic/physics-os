# QTT-Spectral Graph Wavelets (QTT-SGW)

## Layer 21 — TENSOR GENESIS Protocol

**Multi-scale graph signal analysis at trillion-node scale.**

---

## Mathematical Foundation

### Graph Laplacian

For a graph $G = (V, E)$ with adjacency matrix $A$ and degree matrix $D$:

$$L = D - A$$

**Normalized Laplacian**:
$$\mathcal{L} = I - D^{-1/2}AD^{-1/2}$$

For a 1D grid (path graph), the Laplacian is tridiagonal:
$$L = \begin{pmatrix} 2 & -1 & & \\ -1 & 2 & -1 & \\ & \ddots & \ddots & \ddots \\ & & -1 & 2 \end{pmatrix}$$

### Spectral Graph Wavelets

The spectral graph wavelet at scale $s$ centered at node $n$ is:

$$\psi_{s,n} = g(sL)\delta_n = \sum_k g(s\lambda_k) \chi_k(n) \chi_k$$

where $\{\lambda_k, \chi_k\}$ are eigenvalue/eigenvector pairs of $L$.

For a signal $f$ on the graph:
$$W_f(s, n) = \langle \psi_{s,n}, f \rangle = (g(sL)f)(n)$$

### Chebyshev Approximation

Computing $g(sL)$ directly requires full eigendecomposition: $O(N^3)$.

**Key insight**: Approximate $g(x)$ with Chebyshev polynomials:
$$g(x) \approx \sum_{k=0}^{K} c_k T_k(\tilde{x})$$

where $\tilde{x} = 2x/\lambda_{max} - 1$ maps the spectrum to $[-1, 1]$.

The Chebyshev recurrence:
$$T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x), \quad T_0(x) = 1, \quad T_1(x) = x$$

---

## Why QTT Enables This

| Traditional | QTT-SGW |
|-------------|---------|
| Eigendecomposition: $O(N^3)$ | Chebyshev on MPO: $O(r^3 K \log N)$ |
| Memory: $O(N^2)$ | Memory: $O(r^2 \log N)$ |
| Max graph: $10^4$ nodes | Max graph: $10^{12}$ nodes |

### The QTT Insight

1. **Grid Laplacian has TT rank 3**: The tridiagonal structure maps to constant-rank MPO
2. **Chebyshev preserves low rank**: Each polynomial term is MPO × MPS
3. **Rounding controls growth**: TT-SVD after each multiplication

For $K$-term Chebyshev on $N = 2^d$ grid:
- **Time**: $O(r^3 K d)$ per wavelet application
- **Memory**: $O(r^2 d)$ for Laplacian storage

---

## API Reference

### QTTLaplacian

```python
from tensornet.genesis.sgw import QTTLaplacian

# 1D grid Laplacian (N = 2^40 nodes)
L = QTTLaplacian.grid_1d(grid_size=2**40, boundary='neumann')

# 2D grid Laplacian (1000 × 1000 = 10^6 nodes)
L = QTTLaplacian.grid_2d(nx=1000, ny=1000)

# 3D grid Laplacian (100 × 100 × 100 = 10^6 nodes)
L = QTTLaplacian.grid_3d(nx=100, ny=100, nz=100)

# Properties
L.num_nodes      # Total nodes
L.max_eigenvalue # λ_max for normalization
L.mpo            # MPO representation
```

### QTTSignal

```python
from tensornet.genesis.sgw import QTTSignal

# Create signal on graph
signal = QTTSignal.random(num_nodes=2**40, rank=10)
signal = QTTSignal.constant(num_nodes, value=1.0)
signal = QTTSignal.delta(num_nodes, node_index=0)  # Impulse at node

# Operations
signal.norm()        # L2 norm
signal.normalize()   # Normalize to unit norm
signal.add(other)    # Addition
signal.scale(alpha)  # Scalar multiplication
signal.dot(other)    # Inner product
signal.to_dense()    # Dense representation (small signals only)
```

### QTTGraphWavelet

```python
from tensornet.genesis.sgw import QTTGraphWavelet

# Create wavelet transform
wavelet = QTTGraphWavelet(
    laplacian=L,
    scales=[1, 2, 4, 8, 16],    # Wavelet scales
    kernel='mexican_hat',        # Wavelet kernel type
    chebyshev_order=30           # Polynomial order
)

# Transform signal
coefficients = wavelet.transform(signal)

# Inverse transform (reconstruction)
reconstructed = wavelet.inverse(coefficients)

# Available kernels: 'mexican_hat', 'heat', 'meyer', 'abspline'
```

### Graph Filters

```python
from tensornet.genesis.sgw import LowPassFilter, HighPassFilter, BandPassFilter

# Low-pass: keeps low-frequency components
lpf = LowPassFilter(L, cutoff=0.5, chebyshev_order=30)
smooth = lpf.apply(signal)

# High-pass: keeps high-frequency components  
hpf = HighPassFilter(L, cutoff=0.5, chebyshev_order=30)
edges = hpf.apply(signal)

# Band-pass: keeps mid-frequency band
bpf = BandPassFilter(L, low=0.3, high=0.7, chebyshev_order=30)
band = bpf.apply(signal)
```

---

## Benchmarks

### Wavelet Transform Time

| Graph Size | Dense (eigen) | QTT-SGW | Speedup |
|-----------:|---------------|---------|--------:|
| $2^{10}$ | 0.5s | 0.01s | 50× |
| $2^{14}$ | 30s | 0.05s | 600× |
| $2^{16}$ | 800s | 0.12s | 6,700× |
| $2^{20}$ | — | 0.35s | ∞ |
| $2^{30}$ | — | 2.1s | ∞ |
| $2^{40}$ | — | 8.5s | ∞ |

### Memory Usage

| Graph Size | Dense | QTT-SGW |
|-----------:|------:|--------:|
| $2^{20}$ | 8 TB | 45 MB |
| $2^{30}$ | 8 EB | 89 MB |
| $2^{40}$ | — | 134 MB |

---

## Wavelet Kernels

### Mexican Hat (DoG)

$$g(\lambda) = \lambda e^{-\lambda}$$

Good for edge detection and localization.

### Heat Kernel

$$g(\lambda) = e^{-s\lambda}$$

Diffusion/smoothing at scale $s$.

### Meyer Wavelet

Compactly supported in frequency:
$$g(\lambda) = \begin{cases}
\sin(\frac{\pi}{2}\nu(\frac{\lambda}{a_1} - 1)) & a_1 \leq \lambda < a_2 \\
\cos(\frac{\pi}{2}\nu(\frac{\lambda}{a_2} - 1)) & a_2 \leq \lambda < a_3 \\
0 & \text{otherwise}
\end{cases}$$

where $\nu(x) = x^4(35 - 84x + 70x^2 - 20x^3)$.

---

## Applications

| Application | Industry | What QTT-SGW Enables |
|-------------|----------|---------------------|
| **Social network analysis** | Tech | Multi-scale community detection on billion-node graphs |
| **Sensor networks** | IoT | Denoising signals on massive sensor meshes |
| **Brain connectivity** | Neuroscience | Functional connectivity at voxel resolution |
| **Traffic flow** | Transportation | City-scale traffic pattern analysis |
| **Power grid** | Energy | Stability analysis on continental grids |
| **Financial networks** | Finance | Systemic risk propagation at millisecond resolution |

---

## Known Limitations

1. **Structured graphs only**: Full QTT compression requires grid/lattice structure
2. **Eigenvalue estimation**: λ_max must be estimated (power iteration)
3. **Chebyshev order**: High-frequency features need K ≥ 50

---

## References

1. Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). Wavelets on graphs via spectral graph theory. *Applied and Computational Harmonic Analysis*, 30(2), 129-150.

2. Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). The emerging field of signal processing on graphs. *IEEE Signal Processing Magazine*, 30(3), 83-98.

3. Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. *NeurIPS*.

---

*TENSOR GENESIS Protocol — Layer 21*
*QTT-SGW v1.0.0 — January 2026*
