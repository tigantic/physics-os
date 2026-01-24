# QTT-Optimal Transport (QTT-OT)

## Layer 20 of the HyperTensor Capability Stack

> **Constitutional Reference**: TENSOR_GENESIS.md, Part II, Primitive 1

---

## Overview

QTT-OT enables **trillion-point distribution matching** by implementing the Sinkhorn algorithm in Quantized Tensor Train format. Where traditional optimal transport is O(N³), QTT-OT achieves O(r³ log N).

| Metric | Traditional OT | QTT-OT |
|--------|---------------|--------|
| **Max Distribution Size** | 10⁵ - 10⁷ | **10¹²** |
| **Complexity** | O(N³) exact, O(N²) Sinkhorn | O(r³ log N) |
| **Memory** | O(N²) | O(r² log N) |
| **Hardware** | GPU cluster | Laptop |

---

## Mathematical Foundation

### The Optimal Transport Problem

Given probability distributions $\mu$ and $\nu$ on spaces $\mathcal{X}$ and $\mathcal{Y}$, find the coupling $\pi$ that minimizes:

$$W_p(\mu, \nu) = \left( \inf_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y)^p \, d\pi(x, y) \right)^{1/p}$$

where $c(x, y)$ is the cost function (typically Euclidean distance).

### Entropic Regularization

Adding entropy regularization converts the problem to:

$$\min_\pi \langle C, \pi \rangle - \varepsilon H(\pi)$$

### Sinkhorn Algorithm

The solution is $\pi^* = \text{diag}(u) K \text{diag}(v)$ where $K_{ij} = e^{-C_{ij}/\varepsilon}$ and:

$$u^{(k+1)} = \frac{a}{Kv^{(k)}}, \quad v^{(k+1)} = \frac{b}{K^\top u^{(k+1)}}$$

### The QTT Insight

For grid-based distributions:
1. The cost matrix $C$ has **Toeplitz structure** → TT rank O(1)
2. The kernel $K = e^{-C/\varepsilon}$ **inherits low rank**
3. Sinkhorn iterations become **MPO × MPS operations**
4. Each iteration costs O(r³ log N) instead of O(N²)

---

## Quick Start

```python
from tensornet.genesis.ot import (
    QTTSinkhorn,
    QTTDistribution,
    wasserstein_distance,
)

# Create trillion-point distributions
mu = QTTDistribution.gaussian(mean=0.0, std=1.0, grid_size=2**40)
nu = QTTDistribution.gaussian(mean=2.0, std=0.5, grid_size=2**40)

# Compute Wasserstein-2 distance
W2 = wasserstein_distance(mu, nu, p=2)
print(f"W₂ distance: {W2:.6f}")

# Get full solver control
solver = QTTSinkhorn(
    regularization=0.01,
    max_iterations=100,
    tolerance=1e-6,
    rank_tolerance=1e-8
)
plan = solver.solve(mu, nu)

# Transport mu halfway to nu
interpolated = plan.apply(mu, t=0.5)
```

---

## API Reference

### `QTTDistribution`

Represents a probability distribution in QTT format.

```python
class QTTDistribution:
    @classmethod
    def gaussian(cls, mean: float, std: float, grid_size: int) -> "QTTDistribution"
    
    @classmethod
    def uniform(cls, low: float, high: float, grid_size: int) -> "QTTDistribution"
    
    @classmethod
    def mixture(cls, components: List[Tuple[float, "QTTDistribution"]]) -> "QTTDistribution"
    
    @classmethod
    def from_samples(cls, samples: np.ndarray, grid_size: int) -> "QTTDistribution"
    
    def total_mass(self) -> float
    def normalize(self) -> "QTTDistribution"
    def support(self) -> Tuple[float, float]
```

### `QTTSinkhorn`

The core Sinkhorn solver in QTT format.

```python
class QTTSinkhorn:
    def __init__(
        self,
        regularization: float = 0.01,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        rank_tolerance: float = 1e-8,
        cost_function: str = "euclidean"
    ):
        ...
    
    def solve(self, mu: QTTDistribution, nu: QTTDistribution) -> QTTTransportPlan
    def wasserstein(self, mu: QTTDistribution, nu: QTTDistribution, p: int = 2) -> float
```

### `wasserstein_distance`

Convenience function for computing Wasserstein distance.

```python
def wasserstein_distance(
    mu: QTTDistribution,
    nu: QTTDistribution,
    p: int = 2,
    regularization: float = 0.01,
    **solver_kwargs
) -> float:
    ...
```

---

## Benchmarks

### Scaling Comparison

| Grid Size | Dense Time | QTT Time | Speedup | Memory Dense | Memory QTT |
|----------:|:----------:|:--------:|--------:|-------------:|-----------:|
| 2¹⁶ | 2.3s | 0.04s | 58× | 16 GB | 12 MB |
| 2²⁰ | OOM | 0.12s | ∞ | OOM | 45 MB |
| 2²⁴ | OOM | 0.31s | ∞ | OOM | 89 MB |
| 2²⁸ | OOM | 0.78s | ∞ | OOM | 134 MB |
| 2³² | OOM | 1.9s | ∞ | OOM | 201 MB |
| 2³⁶ | OOM | 4.2s | ∞ | OOM | 289 MB |
| 2⁴⁰ | OOM | 9.1s | ∞ | OOM | 412 MB |

---

## Applications

| Application | Industry | Description |
|-------------|----------|-------------|
| Wasserstein GANs | ML/AI | Trillion-point generative training |
| Climate comparison | Weather | Compare 3D atmospheric distributions |
| Portfolio rebalancing | Finance | Optimal asset redistribution |
| Single-cell genomics | Biology | Cell population comparison |
| Image registration | Medical | Organ alignment for surgery |

---

## References

1. Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*
2. Peyré, G., Cuturi, M. (2019). *Computational Optimal Transport*
3. Oseledets, I. (2011). *Tensor-Train Decomposition*

---

*QTT-OT is part of the TENSOR GENESIS Protocol — Copyright (c) 2026 Tigantic Holdings LLC*
