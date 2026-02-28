# QTT-PH: Persistent Homology with QTT Acceleration

## TENSOR GENESIS Protocol — Layer 25

### Overview

This module implements Persistent Homology using QTT tensor compression
for efficient computation of topological features across scales.

### Mathematical Foundation

**Homology Theory:**
- Chain groups: $C_k$ = formal sums of k-simplices
- Boundary operator: $\partial_k: C_k \to C_{k-1}$
- Homology groups: $H_k = \ker(\partial_k) / \text{im}(\partial_{k+1})$
- Betti numbers: $\beta_k = \dim(H_k)$

**Persistent Homology:**
- Filtration: $K_0 \subseteq K_1 \subseteq \cdots \subseteq K_n$
- Birth time: when a topological feature appears
- Death time: when it merges with an older feature
- Persistence: death - birth

**QTT Insight:**
For structured data (lattices, low-dimensional manifolds), the boundary
matrix has low TT rank due to the locality of simplex boundaries.

### Complexity

| Operation | Classical | QTT-Accelerated |
|-----------|-----------|-----------------|
| Boundary matrix storage | O(N²) | O(r² log N) |
| Matrix-chain product | O(N³) | O(r³ log N) |
| Persistence computation | O(N³) | O(r³ log N · N) |

### Module Structure

- `simplicial.py`: Simplicial complex construction (Rips, Čech)
- `boundary.py`: Boundary operator matrices in QTT format
- `persistence.py`: Persistent homology via matrix reduction
- `distances.py`: Bottleneck and Wasserstein distances on diagrams

### Example Usage

```python
from ontic.genesis.topology import (
    RipsComplex, compute_persistence, bottleneck_distance
)

# Build Rips complex
points = torch.randn(100, 3)
complex = RipsComplex(points, max_radius=2.0, max_dim=2)

# Compute persistence
diagram = compute_persistence(complex)

# Extract features
print(f"H_0 features: {len(diagram.pairs[0])}")
print(f"H_1 features: {len(diagram.pairs[1])}")

# Compare diagrams
d_bottle = bottleneck_distance(diagram1, diagram2)
```

### References

1. Edelsbrunner, Letscher, Zomorodian - "Topological Persistence and
   Simplification" (2002)
2. Zomorodian, Carlsson - "Computing Persistent Homology" (2005)
3. Cohen-Steiner, Edelsbrunner, Harer - "Stability of Persistence Diagrams"
   (2007)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
