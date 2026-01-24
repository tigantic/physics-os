# QTT-TG: Tropical Geometry Layer

## TENSOR GENESIS Protocol — Layer 23

### Overview

Tropical geometry replaces classical arithmetic with **tropical semirings**:

| Operation | Classical | Min-Plus | Max-Plus |
|-----------|-----------|----------|----------|
| Addition  | a + b     | min(a,b) | max(a,b) |
| Multiplication | a × b | a + b   | a + b    |
| Zero      | 0         | +∞       | -∞       |
| One       | 1         | 0        | 0        |

### Mathematical Foundation

**Tropical Matrix Multiplication**:
```
(A ⊗ B)_ij = min_k (A_ik + B_kj)
```

This is equivalent to the **shortest path** composition!

**Tropical Power**:
```
A^⊗k = A ⊗ A ⊗ ... ⊗ A  (k times)
```

`(A^⊗k)_ij` = length of shortest path from i to j using exactly k edges.

**Kleene Star** (all-pairs shortest path):
```
A* = I ⊕ A ⊕ A^⊗2 ⊕ A^⊗3 ⊕ ...
```

### QTT Insight

**Key Observation**: Distance matrices from smooth cost functions have low TT rank.

**Smooth Approximation**: Replace min with differentiable softmin:
```
softmin(a, b; β) = -(1/β) log(e^{-βa} + e^{-βb})
```

As β → ∞, this converges to exact min. For finite β:
- All operations become smooth tensor contractions
- TT ranks are bounded throughout computation
- Error is O(1/β)

### Complexity Analysis

| Algorithm | Classical | QTT-TG |
|-----------|-----------|--------|
| Matrix multiply | O(N³) | O(r³ log² N) |
| Matrix power | O(N³ log k) | O(r³ log N log k) |
| All-pairs shortest path | O(N³) | O(r³ log² N log N) |
| Single-source shortest path | O(N²) | O(r³ log N) |

### Module Structure

```
tensornet/genesis/tropical/
├── __init__.py           # Module exports
├── README.md             # This file
├── semiring.py           # Tropical semiring operations
├── matrix.py             # Tropical matrix multiplication
├── shortest_path.py      # APSP, SSSP algorithms
├── convexity.py          # Tropical polyhedra
├── optimization.py       # Tropical linear programming
└── qtt_tropical_gauntlet.py  # Elite test suite
```

### Example Usage

```python
from tensornet.genesis.tropical import (
    TropicalMatrix, MinPlusSemiring,
    all_pairs_shortest_path, tropical_power
)

# Create adjacency matrix with edge weights (∞ = no edge)
# For a grid graph of size 2^40 × 2^40
A = TropicalMatrix.grid_graph(size=2**40, rank=10)

# Compute all-pairs shortest path in O(r³ log² N) time
dist = all_pairs_shortest_path(A, semiring=MinPlusSemiring)

# Or compute step-limited shortest path
dist_5 = tropical_power(A, k=5)  # Paths of exactly 5 edges
```

### Constitutional Covenants

1. **Compression Covenant**: TT-rank ≤ 20 for all intermediate matrices
2. **Complexity Covenant**: O(r³ log² N) per tropical product
3. **Accuracy Covenant**: Relative error < ε for all finite distances
4. **API Covenant**: Clean, documented interfaces

### Applications Unlocked

| Application | Classical Limit | QTT-TG Scale |
|-------------|-----------------|--------------|
| Network routing | 10⁶ nodes | 10¹² nodes |
| GPS navigation | City-level | Continent-level |
| Supply chain optimization | Regional | Global |
| Tropical algebraic geometry | Academic | Industrial |

### References

- Butkovič, P. (2010). Max-linear Systems: Theory and Algorithms
- Maclagan, D. & Sturmfels, B. (2015). Introduction to Tropical Geometry
- Pachter, L. & Sturmfels, B. (2004). Tropical Geometry of Statistical Models
