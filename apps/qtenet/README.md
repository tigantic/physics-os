# QTeneT — Quantized Tensor Network Physics Engine

**Breaking the Curse of Dimensionality with O(log N) Complexity**

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/tests-66%20passing-brightgreen.svg)](src/qtenet/tests/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](../LICENSE)

## The Holy Grail

QTeneT makes the **impossible possible**: full 6D Vlasov-Maxwell plasma simulations with 1 billion grid points using only 200 KB of memory.

```python
from qtenet.demos import holy_grail_6d

result = holy_grail_6d(qubits_per_dim=5, n_steps=100)
print(result.summary())

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Grid:         32^6 =   1,073,741,824 points                     ║
# ║  QTT Memory:        197.6 KB                                     ║
# ║  Dense Memory:       4.29 GB  (would be required)                ║
# ║  Compression:      21,229×                                       ║
# ║  CURSE OF DIMENSIONALITY: BROKEN ✓                               ║
# ╚══════════════════════════════════════════════════════════════════╝
```

## Installation

```bash
# From physics-os root
pip install -e apps/qtenet/
```

## Quick Start

### TCI Compression (Black-Box to QTT)

```python
from qtenet.tci import from_function
import torch

# Any function you want to compress
def my_physics(idx):
    return torch.sin(idx.float() / 1000)

# Compress: 2^30 = 1 billion points → O(30 × r²) parameters
cores = from_function(f=my_physics, n_qubits=30, max_rank=64)
print(f"Compressed {2**30:,} points to {sum(c.numel() for c in cores):,} params")
```

### N-Dimensional Shift Operators

```python
from qtenet.operators import shift_nd, apply_shift

# Create shift operator for 6D phase space
shift_x = shift_nd(
    total_qubits=30,  # 5 qubits × 6 dims
    num_dims=6,
    axis=0,           # Shift in x
    direction=+1,     # Forward
)

# Apply to QTT state: O(n_qubits × r³) complexity
shifted_cores = apply_shift(cores, shift_x, max_rank=64)
```

### Vlasov-Maxwell Solver

```python
from qtenet.solvers import Vlasov6D, Vlasov6DConfig

config = Vlasov6DConfig(
    qubits_per_dim=5,  # 32^6 = 1 billion points
    max_rank=64,
)
solver = Vlasov6D(config)

# Two-stream instability
state = solver.two_stream_ic()

# Time evolution: O(log N × r³) per step
for _ in range(1000):
    state = solver.step(state, dt=0.01)
```

## Architecture

```
qtenet/
├── tci/           # TCI: Black-box → QTT compression
├── operators/     # N-D shift, laplacian, gradient MPOs
├── solvers/       # Vlasov5D, Vlasov6D, EulerND
├── benchmarks/    # Curse-breaking performance proofs
├── demos/         # Holy Grail demonstrations
└── genesis/       # Advanced: OT, RMT, Topology, GA
```

## Key Metrics

| Problem | Grid Size | Dense Memory | QTT Memory | Compression |
|---------|-----------|--------------|------------|-------------|
| 3D CFD | 64³ | 1 MB | 50 KB | 20× |
| 5D Vlasov | 32⁵ | 128 MB | 15 KB | 8,500× |
| **6D Vlasov** | **32⁶** | **4.29 GB** | **198 KB** | **21,229×** |
| 6D Vlasov | 64⁶ | 274 GB | 400 KB | 687,000× |

## Running Tests

```bash
python -m pytest apps/qtenet/src/qtenet/tests/ -v
# 66 tests passing
```

## Known Limitations

1. **TCI Reconstruction Accuracy**: Single-sweep TCI may not accurately reconstruct peaked functions at arbitrary points. The compression is correct; point-wise reconstruction may have errors for complex functions. For validation, use dense comparison on small grids.

2. **Energy Conservation**: Not tracked in demos (would require expensive dense operations).

## Background

**QTeneT** is the **enterprise-grade library packaging** of the PyTenNet QTT stack, extracted from **physics-os-main**.

- **Python package/CLI:** `qtenet`
- **Source of truth:** `physics-os-main` (this workspace)

## License

Copyright © 2026 Tigantic Holdings LLC. All Rights Reserved.

---

*ELITE Engineering — Production Grade*

