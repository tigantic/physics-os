# fea-qtt — Structural Mechanics FEA via Quantized Tensor Trains

> **Elasticity. Fixed-point. Tensor-compressed. Deterministic.**

## What Is This?

A static linear elasticity solver in Q16.16 fixed-point arithmetic with 8-node hexahedral (Hex8) elements, Conjugate Gradient iteration, and strain energy conservation verification.

**Governing Equations:**
```
∇·σ + f = 0        (equilibrium)
σ = D·ε             (constitutive, Hooke's law)
ε = ½(∇u + ∇uᵀ)   (strain-displacement)
→ Ku = F            (assembled system)
```

## Architecture

```
Elasticity: ∇·σ + f = 0
       │
       ▼
 Hex8 Isoparametric Elements (trilinear shape functions)
       │
       ▼
 2×2×2 Gauss Quadrature (B-matrix, Ke = ∫BᵀDB|J|dV)
       │
       ▼
 Global Assembly (sparse COO → Ku = F)
       │
       ▼
 Conjugate Gradient Solver (Q16.16 fixed-point)
       │
       ▼
 Stress Recovery (σ = D·B·u at centroids)
       │
       ▼
 Energy Conservation (½uᵀKu = ½Fᵀu)
```

## Modules

| Module | Description |
|--------|-------------|
| `q16` | Q16.16 fixed-point arithmetic |
| `material` | Isotropic linear elastic (E, ν → D matrix) |
| `element` | Hex8 shape functions, B-matrix, element stiffness |
| `mesh` | Structured hexahedral mesh generation |
| `solver` | Sparse assembly, CG solver, energy verification |

## Validation (32/32 PASSED)

```
Stage 1:  Q16.16 arithmetic             ✓
Stage 2:  Shape function properties      ✓  (partition of unity, Kronecker δ)
Stage 3:  Jacobian computation           ✓  (det(J) = 0.125 for unit cube)
Stage 4:  Constitutive matrix            ✓  (symmetric, positive diagonal)
Stage 5:  Element stiffness              ✓  (symmetric, 276/276 pairs)
Stage 6:  Mesh generation                ✓  (27 nodes, 8 elements for 2³)
Stage 7:  Uniaxial tension               ✓  (CG converged, ux = 0.298)
Stage 8:  Energy conservation            ✓  (U = ½Fᵀu = 0.060)
Stage 9:  Cantilever beam                ✓  (tip deflection uy = -0.543)
Stage 10: Deterministic execution        ✓  (bit-identical)
Stage 11: Stress recovery                ✓  (σxx dominant, VM > 0)
Stage 12: Mesh refinement                ✓  (coarse vs fine convergence)
Stage 13: Architecture validation        ✓
```

## Key Properties

- **Deterministic**: Q16.16 fixed-point, bit-identical across platforms
- **Conservative**: Strain energy = ½Fᵀu verified
- **ZK-Ready**: Integer arithmetic maps to circuit constraints
- **Compressible**: Displacement/stress fields compress via QTT

## Integration

Designed to integrate with:
- **HyperTensor**: Physics engine (upstream)
- **CEM-QTT**: Electromagnetics solver (sibling domain)
- **FluideLite**: ZK proof framework (downstream)
- **QTT compression**: Published O(log N) scaling

## Build

```bash
cargo build --release
cargo test
cargo test --test integration
```

## License

© 2026 Brad McAllister. All rights reserved. PROPRIETARY.
