# cem-qtt — Computational Electromagnetics via Quantized Tensor Trains

> **Maxwell's equations. Fixed-point. Tensor-compressed. Deterministic.**

## What Is This?

A complete FDTD Maxwell solver in Q16.16 fixed-point arithmetic with MPS/MPO
tensor network compression. Solves the full 3D Maxwell curl equations on a
Yee lattice with leapfrog time integration.

**Governing Equations:**
```
∂B/∂t = −∇ × E
∂E/∂t = (1/ε)(∇ × H − J − σE)
```

## Architecture

```
Maxwell's Equations
       │
       ▼
 Yee Lattice (staggered E/H grid)
       │
       ▼
 FDTD Leapfrog (explicit timestepping, CFL-bounded)
       │
       ▼
 Q16.16 Fixed-Point (deterministic, ZK-friendly)
       │
       ▼
 QTT Compression (MPS/MPO, SVD truncation, χ_max bounded)
       │
       ▼
 Conservation Verification (Poynting theorem)
```

## Modules

| Module | Description |
|--------|-------------|
| `q16` | Q16.16 fixed-point arithmetic |
| `mps` | Matrix Product State field representation |
| `mpo` | Matrix Product Operator for differential operators |
| `material` | Electromagnetic material properties (ε, μ, σ) |
| `fdtd` | Yee lattice FDTD solver with leapfrog integration |
| `pml` | Perfectly Matched Layer absorbing boundaries |
| `conservation` | Poynting theorem energy conservation verification |

## Key Properties

- **Deterministic**: Q16.16 fixed-point eliminates floating-point nondeterminism
- **Stable**: CFL condition enforced, Courant number 0.5
- **Conservative**: Energy conservation verified via Poynting theorem
- **Compressible**: QTT format gives O(log N) memory scaling for smooth fields
- **ZK-Ready**: Fixed-point arithmetic maps directly to Halo2 circuit constraints

## Validation (28/28 PASSED)

```
Stage 1:  Zero-field stability          ✓
Stage 2:  Energy conservation (vacuum)  ✓  (max drift: 0.032)
Stage 3:  Source injection              ✓
Stage 4:  PEC boundary enforcement      ✓
Stage 5:  CFL condition validation      ✓
Stage 6:  Field propagation             ✓  (193 nonzero H-points)
Stage 7:  Lossy medium energy decay     ✓  (0.500 → 0.339)
Stage 8:  Deterministic execution       ✓  (bit-identical)
Stage 9:  Poynting flux                 ✓
Stage 10: Long-run stability (100 steps)✓
Stage 11: Dielectric material           ✓  (cb ratio 4:1)
Stage 12: PML damping profile           ✓  (monotonic grading)
Stage 13: Conservation verifier         ✓
Stage 14: Q16.16 arithmetic            ✓
Stage 15: Architecture validation       ✓
```

## Boundary Conditions

- **Periodic**: Wrap-around (default)
- **PEC**: Perfect Electric Conductor (tangential E = 0)
- **PML**: Perfectly Matched Layer (absorbing, Berenger split-field)

## Materials

- Vacuum, dielectric (εr), conductor (σ), lossy dielectric (εr, σ)
- Slab and sphere geometry primitives
- Spatially-varying material maps

## Sources

- Gaussian pulse
- Sinusoidal (CW)
- Plane wave (soft source)

## Integration

This crate is designed to integrate with:
- **HyperTensor**: Physics engine (upstream solver)
- **FluideLite**: ZK proof framework (downstream verification)
- **QTT compression**: Published O(log N) scaling (Zenodo)
- **INVARIAN/Trust Fabric**: Post-quantum provenance signatures

## Build

```bash
cargo build --release
cargo test
cargo test --test integration
```

## License

© 2026 Brad McAllister. All rights reserved. PROPRIETARY.
