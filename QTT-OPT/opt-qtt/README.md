# opt-qtt — PDE-Constrained Optimization via Quantized Tensor Trains

> **Adjoint methods. Topology optimization. Inverse problems. Deterministic.**

## What Is This?

A PDE-constrained optimization framework in Q16.16 fixed-point arithmetic with SIMP topology optimization, adjoint sensitivity analysis, mesh-independent filtering, and gradient-based inverse problem solving.

**Core Algorithms:**
```
Topology Optimization (SIMP):
  min  C(ρ) = Fᵀu(ρ)                    (compliance)
  s.t. K(ρ)u = F                          (equilibrium)
       Σρₑvₑ ≤ V*                         (volume constraint)
       E(ρ) = E_min + ρᵖ(E₀ - E_min)     (SIMP interpolation)

Adjoint Sensitivity:
  dC/dρₑ = -p·ρₑᵖ⁻¹·uₑᵀKe₀uₑ           (self-adjoint)
  dJ/dρₑ = ∂J/∂ρₑ + λᵀ∂R/∂ρₑ            (general)

Inverse Problem:
  min  J(θ) = ½‖u(θ) - u_obs‖² + λ/2‖θ‖² (Tikhonov)
  s.t. R(u,θ) = 0                          (PDE constraint)
```

## Architecture

```
Design Variables ρ (densities or parameters)
       │
       ▼
Forward Solve: K(ρ)u = F  (Quad4 plane stress, CG solver)
       │
       ▼
Objective: J(ρ,u)  (compliance, misfit, etc.)
       │
       ▼
Adjoint: dJ/dρ  (self-adjoint or general Kᵀλ = -∂J/∂u)
       │
       ▼
Filter: weighted average (rmin radius, mesh-independent)
       │
       ▼
Update: OC bisection (topology) or projected gradient (inverse)
       │
       ▼
Convergence → loop
```

## Modules

| Module | Description |
|--------|-------------|
| `q16` | Q16.16 fixed-point with powi, clamp |
| `forward` | 2D Quad4 elasticity, SIMP assembly, CG solver |
| `adjoint` | Compliance & general adjoint sensitivities |
| `filter` | Weighted-average sensitivity filter (rmin) |
| `topology` | SIMP TopOpt with OC update, volume constraint |
| `inverse` | Parameter estimation, Tikhonov, Poisson 1D model |

## Validation (36/36 PASSED)

```
Stage 1:  Q16.16 arithmetic (powi, clamp)       ✓
Stage 2:  Quad4 element stiffness                ✓  (symmetric, 28/28)
Stage 3:  Forward solver                         ✓  (CG converged, 16 iters)
Stage 4:  Adjoint vs FD sensitivity              ✓  (ratio = 0.982)
Stage 5:  Sensitivity filter                     ✓  (smoothing verified)
Stage 6:  Topology optimization                  ✓  (compliance: 168→94)
Stage 7:  Volume constraint                      ✓  (vol = 0.500)
Stage 8:  OC update bounds                       ✓  (solid + void regions)
Stage 9:  Inverse problem (Poisson)              ✓  (κ recovery, J↓ monotone)
Stage 10: Tikhonov regularization                ✓  (reg ≥ unregularized)
Stage 11: Deterministic execution                ✓  (bit-identical)
Stage 12: Convergence diagnostics                ✓  (299/299 steps decrease)
Stage 13: Architecture validation                ✓
```

## Key Properties

- **Deterministic**: Q16.16 inner loop, bit-identical across platforms
- **Self-Adjoint**: Compliance problems exploit Kᵀ = K symmetry → λ = -u
- **General Adjoint**: Framework supports non-self-adjoint objectives
- **Mesh-Independent**: Sensitivity filter prevents checkerboard artifacts
- **ZK-Ready**: Integer arithmetic maps to circuit constraints

## Integration

Builds on top of and integrates with:
- **FEA-QTT**: Structural mechanics (forward solver)
- **CEM-QTT**: Electromagnetics (multi-physics optimization)
- **CFD (HyperTensor)**: Fluid topology optimization
- **FluideLite**: ZK proof framework (verifiable optimization)

## Build

```bash
cargo build --release
cargo test
cargo test --test integration
```

## License

© 2026 Brad McAllister. All rights reserved. PROPRIETARY.
