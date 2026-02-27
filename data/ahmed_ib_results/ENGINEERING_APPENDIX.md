# HyperTensor QTT Engine — Engineering Specification Appendix

**QTT Navier-Stokes Solver: Architecture, Algorithms, and Compliance**

| | |
|---|---|
| **Report ID** | HTR-2026-002-APPENDIX-A |
| **Date** | February 22, 2026 |
| **Author** | Brad Adams, Tigantic Holdings LLC |
| **Revision** | 1.0 |

---

## A.1 Solver Architecture

### A.1.1 Data Flow

```
                    ┌──────────────────────────────────────┐
                    │         QTT State Vector              │
                    │   u = (u_x, u_y, u_z) in TT format   │
                    │   Storage: O(r² · 3n_bits) per field  │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         _rhs(u)                       │
                    │   ω = ∇×u          (QTT curl)        │
                    │   u×ω              (QTT cross)       │
                    │   ν∇²u             (QTT Laplacian)   │
                    │   du/dt = -u×ω + ν∇²u                │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Forward Euler                  │
                    │   u_pred = u + dt · rhs(u)            │
                    │   QTT fused sum (no temporaries)      │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Brinkman  IB                   │
                    │   u = u + u ⊙ (mask_impl − 1)        │
                    │   Correction-based: avoids near-1     │
                    │   catastrophic cancellation            │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Sponge Layer                   │
                    │   u = u + u ⊙ (decay − 1) + compl    │
                    │   Correction-based: localized         │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Adaptive Truncation            │
                    │   QR left-to-right sweep              │
                    │   rSVD right-to-left with rank_profile│
                    │   Bond k capped at rank_profile[k]    │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Energy Safety Valve            │
                    │   If E_new > E_old:                   │
                    │     rescale u by √(E_old/E_new)·0.999 │
                    │   Prevents runaway from truncation     │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │         Convergence Check              │
                    │   ΔE/E < 10⁻⁴ and step > max(50,n/2)│
                    │   Skip clamped steps in convergence   │
                    └──────────────────────────────────────┘
```

### A.1.2 Morton Ordering

The 3D grid of N³ = (2^n)³ cells is mapped to a 1D vector of length 8^n using
Morton (Z-order) indexing. Each QTT site k corresponds to bit level k//3 of
dimension k%3:

- k%3 == 0 → x-bit
- k%3 == 1 → y-bit
- k%3 == 2 → z-bit

This interleaving ensures spatial locality across all three dimensions at every
scale, enabling efficient TT compression of 3D fields with localized features.

### A.1.3 Correction-Based Operators

**Problem**: Standard Hadamard product u ⊙ F where F ≈ 1 almost everywhere
(e.g., implicit Brinkman mask, sponge decay) is catastrophically lossy under
TT truncation. The result u ⊙ F ≈ u requires the Hadamard to preserve the full
freestream to many digits of precision — impossible at low rank.

**Solution**: Reformulate as:

```
u_new = u + u ⊙ (F − 1)
```

where (F − 1) is localized (≈ 0 far from body/boundary). The Hadamard product
u ⊙ (F − 1) is now between a flow field and a sparse correction — well-compressed,
accurate, and stable.

This reformulation is applied to:
- **Brinkman**: F = 1/(1 + dt/η · χ), correction (F − 1) ≈ −1 inside body, 0 elsewhere
- **Sponge**: F = exp(−σ dt), correction (F − 1) < 0 in sponge zone, 0 in interior

## A.2 QTT Operations Specification

### A.2.1 Core Operations

| Operation | Function | Complexity | Triton |
|:----------|:---------|:----------:|:------:|
| Add | `qtt_add_native` | O(r² · n) | — |
| Scale | `qtt_scale_native` | O(r · n) | — |
| Subtract | `qtt_sub_native` | O(r² · n) | — |
| Hadamard | `qtt_hadamard_native` | O(r⁴ · n) | ✓ |
| Inner Product | `qtt_inner_native` | O(r⁴ · n) | ✓ |
| Fused Sum | `qtt_fused_sum` | O(k · r² · n) | — |
| Truncation | `qtt_truncate_sweep` | O(r³ · n) | — |
| MPO Apply | `triton_mpo_apply` | O(r⁴ · n) | ✓ |

### A.2.2 Randomized SVD (rSVD)

The `rsvd_truncate` function is used for all truncation SVDs. Configuration:

| Parameter | Value | Justification |
|:----------|:-----:|:--------------|
| Threshold | 48 | Fire rSVD for all matrices ≤ 48 × N (covers all TT cores) |
| Oversampling | 10 | Standard oversampling for rSVD accuracy |
| Power iterations | 1 | Single power iteration for spectral decay |

At TT rank 48 with 2×2 physical dimensions, the largest SVD encountered is
96 × 96. Full SVD of this size costs O(96³) ≈ 885K ops. rSVD with oversampling 10
costs O(96 × 58 × 10) ≈ 56K ops — a **16× speedup** per SVD call, applied at every
bond of every truncation.

### A.2.3 Adaptive Rank Profile

The `turbulence_rank_profile` function generates a bell-curve rank distribution:

```
rank(k) = base_rank + (peak_rank − base_rank) · sin²(π · k / n_sites)
```

| Parameter | 128³ | 256³ | 512³ |
|:----------|:----:|:----:|:----:|
| n_sites | 21 | 24 | 27 |
| n_bonds | 20 | 23 | 26 |
| base_rank | 24 | 24 | 24 |
| peak_rank | 48 | 48 | 48 |

The physical motivation: the largest (small k) and smallest (large k) scales
carry less turbulent energy and require lower rank. Mid-scale bonds (inertial range)
carry the k^{-5/3} cascade and receive peak rank allocation.

### A.2.4 Triton Kernel Dispatch

Hadamard and inner product operations dispatch to Triton kernels when:
1. Input tensors are on CUDA device
2. `_HAS_TRITON_KERNELS` flag is True (Triton importable)

Kernels use power-of-2 block sizes for L2 cache optimization:

```python
BLOCK_M = triton.next_power_of_2(R_a)
BLOCK_N = triton.next_power_of_2(R_b)
```

Fallback path uses `torch.einsum` for CPU or when Triton is unavailable.

## A.3 Separable Field Construction

### A.3.1 Algorithm

For fields f(x,y,z) = g(x) (varying only along x, constant in y and z):

1. Compute 1D QTT of g(x) using TT-SVD: n_bits cores of shape (r_in, 2, r_out)
2. Interleave identity cores at y-bit and z-bit positions:
   - z-site (k%3==0): I_{r×r} expanded over bit dimension → shape (r, 2, r)
   - y-site (k%3==1): I_{r×r} expanded over bit dimension → shape (r, 2, r)
   - x-site (k%3==2): 1D QTT core → shape (r_in, 2, r_out)
3. Result: 3n_bits total TT sites; rank profile matches 1D QTT

### A.3.2 Memory Savings

| Grid | Dense 3D Array | 1D Array | Ratio |
|:----:|:--------------:|:--------:|:-----:|
| 128³ | 8.4 MB | 512 B | 16,384× |
| 256³ | 67.1 MB | 1.0 KB | 67,109× |
| 512³ | 536.9 MB | 2.0 KB | 268,435× |
| 1024³ | 4.3 GB | 4.0 KB | 1,073,742× |

At 1024³, this avoids allocating 4.3 GB per separable field — three sponge fields
would require 12.9 GB of dense init memory eliminated.

## A.4 Convergence Diagnostics

### A.4.1 Energy Clamp Logic

```python
if E_new > E_prev:
    scale = sqrt(E_prev / E_new) * 0.999  # 0.1% dissipative
    u = scale * u
    clamped = True
```

The energy safety valve ensures monotonic energy decay. The 0.999 factor introduces
0.1% artificial dissipation per clamp event to prevent oscillation at the clamp
boundary. Clamped steps are excluded from convergence criterion evaluation.

### A.4.2 Convergence Gate

```python
converged = (ΔE/E < tol) and (step > max(50, n_steps // 2)) and (not clamped)
```

The step gate prevents false convergence triggers during:
- Early transient phase (step < 50)
- First half of simulation (step < n_steps/2)
- Clamped steps (ΔE artificially zero)

## A.5 File Manifest

### Source Files Modified

| File | Lines | Modifications |
|:-----|:-----:|:-------------|
| `tensornet/cfd/qtt_native_ops.py` | ~1,109 | rSVD threshold 512→48; Triton dispatch in hadamard/inner; rank_profile param in truncation |
| `scripts/ahmed_body_ib_solver.py` | ~916 | Adaptive rank profile; dead dense removal; separable sponge; correction-based operators |
| `scripts/ahmed_body_spectrum.py` | ~410 | Reconstructs mask from SDF instead of stored dense |
| `scripts/gauntlet_vs_nvidia.py` | ~380 | Multi-resolution benchmark harness |

### Source Files Unmodified (Dependencies)

| File | Lines | Role |
|:-----|:-----:|:-----|
| `tensornet/cfd/ns3d_native.py` | 1,402 | Native QTT NS solver, derivatives, TT-SVD |
| `tensornet/cfd/triton_qtt_kernels.py` | 433 | Triton GPU kernels (MPO, Hadamard, inner) |
| `tensornet/cfd/kolmogorov_spectrum.py` | ~200 | Energy spectrum analysis |

---

*Appendix HTR-2026-002-APPENDIX-A*
*HyperTensor QTT Engine v2.0 — Tigantic Holdings LLC*
*February 22, 2026*
