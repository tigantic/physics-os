# QTT Batched Operations — Integration Guide

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `triton_qtt3d.py` | 398 | Triton 3D kernels: residual absorption, MPO apply, inner product |
| `qtt_batched_ops.py` | 807 | Batched truncation sweep, phase-level physics operations |
| `qtt_batched_patch.py` | 267 | Drop-in integration for `TurboNS3DSolver` |
| `benchmark_batched.py` | 493 | Correctness validation + performance benchmarks |

## Installation

Extract into your `ontic/cfd/` directory alongside existing files:

```bash
cd physics-os-main
tar xzf qtt_batched_ops.tar.gz
# Creates: ontic/cfd/triton_qtt3d.py, qtt_batched_ops.py, etc.
```

## Quick Start

### Option A: Monkey-patch (zero changes to existing code)

```python
from ontic.cfd.ns3d_turbo import TurboNS3DConfig, TurboNS3DSolver
from ontic.cfd.qtt_batched_patch import patch_solver

config = TurboNS3DConfig(
    n_bits=5, nu=0.01, dt=0.02,
    adaptive_rank=False,  # IMPORTANT: must be False for batched path
    max_rank=32, device='cuda',
)
solver = TurboNS3DSolver(config)
solver.initialize_taylor_green()

# Patch hot path — replaces step(), _compute_rhs(), _truncate_terms()
patch_solver(solver)

# Now runs with batched operations
for i in range(100):
    diag = solver.step()
```

### Option B: Standalone (no dependency on existing solver)

```python
from ontic.cfd.qtt_batched_ops import (
    rk2_step_batched,
    batched_diagnostics,
)

# u, omega are lists of 3 QTTs each (List[List[Tensor]])
# shift_plus, shift_minus are lists of 3 MPOs (List[List[Tensor]])

u_new, omega_new = rk2_step_batched(
    u, omega, nu=0.01, dt=0.02, dx=2*np.pi/32,
    shift_plus=shift_plus, shift_minus=shift_minus,
    max_rank=32,
)
diag = batched_diagnostics(u_new, omega_new)
```

## What Changed

### Bug Fix: `_truncate_terms` dispatch

**Problem:** When `adaptive_rank=False`, the solver was still calling
`turbo_linear_combination_adaptive`, which uses `torch.linalg.svd` directly
(bypassing rSVD) for rank estimation it never uses.

**Result:** 1008 full SVDs instead of 840 rSVDs. 12x slower than necessary.

**Fix:** `_truncate_terms_fixed` routes to `batched_linear_combination` when
`adaptive_rank=False`.

### Optimization 1: Batched SVD across fields

**Before:** 6 independent truncation sweeps, each with 15 individual SVD calls = 90 kernel launches.

**After:** 1 batched truncation sweep with 15 batched SVD calls = 15 kernel launches. Each batched SVD processes all 6 fields in one `torch.linalg.svd` call on a (6, M, N) tensor.

**Speedup:** 50-60x for small matrices (2×8, 4×16), 1.5-2x for large matrices (128×52). Aggregate: 3-5x on SVD time.

### Optimization 2: Phase-level truncation

**Before (per cross product):**
```
hadamard(uy, wz) → truncate (15 SVDs)
hadamard(uz, wy) → truncate (15 SVDs)
sub(term1, term2) → truncate (15 SVDs)
... repeat for y, z components ...
Total: 9 truncation sweeps = 135 SVDs
```

**After:**
```
hadamard_raw(uy, wz)    # no truncation, rank grows to r²
hadamard_raw(uz, wy)    # no truncation
sub_raw(term1, term2)   # no truncation, rank = 2r²
... all 3 components raw ...
batched_truncation_sweep([cx, cy, cz])  # ONE batched truncation = 15 SVDs
```

**SVD reduction for full RHS:**

| Operation | Before (SVDs) | After (batched SVDs) |
|-----------|--------------|---------------------|
| Cross product | 135 | 15 |
| Curl | 270 | 15 |
| Laplacian (vector) | 315 | 15 |
| Final combine | 45 | 15 |
| **Total per RHS** | **765** | **60** |
| **Total per step (RK2)** | **1530** | **120** |

Each "batched SVD" processes 3 fields in one kernel call, so effective work is 120 × 3 = 360 SVDs. But kernel launch overhead drops from 1530 to 120 launches — a **12.75x reduction**.

### Optimization 3: Triton 3D kernels

The residual absorption `out[i,s,k] = Σ_j R[i,j] * core[j,s,k]` is a 3D contraction
called ~840 times per step. The Triton kernel:
- Uses a 3D grid: `(ceil(M/BM), D, ceil(N/BN))`
- Handles the physical dimension d=2 as a grid axis (2 program instances)
- Falls back to einsum for tensors with < 512 total flops (faster due to launch overhead)
- Eliminates Python einsum dispatch (string parsing, shape checking)

## Expected Performance

| Configuration | Before | After | Speedup |
|--------------|--------|-------|---------|
| 32³, rank 32, full NS | 1800ms | ~400ms | ~4.5x |
| 32³, rank 48, full NS | 2200ms (fixed) / 26000ms (adaptive) | ~600ms | ~3.7x / ~43x |
| 32³, rank 32, diffusion only | 346ms | ~120ms | ~2.9x |
| 64³, rank 48 | ~550ms (diffusion) | ~200ms | ~2.8x |

The dramatic 43x improvement for rank 48 is from fixing the adaptive_rank routing bug.

## Running the Benchmark

```bash
cd physics-os-main
PYTHONPATH="$PWD:$PYTHONPATH" python3 ontic/cfd/benchmark_batched.py
```

This runs 6 tests:
1. **Batched SVD speedup** — measures raw SVD batching gain per matrix size
2. **QTT operation correctness** — verifies add, hadamard, norm, inner product
3. **Batched truncation performance** — compares 6-field batched vs individual
4. **Triton kernel performance** — benchmarks 3D residual absorption
5. **Phase-level cross product** — compares sequential vs batched cross product
6. **Full solver integration** — end-to-end step time comparison

## Architecture Notes

### Data format
QTTs are `List[torch.Tensor]` where each tensor has shape `(r_left, 2, r_right)`.
The physical dimension is always 2 (binary). Bond dimensions vary per site.

### Memory layout
The batched SVD pads all fields to common (M_max, N_max) before batching.
For 6 fields with similar ranks, padding waste is minimal (<10%).

### Triton fallback
All Triton kernels have pure-PyTorch fallbacks via `torch.einsum`.
If Triton is unavailable (CPU, older GPU, compilation issues), everything
still works at slightly lower performance.

### Phase accumulation memory
Raw (untruncated) operations create high-rank intermediates:
- Hadamard: rank r → r² (e.g., 32 → 1024)
- Addition: rank r + r' (e.g., 1024 + 1024 = 2048)
- Linear combination of 7 terms: rank 7 × r (e.g., 7 × 1024 = 7168)

The final batched truncation compresses everything back to max_rank.
Peak memory per component: O(L × max_intermediate_rank² × d) where d=2.
For L=15, intermediate rank ~2048: peak ~120MB per component.
With 3 components in a batch: ~360MB. Fits comfortably on any modern GPU.

If memory is tight, set a lower intermediate cap by truncating after
each phase instead of accumulating all the way to the end.
