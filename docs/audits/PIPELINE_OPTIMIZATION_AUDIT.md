# Pipeline Optimization Audit
# Generated: December 28, 2025
# Status: CUDA kernels deployed, identifying remaining bottlenecks

## CRITICAL PATH LOOPS (Hot Spots)

### 1. **ontic/mpo/operators.py** ✓ OPTIMIZED
**Line 124**: Loop over 12 QTT cores for MPO-TT contraction
```python
for i, (mpo_core, qtt_core) in enumerate(zip(self.laplacian_cores, qtt_cores)):
    contracted = torch.einsum('ijkl,mjn->imknl', mpo_core, qtt_core, optimize='optimal')
```
**Status**: ✓ Now uses CUDA kernel (laplacian_cuda.py)
**Performance**: 128ms → <1ms (100×+ speedup)

**Line 152**: Lazy compression loop
```python
for idx, core in needs_compression:
    new_cores[idx] = _compress_core(core, max_rank)
```
**Status**: ✓ Batched compression applied
**Performance**: Reduced overhead from eager to lazy strategy

---

### 2. **ontic/gateway/orbital_command.py** ⚠ NEEDS VECTORIZATION
**Lines 208, 213**: Grid generation loops
```python
for lat in range(-90, 91, grid_spacing_lat):  # 181 iterations
    ...
for lon in range(-180, 181, grid_spacing_lon):  # 361 iterations
```
**Current**: 181 × 361 = 65,341 iterations (0.46ms)
**Target**: Vectorized meshgrid (0.10ms, 4.6× speedup)
**Action**: Replace with:
```python
lats = torch.arange(-90, 91, grid_spacing_lat, device=device)
lons = torch.arange(-180, 181, grid_spacing_lon, device=device)
lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
```

**Lines 373, 381**: Cartesian grid loops
```python
for x in range(0, self.width, grid_spacing):  # ~192 iterations
    ...
for y in range(0, self.height, grid_spacing):  # ~108 iterations
```
**Same issue**: Should use torch.meshgrid

---

### 3. **ontic/gateway/onion_renderer.py** ⚠ LAYER LOOPS
**Line 316**: Layer blending loop
```python
for layer in enabled[1:]:  # 5 layers
    # Alpha blending operation
```
**Current**: Sequential CPU blending
**Target**: Fused CUDA kernel (marked as ✓ in todo but not implemented)
**Performance**: Could save ~2-3ms with proper GPU kernel

---

### 4. **ontic/quantum/cpu_qtt_evaluator.py** ✓ REPLACED WITH GPU
**Lines 139-143**: Nested contraction loops
```python
for i in range(r_left):
    for j in range(r_right):
        result[j] += prev[i] * cores_flat[...]
```
**Status**: ✓ Replaced by GPU QTT eval kernel
**Performance**: 164ms → 1.46ms (112× speedup)

---

## SVD → rSVD CONVERSION AUDIT

### ✓ ALREADY USING rSVD (torch.svd_lowrank)

1. **ontic/core/decompositions.py** ✓
   - Line 69: `torch.svd_lowrank(A, q=target_rank, niter=2)`
   - Line 166: `torch.svd_lowrank(A, q=min_dim, niter=2)`
   - Line 182: `torch.svd_lowrank(A, q=min_dim, niter=2)`

2. **ontic/cfd/qtt.py** ✓
   - Line 137: `torch.svd_lowrank(current, q=target_rank, niter=2)`

3. **ontic/mpo/operators.py** ✓
   - Line 178: `torch.svd_lowrank(mat_left, q=max_rank, niter=1)`
   - Line 188: `torch.svd_lowrank(mat_right, q=max_rank, niter=1)`

### ⚠ NEEDS CONVERSION TO rSVD (Using torch.linalg.svd)

#### HIGH PRIORITY (On critical path):

1. **ontic/mpo/laplacian_cuda.py** (Lines 143, 158)
```python
# BEFORE:
U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

# AFTER:
U, S, Vh = torch.svd_lowrank(mat, q=min(max_rank, min(mat.shape)), niter=1)
```
**Impact**: Compression happens every frame, causes 1.5s spikes

2. **ontic/cfd/qtt_2d_shift.py** (Line 299)
```python
# Shift operations (cached but still used on init)
U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
```

3. **ontic/cfd/qtt_2d_shift_native.py** (Line 234)
```python
# Similar shift SVD
U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
```

#### MEDIUM PRIORITY (Off critical path but frequent):

4. **ontic/cfd/pure_qtt_ops.py** (Lines 282, 521, 750)
5. **ontic/distributed_tn/parallel_tebd.py** (Line 191)
6. **ontic/cfd/adaptive_tt.py** (Lines 157, 480)
7. **ontic/algorithms/tdvp.py** (Line 291)
8. **ontic/cfd/tt_cfd.py** (Lines 672, 706)
9. **ontic/cfd/nd_shift_mpo.py** (Line 238)

#### LOW PRIORITY (Offline/initialization only):

10. **ontic/digital_twin/reduced_order.py** (Lines 210, 293, 622) - POD computation
11. **ontic/adaptive/compression.py** (Lines 149, 282, 441) - Already has RandomizedSVD class!
12. **ontic/substrate/morton_ops.py** (Line 410)
13. **ontic/substrate/field.py** (Line 668)

---

## IMMEDIATE ACTION ITEMS

### Priority 1: Replace full SVD with rSVD in hot path
**Files to fix**:
1. `ontic/mpo/laplacian_cuda.py` (lines 143, 158)
2. `ontic/cfd/qtt_2d_shift.py` (line 299) 
3. `ontic/cfd/qtt_2d_shift_native.py` (line 234)

**Expected impact**: Eliminate 1.5s compression spikes → stable <100ms frames

### Priority 2: Vectorize grid generation
**File**: `ontic/gateway/orbital_command.py` (lines 208-213, 373-381)
**Expected impact**: 0.46ms → 0.10ms (4.6× speedup on grid rendering)

### Priority 3: Implement fused blend kernel
**File**: `ontic/gateway/onion_renderer.py` (line 316)
**Expected impact**: ~2-3ms savings on compositor

---

## PERFORMANCE SUMMARY

### Current State (After CUDA kernels):
- **Frame time**: 26-67ms (15-38 FPS)
- **Bottlenecks**:
  1. Colormap: 24ms ⚠ (can optimize further)
  2. Compression spikes: 1570ms every ~3rd frame ⚠⚠⚠
  3. Grid generation: 0.46ms (minor)

### Post-rSVD Conversion Projection:
- **Compression spikes**: 1570ms → <100ms (15× improvement)
- **Stable frame time**: <30ms (>33 FPS sustained)
- **Peak frame time**: <50ms (>20 FPS worst case)

### Target (60 FPS):
- **Remaining gap**: ~10-13ms after rSVD fixes
- **Final optimizations needed**:
  1. Colormap caching/optimization (-10ms)
  2. Vectorized grids (-0.36ms)
  3. Fused blend kernel (-2ms)

**Total projected**: 26ms - 12ms = **14ms (71 FPS)** ✓✓✓

---

## ARCHITECTURAL NOTES

### Why rSVD matters:
- **Full SVD**: O(min(m,n)³) - scales cubically
- **rSVD**: O(mn × rank) - linear in matrix size
- **For 64×64 cores**: Full=262K ops, rSVD=4K ops (65× fewer)

### Current rSVD usage:
- **decompositions.py**: niter=2 (accurate)
- **mpo/operators.py**: niter=1 (fast, MPO context allows less accuracy)
- **Need**: Apply niter=1 everywhere on critical path

### Loop vectorization:
- **Python loops**: 100-1000× slower than vectorized ops
- **Meshgrid**: Single GPU kernel vs thousands of Python iterations
- **Critical for**: Grid generation, contraction batching

---

## VALIDATION CHECKLIST

After applying fixes:
- [ ] Run 100-frame benchmark (check for spikes)
- [ ] Verify rank stability (should stay ≤8)
- [ ] Check accuracy (<1% error vs full SVD)
- [ ] Profile with Nsight (GPU utilization should be 80%+)
- [ ] Measure sustained FPS (target: >60 FPS for 1000 frames)
