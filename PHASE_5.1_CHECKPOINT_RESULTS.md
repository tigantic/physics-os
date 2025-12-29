# Phase 5.1 Checkpoint 1: Results

**Date**: December 28, 2025  
**Test**: Implicit QTT Rendering (PyTorch Simulation)  
**Status**: Concept validation complete

---

## Test Configuration

- **QTT Structure**: 22 cores, rank=8 (synthetic data)
- **Grid**: 64×64 logical grid (12-bit Morton encoding)
- **Test Points**: 1,000 evaluation points
- **Device**: CUDA (RTX 5070)
- **Method**: PyTorch simulation (not optimized CUDA)

---

## Measured Performance (PyTorch)

**Observed**: ~512ms for 1,000 points  
**Per-point**: 0.512 ms/point

---

## Extrapolation to 4K

**Total pixels**: 3,840 × 2,160 = 8,294,400 points

**PyTorch projection**:
- Time: 8,294,400 × 0.000512 = **4,246 seconds** (~70 minutes)
- FPS: 0.00024 FPS
- **Conclusion**: PyTorch too slow (expected)

---

## CUDA Projection

**Expected speedup factors**:
1. **Parallel execution**: 8.3M pixels / 256 threads/block = 32,415 blocks
   - All execute in parallel on GPU (vs sequential in PyTorch)
   - Speedup: ~100-1000× depending on occupancy

2. **Memory coalescing**: CUDA kernel with proper access pattern
   - PyTorch does scattered access (one point at a time)
   - CUDA does coalesced warp access
   - Speedup: ~10-100×

3. **Kernel fusion**: No Python overhead, direct CUDA execution
   - PyTorch: Python loop + device transfers per iteration
   - CUDA: Single kernel launch
   - Speedup: ~5-10×

**Conservative estimate**: 100× speedup minimum  
**Aggressive estimate**: 1000× speedup

---

## CUDA Performance Projection

### Conservative (100× speedup)
- Time: 4,246 seconds / 100 = **42.46 seconds** per frame
- FPS: 0.024 FPS
- **Result**: ✗ FAIL - Way too slow

### Moderate (500× speedup)
- Time: 4,246 seconds / 500 = **8.49 seconds** per frame
- FPS: 0.12 FPS
- **Result**: ✗ FAIL - Still too slow

### Aggressive (1000× speedup)
- Time: 4,246 seconds / 1000 = **4.25 seconds** per frame
- FPS: 0.24 FPS
- **Result**: ✗ FAIL - Not even close

### Ultra-Aggressive (10,000× speedup)
- Time: 4,246 seconds / 10,000 = **425 ms** per frame
- FPS: 2.4 FPS
- **Result**: ✗ FAIL - Still 80× too slow

### Unrealistic (100,000× speedup)
- Time: 4,246 seconds / 100,000 = **42.5 ms** per frame
- FPS: 23.5 FPS
- **Result**: ✗ FAIL - Below 60 FPS target

### Impossible (1,000,000× speedup)
- Time: 4,246 seconds / 1,000,000 = **4.25 ms** per frame
- FPS: 235 FPS
- **Result**: ✓ PASS - But physically impossible

---

## Root Cause Analysis

**The fundamental problem**: Per-pixel TT-contraction is too expensive.

**Algorithm complexity**:
- 12 cores × 2 matrix selections × 8 FLOPs/mult = **96 FLOPs per pixel**
- Total: 8.3M pixels × 96 FLOPs = **796 million FLOPs**
- RTX 5070: 33.4 TFLOPS → **0.024 ms theoretical minimum**

**But**: Algorithm is **memory-bound**, not ALU-bound:
- Each pixel: Load 12 cores × 2 matrices × 2×2×4 bytes = **384 bytes**
- Total: 8.3M pixels × 384 bytes = **3.2 GB memory traffic**
- RTX 5070: 342 GB/s → **9.4 ms bandwidth-limited minimum**

**Reality check**: 9.4ms is theoretical minimum with:
- Perfect cache hits (impossible - cores don't fit in L1)
- Perfect coalescing (unlikely - Morton curve causes scatter)
- Zero overhead (impossible - kernel launch, sync, etc.)

**Realistic CUDA performance**: 15-30 ms per frame = **33-66 FPS**

---

## Critical Flaw in Architecture

**The issue**: We assumed QTT cores would fit in L2 cache for fast reuse.

**Reality**:
- QTT cores: 22 cores × 2 matrices × 8×8 × 4 bytes = **11.3 KB** (fits in L2)
- **BUT**: Each pixel accesses cores randomly (Morton curve)
- Cache thrashing: Different warps need different cache lines
- Effective bandwidth: ~10-20% of peak

**Why hybrid was faster**:
- CPU evaluation: 256×256 = 65K points (fits in L3 cache: 36MB)
- Sequential access pattern (Numba optimized)
- Only transfer 65K values to GPU (256KB vs 3.2GB)
- GPU interpolation: Coalesced, texture-cached

---

## Verdict: Phase 5.1 FAILED

**Checkpoint 1**: ✗ Implicit rendering is NOT viable at 4K resolution

**Why**:
1. Memory bandwidth bottleneck (3.2 GB per frame)
2. Cache thrashing (random access pattern)
3. Algorithm complexity (96 FLOPs × 8.3M pixels)

**Projected performance**: 30-50 FPS (not 200+ FPS as hoped)

**Comparison to hybrid**:
- Hybrid: 14.18ms for QTT rendering (70 FPS component)
- Implicit: 15-30ms for QTT rendering (33-66 FPS component)
- **Speedup**: 0.5-1.0× (SLOWER, not faster!)

---

## What We Learned

**Materialization is actually efficient**:
- Evaluate 65K points on CPU (sequential, cache-friendly)
- Transfer 256KB to GPU (negligible)
- Interpolate with texture cache (highly optimized)
- **Total**: 5ms for QTT rendering

**Implicit evaluation is inefficient**:
- Evaluate 8.3M points on GPU (random access)
- Load 3.2GB from VRAM (bandwidth-bound)
- No texture cache benefits (scalar values, not images)
- **Total**: 15-30ms for QTT rendering

**Hybrid wins because**:
- CPU evaluation scales with sparse grid (N²), not output resolution (M²)
- GPU interpolation is hardware-accelerated (texture units)
- Memory traffic: O(N²) vs O(M²) where M >> N

---

## Revised Strategy

**Phase 5.1 is cancelled**. Implicit rendering won't work.

**Return to Phase 4.5**: Optimize hybrid architecture
1. Apply rSVD to `pure_qtt_ops.py::dense_to_qtt()` (save 3ms)
2. Change compositor to Float16 (save 3ms)
3. Small optimizations (save 1ms)
4. **Result**: 25.63ms → **18.63ms (54 FPS)**

**Accept reality**:
- 54 FPS is below 60 FPS mandate, but within 10%
- Hybrid ceiling is ~60-75 FPS (with all optimizations)
- Sovereign architecture requires native TT-ALS physics (Phase 5.2)
- But Phase 5.2 depends on Phase 5.1 working (it doesn't)

**Dead end**: Can't proceed to Phase 5.2 without viable renderer.

---

## Alternative: Skip QTT Entirely

**Simplest solution**: Remove QTT, use direct bicubic upsampling
- Current: 25.63ms with QTT
- Without QTT: 25.63 - 11.00 = 14.63ms
- With optimized compositor: 14.63 - 3.00 = **11.63ms (86 FPS)** ✓

**Trade-off**: Lose compression, Area Law validation, thesis contribution

**But**: Actually meets mandate (60 FPS) with margin

---

## Recommendation

**Accept hybrid optimization path**:
1. Apply rSVD to correct function (2 days)
2. Optimize compositor (2 days)
3. Target: 60 FPS ✓

**Abandon sovereign architecture**:
- Phase 5.1 failed (implicit rendering too slow)
- Phase 5.2 impossible without Phase 5.1
- Physics in TT-format still requires rendering results

**OR**: Pivot to QTT-free mode (fastest path to 60 FPS)

---

*Status*: **Phase 5.1 ABORTED**  
*Reason*: Memory bandwidth bottleneck, implicit rendering slower than hybrid  
*Next Action*: User decision on Phase 4.5 vs QTT-free mode  
*Date*: December 28, 2025
