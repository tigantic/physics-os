# HVAC CFD Session State
## DO NOT DELETE - Read this before making changes

---
## 🚨 MANDATORY PRE-FLIGHT CHECK 🚨

**BEFORE writing ANY code, answer these questions:**

1. Does my code call `qtt_3d_to_dense()` or `.to_dense()` in a loop? → **FORBIDDEN**
2. Does my code call `dense_to_qtt_3d()` in a loop? → **FORBIDDEN**  
3. Does my code use Python `for` loops over tensor elements? → **FORBIDDEN**
4. Does my code require N×N matrices where N > 1000? → **FORBIDDEN**
5. Am I taking a shortcut that "works for small grids"? → **STOP. It won't scale.**

**If ANY answer is YES → STOP and redesign before proceeding.**

---
## 🔴 FORBIDDEN PATTERNS (2026-01-07)

These patterns have caused repeated failures. DO NOT USE:

```python
# ❌ FORBIDDEN: Dense conversion in time loop
for step in range(n_steps):
    u_dense = qtt_3d_to_dense(u_qtt)  # VIOLATION: O(N) every step
    ...
    u_qtt = dense_to_qtt_3d(u_new)    # VIOLATION: O(N) every step

# ❌ FORBIDDEN: Upwind advection requiring dense velocity
adv = ops.upwind_advection(f_qtt, u_dense, v_dense, w_dense)  # Needs dense!

# ❌ FORBIDDEN: Applying BCs via dense slicing
u_dense[0, :, :] = U_inlet  # Requires full decompression
```

**CORRECT PATTERNS:**

```python
# ✓ ALLOWED: Pure QTT operations
grad_x, grad_y, grad_z = ops.gradient(f_qtt)  # Returns QTT
lap = ops.laplacian_3d(f_qtt)                  # Returns QTT
result = ops.add(a_qtt, b_qtt)                 # Returns QTT

# ✓ ALLOWED: Dense ONLY at final output for visualization
final_u = qtt_3d_to_dense(u_qtt)  # Once, at end, for plotting
```

---
### Critical Decisions Made

1. **USE QTT, NOT DENSE** (2026-01-07)
   - User explicitly said: "You've been using 'DENSE'????? That was literally the first thing I had you transcribe to the source of truth"
   - Source of Truth §1: "Dense is Anti-QTT"
   - Dense at 1024³ = 21.5 GB
   - QTT at 1024³ = ~0.4 MB (50,000× compression)

2. **TIER 1 IS 3D, NOT 2D** (2026-01-07)
   - Target: 1 billion cells (1024³)
   - The 2D projection_solver.py is OBSOLETE for this task
   - Use: `ontic/cfd/qtt_ns_3d.py`

3. **CUDA KERNELS HANG** (2026-01-07)
   - `nd_shift_mpo` tries to compile CUDA kernel and hangs on CPU
   - Solution: Use `shift_mpo` from `pure_qtt_ops.py` instead
   - Already fixed in qtt_ns_3d.py

4. **torch.svd_lowrank returns V, not Vh** (2026-01-07)
   - Fixed 3 locations in pure_qtt_ops.py
   - Created `svd_truncated()` wrapper as single source of truth
   - Regression tests in `tests/test_qtt_regression.py`

### Current State

- **Working solver**: `ontic/cfd/qtt_ns_3d.py`
- **Last successful run**: 128×64×64 grid (524k cells), 20 steps, 0.32 MB memory, 6.4s/step
- **Regression tests**: 12/12 passing

### Next Steps (NOT completed yet)

1. Scale up to larger grid (128³, 256³, etc.)
2. Validate against Nielsen experimental data
3. Achieve <10% RMS error

### Files NOT to use

- `ontic/hvac/projection_solver.py` - 2D dense, violates Source of Truth
- `ontic/cfd/nd_shift_mpo.py` - CUDA kernel hang issue

---
## 🔴 FAILURE LOG (Learn from these)

| Date | Failure | Root Cause | Lesson |
|------|---------|------------|--------|
| 2026-01-07 | Used dense projection solver | Didn't read Source of Truth | Read §1 FIRST |
| 2026-01-07 | Tier 1 sim exploded (1153 m/s) | `qtt_3d_to_dense()` in loop, upwind needs dense | Pure QTT advection required |
| 2026-01-07 | "Lazy path" taken repeatedly | Optimized for "works" over "correct" | Pre-flight check MANDATORY |

**Pattern**: I default to dense because it's "easier" → It always fails at scale → Time wasted

---
## 🟢 WHAT STILL NEEDS TO BE BUILT

~~1. **Pure QTT advection** - central difference, no dense velocity lookup~~ ✓ DONE
~~2. **QTT boundary conditions** - penalty method or MPO masking, no dense slicing~~ ✓ DONE
~~3. **Hyperviscosity stabilization** - ν_h∇⁴ term for stability without upwinding~~ ✓ DONE

**ALL PURE QTT COMPONENTS COMPLETED (2026-01-07)**

### Tier 1 Results (Pure QTT)
- Grid: 32×16×16 = 8,192 cells
- 200 steps, 0.2s simulation time
- Max velocity: 1.99x inlet (STABLE)
- Memory: 34.9 KB (11x compression vs dense)
- Zero dense conversions in time loop
