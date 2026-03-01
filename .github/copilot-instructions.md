## Code Quality

Code to production grade:
- No shortcuts
- No mocks
- No demos
- No placeholders
- No stubs
- No TODO comments as substitutes for implementation
- No "example" or "simplified" versions
- Complete error handling
- Complete type hints
- If you can't implement it fully, say so — don't fake it

## QTT Laws — Non-Negotiable

These rules govern ALL code that touches the QTT runtime, tensors,
operators, and kernels.  Violations require an **explicit written
exception** in the code with a rationale (why, what it costs, and
what would fix it).

### 1. QTT Stays Native
Tensors remain in TT-core format on GPU at all times.  No CPU
fallback in the execution loop.  `to_cpu()` is permitted ONLY for
post-execution reporting/sanitization — NEVER inside the timestep
dispatch.

### 2. SVD = rSVD
All rank truncation uses randomized SVD (`rsvd_native`).  Full
`torch.linalg.svd` / `np.linalg.svd` is permitted ONLY when
`min(m, n) ≤ 4` (exact solution is cheaper than randomized setup).
If you reach for full SVD on anything larger, you need an exception.

### 3. Python Loops → Triton/CUDA Kernels
No Python-level `for` loops over tensor elements, grid points, or
physical modes in the hot path.  Core-level sweeps (QR, rSVD) are
inherently sequential and acceptable, but per-mode contractions
(`n_k = 2` inner loops) must be batched or fused into a single
kernel launch.

### 4. Higher Scale = Higher Compression = Lower Rank
Rank is **adaptive**, never fixed.  Use `GPURankGovernor.get_effective_rank()`
which scales rank with problem size.  Passing a raw `max_rank`
ceiling to any solver or operator without adaptive scaling is a
violation.

### 5. Decompression Kills QTT
`to_dense()` must NEVER appear in any execution path.  It exists
solely for external diagnostics.  Any call to `to_dense()` inside
the VM, operators, or kernels is an automatic rejection.

### 6. Dense Is a Killer
No dense matrix materialization in the compute path.  This includes
dense Kronecker products, dense field reconstruction, or dense
intermediate results.  Everything stays in TT-core format.

### 7. Triton Kernels — L2 Cache Optimized
All Triton kernels must tile their access patterns for L2/SRAM
locality.  Nested serial loops over bond dimensions without blocking
are a violation.  Use shared memory, tiled loads, and batched
operations.

### 8. Adaptive Rank — Not Fixed
Every function that accepts `max_rank` must receive an
adaptively-computed value from `GPURankGovernor`, not a hardcoded
constant.  Default parameter values are acceptable for API
signatures, but callers in the hot path must always pass the
adaptive value.

### Known Violations (tracked for remediation)
- **V-01 RESOLVED**: `LAPLACE_SOLVE` opcode now uses `gpu_poisson_solve()`
  — GPU-native CG with adaptive rank. No CPU fallback.
- **V-02–V-03**: `qtt_dot_native`, `qtt_hadamard_native` use Python
  loops over physical modes.  Fix: fused batched contraction.
- **V-04**: `_tt_core_contract_kernel` has untiled nested loops.
  Fix: tiled SRAM access.
- **V-08**: `gpu_mpo_apply` Python loop over cores.  Fix: batched
  einsum or Triton kernel.
- **V-09 RESOLVED**: Poisson solver now receives adaptive rank via
  `governor.get_effective_rank()`.

### Exception Template
When a rule must be violated, add this comment block:
```python
# QTT-EXCEPTION: Rule <N> — <rule name>
# Why: <one-sentence justification>
# Cost: <what performance/correctness penalty this incurs>
# Fix: <what would eliminate the exception>
```