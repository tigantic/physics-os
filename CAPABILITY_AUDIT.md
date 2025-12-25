# HyperTensor Capability Audit

**Date:** December 25, 2025  
**Version:** 1.0  
**Purpose:** Honest assessment of claims, capabilities, and gaps  
**Principle:** No bullshit. No Potemkin villages. Only provable truth.

### Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 25, 2025 | Initial comprehensive audit |
| — | — | Language decision: Python + PyTorch + Rust |
| — | — | TCI sampling strategy with MaxVol |
| — | — | Batched flux architecture with DLPack |
| — | — | Neighbor index trap identified and resolved |
| — | — | Sound speed sqrt bug documented |
| — | — | Rank explosion mitigation strategy |
| — | — | Complete testing and rollback strategy |

---

## Language Decision

**Stack:** Python (PyTorch) + Rust (TCI Core)

| Layer | Language | Rationale |
|-------|----------|-----------|
| **Orchestration** | Python | 2000+ lines exist, ecosystem, rapid iteration |
| **Tensor Ops** | PyTorch | GPU-native, batching, existing codebase |
| **TCI Algorithm** | Rust (PyO3) | Compiled loops, safe during algo dev, no GIL |

**Why not pure Python?** TCI pivot selection is O(r² × log N) loop iterations per flux.
At r=32, N=2²⁰, that's ~20 million Python loop iterations per timestep. Unacceptable.

**Why Rust over C++?** Algorithm is not yet stable. Rust's memory safety catches bugs
at compile time that C++ hides until runtime. PyO3 bindings are cleaner than pybind11.

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│  Python (Orchestration)                                 │
│  - Calls Rust for index/pivot work                      │
│  - Calls PyTorch for GPU tensor ops                     │
└────────────────┬──────────────────────┬─────────────────┘
                 │                      │
    ┌────────────▼────────────┐   ┌─────▼─────────────────┐
    │  Rust TCI Core (PyO3)   │   │  PyTorch (GPU)        │
    │  ─────────────────────  │   │  ──────────────────── │
    │  • pivot_selection()    │   │  • flux_batch()       │
    │  • index_arithmetic()   │   │  • tensor_contract()  │
    │  • matrix_skeleton()    │   │  • qtt_operations()   │
    │  • cross_approx()       │   │                       │
    └─────────────────────────┘   └───────────────────────┘
```

---

## Executive Summary

HyperTensor aims to solve PDEs using Tensor-Train (TT) and Quantized Tensor-Train (QTT) representations, achieving logarithmic memory complexity for smooth solutions. The core mathematical framework is sound, but **critical gaps exist between claims and implementation**.

| Category | Status |
|----------|--------|
| Linear PDE solving (heat, advection) | ✅ Fully native TT/QTT |
| Storage complexity O(log N) | ✅ Proven |
| Compute complexity O(log N) for linear | ✅ Proven |
| Nonlinear CFD (Euler equations) | ⚠️ **Hybrid** — storage native, compute dense |
| WENO-TT reconstruction | ⚠️ **Incomplete** — falls back to dense |
| True O(log N) nonlinear CFD | ❌ **Not implemented** |

---

## Part 1: Claims We Are Making

### Claim A: Logarithmic Storage
> "Physical fields stored in O(log N × r²) memory instead of O(N)"

**Status:** ✅ TRUE  
**Evidence:** `tensornet/cfd/pure_qtt_ops.py` — `QTTState` stores log₂(N) cores  
**Limitation:** Only holds for fields with bounded TT-rank r. Chaotic turbulence may have rank ~ O(N).

### Claim B: Logarithmic Compute for Linear PDEs
> "Heat equation, advection equation solved in O(log N × r³) per timestep"

**Status:** ✅ TRUE  
**Evidence:** `demos/pure_qtt_pde.py` — MPO operators applied directly to QTT cores  
**What it means:** Laplacian, derivatives as Matrix Product Operators. No dense allocation.

### Claim C: Logarithmic Compute for Nonlinear CFD
> "Euler equations solved in O(log N × r³) per timestep"

**Status:** ❌ FALSE (currently)  
**Reality:** `tensornet/cfd/qtt_cfd.py` lines 303-309 explicitly state:
```python
Current implementation:
1. Decompress QTT → dense O(log N · χ²)
2. Compute fluxes in dense O(N)
3. Update in dense O(N)
4. Recompress dense → QTT O(N)

Total: O(N) per step (compression dominated)
```

### Claim D: WENO-TT Shock Capturing
> "5th-order WENO reconstruction performed natively in TT format"

**Status:** ⚠️ PARTIAL  
**Reality:** `tensornet/cfd/weno_tt.py` line 241:
```python
# Simplified implementation: extract dense, compute, re-compress
# Full implementation would use TT arithmetic
```
The infrastructure exists but falls back to dense extraction.

---

## Part 2: What's Actually Novel

### Novel Contributions (if completed):

| Contribution | Novelty | Status |
|-------------|---------|--------|
| MPO-based PDE operators in CFD context | Novel integration | ✅ Done |
| QTT as native CFD state representation | Novel application | ✅ Done |
| Full Euler solver in TT format | Would be first | ❌ Not done |
| WENO weights as tensor contractions | Novel if native | ⚠️ Partial |
| Adaptive rank for shock tracking | Novel | ⚠️ Documented, not proven |

### What Already Exists Elsewhere:

| Component | Prior Art |
|-----------|-----------|
| TT-SVD compression | Oseledets 2011, ttpy, tntorch |
| MPO operators | DMRG community since 1990s |
| QTT format | Khoromskij 2011 |
| WENO schemes | Shu 1998 |
| Euler equations solver | Every CFD textbook |

**The novelty is the integration, not the components.**

---

## Part 3: Current Implementation Status

### ✅ Complete and Working

| Component | Location | Verified |
|-----------|----------|----------|
| `QTTState` class | `pure_qtt_ops.py` | Yes |
| `qtt_add(a, b)` | `pure_qtt_ops.py` | Yes |
| `qtt_scale(a, scalar)` | `pure_qtt_ops.py` | Yes |
| `qtt_inner_product(a, b)` | `pure_qtt_ops.py` | Yes |
| `qtt_norm(a)` | `pure_qtt_ops.py` | Yes |
| `truncate_qtt(a, max_bond)` | `pure_qtt_ops.py` | Yes |
| `derivative_mpo(n, dx)` | `pure_qtt_ops.py` | Yes |
| `laplacian_mpo(n, dx)` | `pure_qtt_ops.py` | Yes |
| `apply_mpo(mpo, qtt)` | `pure_qtt_ops.py` | Yes |
| `dense_to_qtt(tensor)` | `pure_qtt_ops.py` | Yes |
| `qtt_to_dense(qtt)` | `pure_qtt_ops.py` | Yes |
| Heat equation solver | `pure_qtt_pde.py` | Yes |
| Advection equation solver | `pure_qtt_pde.py` | Yes |

### ⚠️ Exists but Hybrid (Dense Fallback)

| Component | Location | Issue |
|-----------|----------|-------|
| `QTT_Euler1D` | `qtt_cfd.py` | Flux computed in dense |
| `QTTEulerState` | `qtt_cfd.py` | Storage native, ops not |
| WENO-TT weights | `weno_tt.py` | Falls back to dense |
| Smoothness indicators | `weno_tt.py` | Falls back to dense |

### ❌ Missing (Required for True Native CFD)

| Component | What it does | Difficulty |
|-----------|-------------|------------|
| `qtt_hadamard(a, b)` | Element-wise product | Easy (exists in SDK) |
| `qtt_div(a, b)` | Element-wise division | Medium (Newton-Schulz) |
| `qtt_sqrt(a)` | Element-wise sqrt | Medium (Newton iteration) |
| `qtt_apply_function(a, f)` | Pointwise function | Hard (TCI-based) |
| Native Rusanov flux | Flux without decompression | Hard |
| Native WENO weights | Weights without decompression | Hard |

---

## Part 4: Gap Analysis — What Must Be Built

### Phase 1: Core Arithmetic (Priority: CRITICAL)

**Goal:** Complete the TT arithmetic library so CFD can be native.

| Task | Description | Est. Time |
|------|-------------|-----------|
| 1.1 | Import `qtt_elementwise_product` from SDK into `pure_qtt_ops.py` | 15 min |
| 1.2 | Implement `qtt_hadamard(a, b)` as alias with truncation | 15 min |
| 1.3 | Implement `qtt_inverse_newton(a, tol)` — Newton-Schulz for 1/a | 1 hr |
| 1.4 | Implement `qtt_div(a, b)` = `qtt_hadamard(a, qtt_inverse_newton(b))` | 30 min |
| 1.5 | Implement `qtt_sqrt_newton(a)` — Newton iteration for √a | 1 hr |
| 1.6 | **Implement `qtt_from_function(f, n_qubits, max_rank)`** — TT-Cross | 3 hr |
| 1.7 | Unit tests for all new operations | 1 hr |

**Total Phase 1:** ~7 hours

### Phase 2: Native Euler Flux (Priority: HIGH)

**Goal:** Compute Rusanov flux entirely in TT format using TT-Cross Approximation.

**Primary Approach: TT-Cross Approximation (TCI)**

Instead of decomposing flux into primitive TT operations (which causes rank explosion),
we treat the flux as a black-box function and approximate it directly:

```python
def flux_at_index(i, rho_qtt, rhou_qtt, E_qtt):
    """Evaluate flux at single index i by querying QTT cores."""
    rho_i = qtt_eval_at_index(rho_qtt, i)
    rhou_i = qtt_eval_at_index(rhou_qtt, i)
    E_i = qtt_eval_at_index(E_qtt, i)
    
    u = rhou_i / rho_i
    p = (gamma - 1) * (E_i - 0.5 * rho_i * u**2)
    
    return [rhou_i, rhou_i*u + p, u*(E_i + p)]

# Build flux QTT by sampling at O(r² × log N) indices
F_qtt = qtt_from_function(flux_at_index, n_qubits, max_rank)
```

**Why TCI is superior:**
- No rank explosion from repeated Hadamard products
- Single QTT construction per field
- Naturally finds low-rank structure
- O(r² × log N × r³) = O(log N × r⁵) per flux evaluation
- Well-established algorithm (Oseledets & Tyrtyshnikov 2010)

**Fallback: Pure TT Arithmetic**
For cases where TCI struggles (very sharp shocks), we have Option A primitives.

| Task | Description | Est. Time |
|------|-------------|-----------|
| 2.1 | Implement `qtt_eval_at_index(qtt, i)` — O(log N × r²) point query | 1 hr |
| 2.2 | Implement `qtt_eval_batch(qtt, indices)` — vectorized GPU evaluation | 1 hr |
| 2.3 | **Scaffold Rust crate `tci_core/` with PyO3 + ndarray** | 2 hr |
| 2.4 | **Implement DLPack bridge (`__dlpack__` protocol) for zero-copy** | 1 hr |
| 2.5 | **Implement neighbor index generation in Rust (handles i+1 with carry)** | 1 hr |
| 2.6 | **Implement TCI fiber-based sampling in Rust** | 2 hr |
| 2.7 | **Implement MaxVol pivot selection in Rust** | 2 hr |
| 2.8 | **Implement adaptive refinement fallback** | 1 hr |
| 2.9 | Implement `qtt_from_function(f, n_qubits, max_rank)` Python wrapper | 1 hr |
| 2.10 | **CRITICAL: Implement rank truncation policy with hard cap (r ≤ 128)** | 1 hr |
| 2.11 | Implement `flux_batch()` — batched Rusanov on GPU (contiguous QTT cores) | 1 hr |
| 2.12 | Implement `qtt_rusanov_flux_tci(rho, rhou, E)` — full TCI loop | 2 hr |
| 2.13 | Integration test on Sod shock tube | 2 hr |
| 2.14 | Benchmark: TCI vs hybrid, batch sizes, rank limits | 1 hr |

**Total Phase 2:** ~19 hours

### Phase 3: Native WENO-TT (Priority: MEDIUM)

**Goal:** WENO5 reconstruction entirely in TT format.

| Task | Description | Est. Time |
|------|-------------|-----------|
| 3.1 | Implement stencil shift operators as MPOs | 2 hr |
| 3.2 | Implement smoothness indicator β as TT quadratic form | 4 hr |
| 3.3 | Implement WENO-Z weight formula in TT | 4 hr |
| 3.4 | Implement polynomial reconstruction in TT | 3 hr |
| 3.5 | Replace `weno_tt.py` dense fallbacks | 2 hr |
| 3.6 | Convergence tests (smooth and shock) | 2 hr |

**Total Phase 3:** ~17 hours

### Phase 4: Validation and Proof (Priority: HIGH)

**Goal:** Prove the implementation is correct and actually achieves O(log N).

| Task | Description | Est. Time |
|------|-------------|-----------|
| 4.1 | Memory profiling: verify no O(N) allocations | 2 hr |
| 4.2 | Scaling test: N = 2^10 to 2^20, plot memory | 1 hr |
| 4.3 | Accuracy test: compare to dense solver | 2 hr |
| 4.4 | Conservation test: mass, momentum, energy | 1 hr |
| 4.5 | Shock tube validation against exact Riemann | 2 hr |
| 4.6 | Generate evidence pack with cryptographic hashes | 1 hr |

**Total Phase 4:** ~9 hours

---

## Part 5: TCI Sampling Strategy

### The Core Question

TT-Cross Approximation builds a QTT by evaluating the function at carefully chosen indices.
The quality and efficiency of the approximation depends entirely on **which indices we sample**.

For N = 2^n, each index i can be written in binary: `i = (i₁, i₂, ..., iₙ)` where `iₖ ∈ {0, 1}`.

### Strategy Options

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Fiber-Based** | Sample along TT "fibers" (fix all dims except one) | Smooth functions |
| **Greedy Pivot** | Random init, add max-residual points | General purpose |
| **Adaptive Refinement** | Start coarse, refine high-error regions | Shocks, discontinuities |
| **Quasi-Random (Sobol)** | Low-discrepancy sequence sampling | Uniform coverage |
| **MaxVol** | Maximize skeleton matrix determinant | Optimal pivots |

### Recommended: Hybrid Fiber + Adaptive

For CFD flux functions:
1. **Fiber-based TCI** for smooth regions (cheap, provably good)
2. **Adaptive refinement** when residual stays high (shock detected)
3. **MaxVol pivot selection** for optimal basis

```
Smooth region: Fiber sampling is sufficient
    ──────────────────────────────────────
    ·    ·    ·    ·    ·    ·    ·    · (sparse samples)

Shock region: Adaptive clustering
    ───────┐┌───────
           ││
    ·  ·  ·│├·  ·  · (samples cluster at discontinuity)
           ││
    ───────┘└───────
```

### Fiber-Based Sampling (Primary)

For each TT core dimension k, sample "fibers" — fixing all indices except k:

```rust
pub struct FiberSampler {
    n_qubits: usize,
    pivots_left: Vec<Vec<usize>>,   // Multi-indices for dims 1..k-1
    pivots_right: Vec<Vec<usize>>,  // Multi-indices for dims k+1..n
}

impl FiberSampler {
    /// Generate fiber indices for dimension k
    pub fn get_fiber_indices(&self, dim: usize) -> Vec<usize> {
        let mut indices = Vec::new();
        for &left in &self.pivots_left[dim] {
            for bit in 0..2 {  // Sample both 0 and 1 at this qubit
                for &right in &self.pivots_right[dim] {
                    let idx = compose_index(left, bit, right, dim);
                    indices.push(idx);
                }
            }
        }
        indices
    }
}
```

**Sample complexity:** O(r² × n) function evaluations for rank-r approximation on n qubits.

### QTT Batch Evaluation (Critical Implementation)

The `qtt_eval_batch` function is the performance-critical path. It evaluates the QTT at 
multiple indices simultaneously using batched matrix-vector products.

**Mathematical Operation:**
For index $i = (b_1, b_2, \ldots, b_n)$ in binary, the QTT value is:
$$f(i) = G_1[b_1] \cdot G_2[b_2] \cdot \ldots \cdot G_n[b_n]$$
where $G_k[b_k]$ is an $r \times r$ matrix selected by bit $b_k$.

**Batched Implementation:**
```python
def qtt_eval_batch(
    qtt_cores: torch.Tensor,  # (n_qubits, 2, r, r) contiguous
    indices: torch.Tensor,     # (batch,) int64 indices
) -> torch.Tensor:
    """
    Evaluate QTT at batch of indices.
    
    This is a gather + batched matmul sequence.
    For torch.compile to fuse effectively:
    1. QTT cores MUST be contiguous tensor, not list
    2. Use torch.gather for bit extraction
    3. Use torch.bmm for batched matrix multiply
    """
    batch_size = indices.shape[0]
    n_qubits = qtt_cores.shape[0]
    r = qtt_cores.shape[2]
    
    # Extract bits from indices: (batch, n_qubits)
    bits = extract_bits(indices, n_qubits)  # See below
    
    # Initialize accumulator as identity: (batch, 1, r)
    result = torch.zeros(batch_size, 1, r, device=indices.device)
    result[:, 0, 0] = 1.0  # Start with [1, 0, 0, ...]
    
    # Sequential matmul through cores (cannot parallelize due to dependency)
    for k in range(n_qubits):
        # Select matrices based on bit k: (batch, r, r)
        bit_k = bits[:, k]  # (batch,)
        matrices = qtt_cores[k, bit_k]  # (batch, r, r) via advanced indexing
        
        # Batched matmul: (batch, 1, r) @ (batch, r, r) → (batch, 1, r)
        result = torch.bmm(result, matrices)
    
    # Final result is in result[:, 0, 0] after contracting with right boundary
    return result[:, 0, 0]  # (batch,)

def extract_bits(indices: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Extract binary representation as tensor of bits."""
    # indices: (batch,) int64
    # returns: (batch, n_bits) int64, LSB first
    bits = torch.zeros(indices.shape[0], n_bits, dtype=torch.int64, device=indices.device)
    for k in range(n_bits):
        bits[:, k] = (indices >> k) & 1
    return bits
```

**Performance Notes:**
- The loop over `n_qubits` is O(log N) iterations — acceptable
- Each iteration is a batched matmul: O(batch × r²) — GPU-efficient
- Total: O(batch × log N × r²) — dominated by GPU throughput, not loop overhead
- `torch.compile` will fuse the bit extraction and matmuls into fewer kernels

### Adaptive Refinement (For Shocks)

When fiber-based TCI shows high residual in a region, switch to adaptive:

```rust
pub fn adaptive_refine(
    current_samples: &[(usize, f64)],
    qtt: &QTTState,
    tolerance: f64,
    max_samples: usize,
) -> Vec<usize> {
    let mut new_samples = Vec::new();
    
    // Check residual at midpoints between existing samples
    for window in current_samples.windows(2) {
        let (i1, v1) = window[0];
        let (i2, v2) = window[1];
        let mid = (i1 + i2) / 2;
        
        // Evaluate QTT approximation at midpoint
        let approx = qtt_eval_at_index(qtt, mid);
        let true_val = /* need to evaluate f(mid) */;
        let error = (approx - true_val).abs();
        
        if error > tolerance {
            new_samples.push(mid);
        }
    }
    
    // Sort by error, take top max_samples
    new_samples.truncate(max_samples);
    new_samples
}
```

### MaxVol Pivot Selection

After gathering samples, choose pivots to maximize the "volume" (determinant) of the 
skeleton matrix — this ensures linear independence and optimal approximation.

**Mathematical Foundation:**

Given matrix $A \in \mathbb{R}^{m \times r}$ (m samples, r pivots needed), find r row indices 
such that the selected submatrix has **maximum volume** (largest absolute determinant).

The key insight: $C = A \cdot B^{-1}$ where $B = A[\text{pivots}, :]$. Then:
- $C[i, j]$ = "if I replace pivot j with row i, by what factor does volume change?"
- If $|C[i,j]| > 1$, swapping increases volume (good!)
- If $\max|C[i,j]| < 1 + \delta$ for all non-pivot rows, we're at a local maximum

**Algorithm (Goreinov-Tyrtyshnikov):**
```
Input: A ∈ ℝ^{m×r}, tolerance δ
Output: pivot indices I = {i₁, i₂, ..., iᵣ}

1. Initialize: I = {0, 1, ..., r-1}  (first r rows)
2. Loop:
   a. B = A[I, :]           # r × r submatrix
   b. B⁻¹ = inverse(B)      # Use pseudo-inverse for stability
   c. C = A @ B⁻¹           # m × r matrix
   
   d. Find (i*, j*) = argmax |C[i,j]| for i ∉ I
   
   e. If |C[i*, j*]| < 1 + δ:
        STOP (converged)
      Else:
        I[j*] = i*          # Swap: row i* replaces pivot at position j*
        
3. Return I
```

**Rust Implementation:**
```rust
/// MaxVol algorithm for pivot selection
/// 
/// Complexity: O(m × r²) per iteration, typically 2-5 iterations
pub fn maxvol(
    a: &Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<usize>, MaxVolError> {
    let (m, r) = a.dim();
    if m < r {
        return Err(MaxVolError::NotEnoughRows { m, r });
    }
    
    let mut pivots: Vec<usize> = (0..r).collect();
    
    for _iter in 0..max_iterations {
        let b = select_rows(a, &pivots);
        let b_inv = regularized_inverse(&b, 1e-12)?;  // SVD-based for stability
        let c = a.dot(&b_inv);
        
        // Find maximum |C[i,j]| where i not already a pivot
        let pivot_set: HashSet<usize> = pivots.iter().cloned().collect();
        let (max_i, max_j, max_val) = find_max_outside(&c, &pivot_set);
        
        if max_val < 1.0 + tolerance {
            return Ok(pivots);  // Converged
        }
        
        pivots[max_j] = max_i;  // Swap
    }
    
    Err(MaxVolError::NotConverged)
}

/// SVD-based pseudo-inverse for numerical stability
fn regularized_inverse(b: &Array2<f64>, epsilon: f64) -> Result<Array2<f64>, MaxVolError> {
    let svd = b.svd(true, true)?;
    let s_inv: Array1<f64> = svd.singular_values.mapv(|x| {
        if x > epsilon { 1.0 / x } else { 0.0 }
    });
    Ok(svd.vt.t().dot(&Array2::from_diag(&s_inv)).dot(&svd.u.t()))
}
```

**Failure Modes and Mitigations:**

| Failure Mode | Symptom | Cause | Fix |
|-------------|---------|-------|-----|
| Singular B | NaN/Inf in C | Linearly dependent pivots | Use pseudo-inverse |
| Stagnation | max\|C\| stuck at ~1.0 | Local minimum | Random restart |
| Slow convergence | >20 iterations | High true rank | Increase max_rank |
| Oscillation | Pivots keep swapping | tolerance too tight | Increase δ to 0.1 |

**Recommended Parameters:**
```rust
pub struct MaxVolConfig {
    pub tolerance: f64,        // 0.05 (default)
    pub max_iterations: usize, // 15
    pub regularization: f64,   // 1e-12
    pub random_restarts: usize, // 2 if stagnating
}
```

**Convergence Guarantee (Theorem):** If the target function has TT-rank r, then TCI with 
MaxVol pivoting converges in O(n × r) function evaluations to a quasi-optimal approximation.

### TCI Convergence Criteria

The TCI loop terminates when the **relative residual** drops below tolerance:

$$\text{residual} = \frac{\|f - \tilde{f}\|_\infty}{\|f\|_\infty} < \epsilon$$

**Practical Implementation:**
```rust
pub fn check_convergence(&self) -> bool {
    // Estimate error at random non-pivot points
    let test_indices = self.sample_random_indices(100);
    let max_residual = test_indices.iter()
        .map(|&i| {
            let true_val = self.samples.get(&i).copied().unwrap_or(0.0);
            let approx_val = self.evaluate_approximation(i);
            (true_val - approx_val).abs()
        })
        .fold(0.0, f64::max);
    
    let max_value = self.samples.values()
        .fold(0.0, |acc, &v| f64::max(acc, v.abs()));
    
    let relative_residual = max_residual / (max_value + 1e-15);
    relative_residual < self.tolerance
}
```
```

### Complete TCI Sampler Architecture

```rust
// tci_core/src/sampler.rs

pub enum SamplingPhase {
    FiberSweep { current_dim: usize, sweep_count: usize },
    AdaptiveRefine { high_error_regions: Vec<(usize, usize)> },
    Converged,
}

pub struct TCISampler {
    pub n_qubits: usize,
    pub max_rank: usize,
    pub tolerance: f64,
    
    // Pivot state
    pivots_left: Vec<HashSet<usize>>,
    pivots_right: Vec<HashSet<usize>>,
    
    // Sample cache
    samples: HashMap<usize, f64>,
    
    // Current phase
    phase: SamplingPhase,
}

impl TCISampler {
    /// Get next batch of indices to evaluate
    pub fn get_sample_indices(&self, batch_size: usize) -> Vec<usize> {
        match &self.phase {
            SamplingPhase::FiberSweep { current_dim, .. } => {
                self.fiber_indices(*current_dim)
            }
            SamplingPhase::AdaptiveRefine { high_error_regions } => {
                self.adaptive_indices(high_error_regions, batch_size)
            }
            SamplingPhase::Converged => vec![],
        }
    }
    
    /// Receive function values, update pivots, advance phase
    pub fn update(&mut self, indices: Vec<usize>, values: Vec<f64>) {
        // Store samples
        for (i, v) in indices.iter().zip(values.iter()) {
            self.samples.insert(*i, *v);
        }
        
        // MaxVol pivot selection
        self.update_pivots_maxvol();
        
        // Check convergence, possibly switch to adaptive phase
        self.advance_phase();
    }
    
    /// Build final QTT cores from samples
    pub fn build_qtt(&self) -> Vec<Array3<f64>> {
        // Construct TT cores from skeleton decomposition
        self.skeleton_to_tt_cores()
    }
}
```

### CFD-Specific: Batched Flux Evaluation

**Batching is non-negotiable.** The architecture must be:

1. **Rust generates 10,000+ indices** in one batch
2. **Python ships batch to GPU** in one H2D transfer (80 KB for int64)
3. **PyTorch computes flux** in one fused kernel
4. **Python returns batch** in one D2H transfer
5. **Rust updates pivots** using MaxVol

**Zero-copy via DLPack:**
```rust
// Rust side: expose indices via DLPack protocol
#[pymethods]
impl IndexBatch {
    fn __dlpack__(&self, py: Python) -> PyResult<PyObject> {
        // Return DLPack capsule - PyTorch consumes directly
        create_dlpack_capsule(&self.indices, py)
    }
    fn __dlpack_device__(&self) -> (i32, i32) { 
        (1, 0)  // kDLCPU, device_id=0
    }
}
```
```python
# Python side: zero-copy consumption
indices_cpu = torch.from_dlpack(rust_batch)  # No copy on CPU
indices_gpu = indices_cpu.cuda()              # One H2D transfer
```

### Flux Implementation: Rusanov (Local Lax-Friedrichs)

**Phase 1 flux scheme:** Rusanov — simple, robust, GPU-friendly, no branching.

The Rusanov flux at interface i+½:

$$\mathbf{F}_{i+1/2} = \frac{1}{2}(\mathbf{F}_L + \mathbf{F}_R) - \frac{1}{2}\lambda_{\max}(\mathbf{U}_R - \mathbf{U}_L)$$

Where:
- $\mathbf{U} = [\rho, \rho u, E]^T$ = conserved variables
- $\mathbf{F} = [\rho u, \rho u^2 + p, u(E+p)]^T$ = physical flux
- $\lambda_{\max} = \max(|u_L| + c_L, |u_R| + c_R)$ = maximum wave speed
- $c = \sqrt{\gamma p / \rho}$ = sound speed (**CRITICAL: must include sqrt or CFL blows up**)
- $p = (\gamma - 1)(E - \frac{1}{2}\rho u^2)$ = pressure (ideal gas)

**⚠️ THE NEIGHBOR TRAP — DO NOT COMPUTE `i+1` ON GPU:**

In QTT format, the physical index `i` is mapped to binary bits across tensor cores.
Computing `i+1` requires binary carry propagation (e.g., `0111 + 1 = 1000` flips 4 bits).
This causes divergent GPU threads and destroys performance.

**Solution:** Rust generates BOTH index vectors:
```rust
// In Rust TCI core - handle topology on CPU
pub fn generate_interface_indices(&self, batch_size: usize) -> (Vec<i64>, Vec<i64>) {
    let indices_L: Vec<i64> = self.select_pivot_indices(batch_size);
    let indices_R: Vec<i64> = indices_L.iter()
        .map(|&i| {
            let next = i + 1;
            if next >= self.n_points { 0 } else { next }  // Periodic BC
        })
        .collect();
    (indices_L, indices_R)
}
```

**Batched GPU implementation (corrected):**
```python
@torch.compile(mode="reduce-overhead")
def flux_batch(
    indices_L: torch.Tensor,      # (10000,) LEFT cell indices from Rust
    indices_R: torch.Tensor,      # (10000,) RIGHT cell indices from Rust (i+1 precomputed)
    qtt_cores: torch.Tensor,      # (n_fields, n_qubits, 2, r, r) contiguous storage
    gamma: float = 1.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single FUSED GPU kernel for 10,000+ Rusanov flux evaluations.
    
    CRITICAL DESIGN DECISIONS:
    1. indices_L and indices_R are BOTH passed from Rust (no GPU-side i+1)
    2. qtt_cores is a CONTIGUOUS tensor, not a Python list (enables torch.compile fusion)
    3. All intermediate values stay in GPU registers (no VRAM round-trips)
    """
    # Batch-evaluate QTT at LEFT cells (indices from Rust)
    rho_L = qtt_eval_batch(qtt_cores[0], indices_L)
    rhou_L = qtt_eval_batch(qtt_cores[1], indices_L)
    E_L = qtt_eval_batch(qtt_cores[2], indices_L)
    
    # Batch-evaluate QTT at RIGHT cells (indices from Rust - NO i+1 here!)
    rho_R = qtt_eval_batch(qtt_cores[0], indices_R)
    rhou_R = qtt_eval_batch(qtt_cores[1], indices_R)
    E_R = qtt_eval_batch(qtt_cores[2], indices_R)
    
    # Primitive variables (fully vectorized, stays in registers)
    u_L = rhou_L / rho_L
    u_R = rhou_R / rho_R
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
    
    # Sound speed: c = sqrt(γp/ρ) — MUST BE SQRT, NOT γp/ρ!
    # Dimensional analysis: [c] = m/s, [γp/ρ] = m²/s² → need sqrt
    c_L = torch.sqrt(gamma * p_L / rho_L)
    c_R = torch.sqrt(gamma * p_R / rho_R)
    
    # Physical flux vectors
    F_rho_L, F_rho_R = rhou_L, rhou_R
    F_rhou_L = rhou_L * u_L + p_L
    F_rhou_R = rhou_R * u_R + p_R
    F_E_L = u_L * (E_L + p_L)
    F_E_R = u_R * (E_R + p_R)
    
    # Maximum wave speed (Rusanov dissipation)
    lambda_max = torch.maximum(torch.abs(u_L) + c_L, torch.abs(u_R) + c_R)
    
    # Rusanov flux = central average - dissipation
    F_rho = 0.5 * (F_rho_L + F_rho_R) - 0.5 * lambda_max * (rho_R - rho_L)
    F_rhou = 0.5 * (F_rhou_L + F_rhou_R) - 0.5 * lambda_max * (rhou_R - rhou_L)
    F_E = 0.5 * (F_E_L + F_E_R) - 0.5 * lambda_max * (E_R - E_L)
    
    return F_rho, F_rhou, F_E  # Each is (10000,) tensor
```

**QTT Core Storage Format:**

For `torch.compile` to fuse kernels effectively, store QTT cores as a **contiguous tensor**:
```python
# BAD: Python list of tensors (torch.compile can't optimize the loop)
qtt_cores = [core_0, core_1, ..., core_n]  # ❌

# GOOD: Single contiguous tensor (enables gather + batched matmul fusion)
qtt_cores = torch.stack(cores, dim=0)  # shape: (n_fields, n_qubits, 2, r, r) ✅
```

**Future upgrade:** HLLC flux for reduced numerical diffusion (Phase 3+).

### Complete TCI→Flux→QTT Flow (Corrected)

```
TCI Loop (3-5 iterations typical for rank-64)
──────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────┐
│  RUST TCI CORE (CPU)                                            │
│  ───────────────────────────────────────────────────────────    │
│  1. generate_pivots(batch_size=10000)                           │
│  2. compute_neighbors(indices_L) → indices_R                    │
│     ↳ Handle i+1 HERE (binary carry on CPU, not GPU)            │
│     ↳ Handle boundary conditions (periodic/reflective)          │
│  3. Return (indices_L, indices_R) as DLPack capsules            │
└────────────────────────────┬────────────────────────────────────┘
                             │
      ┌──────────────────────┴──────────────────────┐
      │ DLPack (__dlpack__ protocol)                │
      │ indices_L: 80 KB    indices_R: 80 KB        │
      └──────────────────────┬──────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYTHON BRIDGE                                                  │
│  ───────────────────────────────────────────────────────────    │
│  indices_L_gpu = torch.from_dlpack(rust_L).cuda()  # 1 H2D      │
│  indices_R_gpu = torch.from_dlpack(rust_R).cuda()  # 1 H2D      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYTORCH GPU — ONE FUSED KERNEL (torch.compile)                 │
│  ───────────────────────────────────────────────────────────    │
│  @torch.compile(mode="reduce-overhead")                         │
│  def flux_batch(indices_L, indices_R, qtt_cores, gamma):        │
│      # All ops fused — intermediate values in L1/registers      │
│      ρ_L = qtt_eval_batch(qtt_cores[0], indices_L)              │
│      ρ_R = qtt_eval_batch(qtt_cores[0], indices_R)  # NOT i+1!  │
│      ...                                                        │
│      c = sqrt(γ * p / ρ)  # ← MUST BE SQRT                      │
│      ...                                                        │
│      return F_ρ, F_ρu, F_E  # Only final values hit VRAM        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYTHON BRIDGE                                                  │
│  ───────────────────────────────────────────────────────────    │
│  flux_values = (F_rho.cpu(), F_rhou.cpu(), F_E.cpu())           │
│  → DLPack or buffer protocol back to Rust                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  RUST TCI CORE (CPU)                                            │
│  ───────────────────────────────────────────────────────────    │
│  update_skeleton(flux_values)    →  O(r² × batch)               │
│  maxvol_pivots()                 →  O(r³) per dimension         │
│  check_convergence()             →  ‖residual‖ / ‖values‖ < tol │
│                                                                 │
│  if NOT converged: loop back to generate_pivots()               │
│  if converged:                                                  │
│      build_qtt_cores() → skeleton → TT cores (NumPy)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYTHON POST-PROCESSING                                         │
│  ───────────────────────────────────────────────────────────    │
│  cores = [torch.from_numpy(c) for c in rust_cores]              │
│  cores = truncate_qtt(cores, r_max=128)  # CRITICAL: SVD cap    │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture Works

| Component | Single-Index | Batched (10K) | Speedup |
|-----------|--------------|---------------|---------|
| Python→Rust calls | 10,000 | 1 | 10,000× |
| H2D transfers | 10,000 | 2 | 5,000× |
| GPU kernel launches | 10,000 | 1 | 10,000× |
| Rust→Python returns | 10,000 | 1 | 10,000× |
| Neighbor computation | 10K GPU ops | 10K CPU ops | ∞ (no divergence) |

**Critical optimizations:**
1. **Neighbor indices computed in Rust** — no GPU thread divergence from binary carry
2. **QTT cores as contiguous tensor** — enables torch.compile kernel fusion
3. **Single fused kernel** — all intermediate values stay in L1 cache / registers
4. **Only final flux values written to VRAM** — minimizes memory bandwidth

With batching + DLPack + kernel fusion, a single TCI iteration at 10K samples takes **~1ms** instead of ~10s.

---

## Part 6: Execution Plan

### Dependencies and Build Requirements

**Rust Crate (`tci_core/`):**
```toml
# Cargo.toml
[package]
name = "tci_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "tci_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
dlpark = "0.2"  # DLPack support

[dev-dependencies]
criterion = "0.5"  # Benchmarking
```

**Python Environment:**
```bash
# Additional dependencies for Phase 2
pip install maturin  # Rust→Python build tool
pip install torch>=2.0  # For torch.compile
```

**Build Command:**
```bash
cd tci_core && maturin develop --release
```

### Success Metrics (Per Phase)

**Phase 1: Core Arithmetic**
| Metric | Target | Validation Method |
|--------|--------|-------------------|
| `qtt_hadamard` correctness | Error < 1e-12 | Compare to dense |
| `qtt_div` convergence | 10 Newton iterations | Unit test |
| `qtt_sqrt` convergence | 10 Newton iterations | Unit test |
| Truncation preserves norm | \|‖a‖ - ‖truncate(a)‖\| / ‖a‖ < 1e-6 | Unit test |

**Phase 2: Native Euler Flux**
| Metric | Target | Validation Method |
|--------|--------|-------------------|
| TCI convergence | < 10 iterations for smooth | Test on sin(x) |
| MaxVol convergence | < 5 iterations | Unit test |
| `flux_batch` throughput | > 1M flux/sec | Benchmark |
| Memory usage | O(log N) verified | Profiler |
| Sod shock accuracy | L1 error < 1% vs exact | Integration test |

**Phase 3: Native WENO-TT**
| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Smooth flow order | 5th order convergence | Manufactured solution |
| Shock capturing | No oscillations | Visual + TVD check |
| WENO weight accuracy | Match dense WENO | Comparison test |

**Phase 4: Validation**
| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Conservation error | < 1e-10 | Mass/momentum/energy check |
| Scaling exponent | Memory ∝ log(N) | Fit regression |
| Shock tube L1 error | < 1% | Exact Riemann comparison |

### Testing Strategy

**Unit Tests (Rust):**
```rust
// tci_core/tests/test_maxvol.rs
#[test]
fn test_maxvol_converges_for_low_rank_matrix() {
    let a = generate_rank_r_matrix(100, 10);  // 100 rows, rank 10
    let pivots = maxvol(&a, 0.05, 20).unwrap();
    assert_eq!(pivots.len(), 10);
    
    // Verify volume is near-optimal
    let submatrix = select_rows(&a, &pivots);
    let volume = submatrix.det().abs();
    assert!(volume > 0.9 * optimal_volume(&a, 10));
}

#[test]
fn test_maxvol_handles_near_singular() {
    let a = generate_nearly_singular_matrix(50, 5, 1e-10);
    let result = maxvol(&a, 0.05, 20);
    assert!(result.is_ok());  // Should not panic
}
```

**Integration Tests (Python):**
```python
# tests/test_tci_integration.py
def test_tci_sod_shock_tube():
    """End-to-end test: TCI-based Euler solver on Sod problem."""
    # Setup
    n_qubits = 12  # N = 4096
    rho_L, rho_R = 1.0, 0.125
    p_L, p_R = 1.0, 0.1
    
    # Run solver
    solver = QTTEulerSolver(n_qubits, max_rank=64)
    solver.initialize_sod(rho_L, rho_R, p_L, p_R)
    solver.step(dt=0.001, n_steps=100)
    
    # Validate
    rho_exact = exact_riemann_solution(x, t=0.1)
    rho_computed = solver.get_density()
    
    l1_error = np.mean(np.abs(rho_computed - rho_exact))
    assert l1_error < 0.01, f"L1 error {l1_error} exceeds 1%"
    
    # Check O(log N) memory
    assert solver.memory_usage() < 1e6  # Less than 1 MB for N=4096

def test_tci_convergence_smooth():
    """TCI should converge fast for smooth functions."""
    def smooth_flux(i):
        return np.sin(2 * np.pi * i / N)
    
    sampler = TCISampler(n_qubits=16, max_rank=8)
    iterations = 0
    while not sampler.converged(tol=1e-8):
        indices = sampler.get_sample_indices(batch_size=1000)
        values = [smooth_flux(i) for i in indices]
        sampler.update(indices, values)
        iterations += 1
        assert iterations < 10, "TCI failed to converge for smooth function"
```

**Benchmarks:**
```python
# benchmarks/bench_flux_batch.py
import torch
import time

def benchmark_flux_batch():
    """Measure flux_batch throughput."""
    indices_L = torch.randint(0, 2**20, (10000,), device='cuda')
    indices_R = torch.randint(0, 2**20, (10000,), device='cuda')
    qtt_cores = torch.randn(3, 20, 2, 64, 64, device='cuda')
    
    # Warmup
    for _ in range(10):
        flux_batch(indices_L, indices_R, qtt_cores)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        flux_batch(indices_L, indices_R, qtt_cores)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = 100 * 10000 / elapsed
    print(f"Throughput: {throughput:.0f} flux evaluations/sec")
    assert throughput > 1e6, "Flux batch too slow"
```

### Rollback Strategy

If TCI fails to converge or produces unacceptable accuracy:

| Failure | Detection | Rollback |
|---------|-----------|----------|
| TCI doesn't converge | iterations > 20 | Fall back to hybrid (dense flux) |
| Rank explodes | r > 128 after truncation | Increase truncation aggressiveness |
| Accuracy too low | L1 error > 5% | Increase max_rank or switch to dense |
| GPU OOM | CUDA error | Reduce batch size, fall back to CPU |

**Graceful Degradation:**
```python
def compute_flux_with_fallback(rho_qtt, rhou_qtt, E_qtt, max_rank=64):
    """Try TCI, fall back to hybrid if it fails."""
    try:
        flux_qtt = qtt_rusanov_flux_tci(rho_qtt, rhou_qtt, E_qtt, max_rank)
        if flux_qtt.rank > 128:
            warnings.warn("Rank explosion detected, falling back to hybrid")
            return compute_flux_hybrid(rho_qtt, rhou_qtt, E_qtt)
        return flux_qtt
    except TCIConvergenceError:
        warnings.warn("TCI failed to converge, falling back to hybrid")
        return compute_flux_hybrid(rho_qtt, rhou_qtt, E_qtt)
```

### Week 1: Foundation
- [ ] **Day 1:** Phase 1.1-1.4 — Import Hadamard, implement `qtt_div` with Newton-Schulz
- [ ] **Day 2:** Phase 1.5-1.7 — Implement `qtt_sqrt`, `qtt_from_function` stub, unit tests
- [ ] **Day 3:** Phase 2.1-2.2 — `qtt_eval_at_index`, `qtt_eval_batch` (PyTorch)
- [ ] **Day 4:** Phase 2.3-2.4 — Scaffold Rust crate, DLPack bridge
- [ ] **Day 5:** Phase 2.5-2.6 — Neighbor indexing, TCI fiber sampling (Rust)

### Week 2: TCI Core
- [ ] **Day 1:** Phase 2.7 — MaxVol implementation with SVD pseudo-inverse (Rust)
- [ ] **Day 2:** Phase 2.8-2.9 — Adaptive refinement, Python wrapper
- [ ] **Day 3:** Phase 2.10-2.11 — Truncation policy, `flux_batch` (GPU)
- [ ] **Day 4:** Phase 2.12 — Full `qtt_rusanov_flux_tci` integration
- [ ] **Day 5:** Phase 2.13-2.14 — Sod shock tube test, benchmarks

### Week 3: WENO + Validation
- [ ] **Day 1-2:** Phase 3.1-3.3 — Stencil MPOs, smoothness indicators
- [ ] **Day 3:** Phase 3.4-3.5 — Polynomial reconstruction, replace dense fallbacks
- [ ] **Day 4:** Phase 3.6 + Phase 4.1-4.2 — Convergence tests, memory profiling
- [ ] **Day 5:** Phase 4.3-4.6 — Accuracy/conservation tests, evidence pack

### Week 4: Polish (if needed)
- [ ] Documentation update with honest claims
- [ ] Demo video (real physics, real compression)
- [ ] Paper/writeup for publication

---

## Part 7: Risk Assessment

### CRITICAL: The Rank Explosion Trap

**The Core Tension:**
When using Adaptive Refinement to capture a shockwave, the Rank (r) of the Tensor Train 
will naturally spike:

| Function Type | Typical Rank |
|---------------|--------------|
| Smooth sine wave | r ≈ 2 |
| Polynomial | r ≈ degree |
| Smooth but localized | r ≈ 10-20 |
| Weak shock | r ≈ 50 |
| Strong shock/discontinuity | r ≈ 100-200+ |

**The Danger:** TCI computational cost scales as O(r³) per core, O(r⁵) overall.
If rank exceeds ~200, the "sparse" TT solver becomes **slower than dense**.

```
Cost comparison at N = 2²⁰ (1 million points):

Dense solver:     O(N)     = O(10⁶)
TT solver (r=10): O(r³ log N) = O(10³ × 20) = O(2×10⁴)    ← 50× faster
TT solver (r=100): O(r³ log N) = O(10⁶ × 20) = O(2×10⁷)   ← 20× SLOWER
TT solver (r=200): O(r³ log N) = O(8×10⁶ × 20) = O(1.6×10⁸) ← 160× SLOWER
```

**The Trap:** Adaptive refinement near shocks → rank grows → cost grows → 
memory grows → GPU OOM or slower than dense.

### Strategic Mitigation: Aggressive Rank Truncation

**Required:** Implement SVD rounding **immediately after** every TCI step.

```python
def tci_with_truncation(f, n_qubits, max_rank, target_rank):
    """TCI with aggressive post-hoc truncation."""
    # Step 1: Build QTT via TCI (may produce rank > target)
    qtt = qtt_from_function_tci(f, n_qubits, max_rank=max_rank)
    
    # Step 2: CRITICAL - Truncate to target rank
    qtt_truncated = truncate_qtt(qtt, max_bond=target_rank, relative_tol=1e-6)
    
    # Step 3: Estimate truncation error
    error = estimate_truncation_error(qtt, qtt_truncated)
    if error > acceptable_threshold:
        warn(f"Truncation error {error:.2e} exceeds threshold")
    
    return qtt_truncated
```

**The Tradeoff Matrix:**

| Strategy | Rank | Accuracy | Speed | Memory |
|----------|------|----------|-------|--------|
| No truncation | Grows unbounded | ✅ High | ❌ Crashes | ❌ OOM |
| Aggressive truncation (r=32) | Fixed | ⚠️ Smears shocks | ✅ Fast | ✅ Bounded |
| Adaptive truncation | Varies | ✅ Balanced | ⚠️ Varies | ⚠️ Monitored |

**Recommended: Adaptive Truncation with Hard Cap**

```rust
// In Rust TCI core
pub struct TruncationPolicy {
    pub target_rank: usize,      // Preferred rank (e.g., 32)
    pub hard_cap: usize,         // Absolute maximum (e.g., 128)
    pub relative_tol: f64,       // SVD cutoff (e.g., 1e-8)
    pub monitor_growth: bool,    // Warn if rank trending up
}

impl TruncationPolicy {
    pub fn truncate(&self, cores: Vec<Array3<f64>>) -> Vec<Array3<f64>> {
        let mut result = cores;
        for k in 0..result.len()-1 {
            // SVD on bond (k, k+1)
            let (u, s, vt) = svd_truncated(&result[k], &result[k+1], 
                                            self.target_rank, self.relative_tol);
            
            // Apply hard cap
            let kept = s.len().min(self.hard_cap);
            result[k] = u[..kept];
            result[k+1] = vt[..kept];
        }
        result
    }
}
```

### Why This Works for CFD

In compressible CFD, shocks are **localized** — they don't spread across the entire domain.
A Sod shock tube at t > 0 has:
- Contact discontinuity: localized at one point
- Shock wave: localized at one point  
- Rarefaction fan: smooth, low rank

**Key insight:** The *local* rank at a shock may be high, but the *average* rank across the 
domain stays manageable if we truncate aggressively.

```
Domain:  [───smooth───][shock][───smooth───][contact][───smooth───]
Rank:    [    r=8     ][ r=64][    r=8     ][ r=32  ][    r=8     ]
                         ↓
After truncation:      [ r=32]            [ r=32  ]
                         ↓
Global effective rank: ~16 (weighted average)
```

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Newton-Schulz divergence for near-zero values | High | Add regularization ε |
| Rank explosion in Hadamard products | High | Aggressive truncation |
| WENO stencil crossing bond boundaries | Medium | Proper MPO formulation |
| Shock discontinuities increasing rank | **CRITICAL** | **Truncation after every TCI step** |
| Truncation smearing shocks | Medium | Adaptive rank near discontinuities |

### Scientific Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| TT-rank not bounded for turbulence | Fatal | Clearly state smooth/low-rank assumption |
| Accuracy loss from truncation | High | Careful tolerance tuning |
| Conservation broken by truncation | High | Symmetric truncation schemes |
| Rank creep over many timesteps | High | Monitor rank evolution, hard cap |

---

## Part 8: Honest Marketing Position

### What We CAN Claim:

> "HyperTensor stores CFD state in O(log N) memory using Quantized Tensor Trains, enabling billion-point simulations on commodity hardware for problems with smooth or piecewise-smooth solutions."

> "Linear PDEs (heat, advection, diffusion) are solved entirely in TT format with O(log N) complexity per timestep."

### What We CANNOT Claim (Yet):

> ~~"Nonlinear CFD in O(log N)"~~ — Currently hybrid, O(N) per step

> ~~"WENO-TT native reconstruction"~~ — Falls back to dense

> ~~"Works for arbitrary turbulence"~~ — Requires bounded TT-rank

### What We Can Claim After Completing This Roadmap:

> "First fully-native tensor-train CFD solver for compressible Euler equations with O(log N) complexity per timestep for bounded-rank solutions."

---

## Part 9: Code Locations Reference

| File | Purpose | Status |
|------|---------|--------|
| `tensornet/cfd/pure_qtt_ops.py` | Core QTT arithmetic | ✅ Foundation |
| `tensornet/cfd/qtt_cfd.py` | QTT Euler solver | ⚠️ Hybrid |
| `tensornet/cfd/weno_tt.py` | WENO in TT | ⚠️ Hybrid |
| `tensornet/cfd/weno.py` | Dense WENO (reference) | ✅ Complete |
| `tensornet/cfd/tt_cfd.py` | MPS Euler solver | ⚠️ Hybrid |
| `sdk/qtt-sdk/src/qtt_sdk/operations.py` | SDK arithmetic | ✅ Has Hadamard |
| `demos/pure_qtt_pde.py` | Linear PDE demo | ✅ Honest |
| `demos/qtt_shock_tube.py` | CFD demo | ⚠️ Hybrid |

---

## Signatures

This document represents an honest assessment of HyperTensor's capabilities as of December 25, 2025.

### Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language** | Python + PyTorch + Rust | Orchestration, GPU, compiled loops |
| **Flux scheme** | Rusanov (LLF) | Simple, robust, no branching |
| **Nonlinear approach** | TT-Cross (TCI) | Avoids rank explosion |
| **Pivot selection** | MaxVol | Provably optimal |
| **GPU bridge** | DLPack | Zero-copy |
| **Neighbor indexing** | Rust-side | Avoid GPU thread divergence |
| **QTT core storage** | Contiguous tensor | Enable kernel fusion |
| **Truncation** | Hard cap r ≤ 128 | Prevent rank explosion |
| **Batch size** | 10,000+ | Amortize Python overhead |

### Critical Implementation Constraints

1. **Sound speed:** $c = \sqrt{\gamma p / \rho}$ — MUST include sqrt
2. **Neighbor indices:** Compute `i+1` in Rust, not on GPU
3. **QTT cores:** Store as `(n_fields, n_qubits, 2, r, r)` contiguous tensor
4. **Truncation:** Apply SVD rounding after EVERY TCI step
5. **Batching:** Never evaluate flux one index at a time

### Quantitative Targets

| Metric | Target |
|--------|--------|
| Total effort | ~52 hours |
| TCI convergence | < 10 iterations (smooth) |
| MaxVol convergence | < 5 iterations |
| Flux throughput | > 1M evaluations/sec |
| Memory scaling | O(log N) verified |
| Sod shock L1 error | < 1% |
| Max rank | ≤ 128 (hard cap) |

### Go/No-Go Criteria

**Phase 2 is SUCCESS if:**
- [ ] TCI converges for smooth flux in < 10 iterations
- [ ] Sod shock tube L1 error < 1% vs exact Riemann
- [ ] Memory usage is O(log N) as measured by profiler
- [ ] No O(N) allocations in hot path

**Phase 2 is FAILURE if:**
- [ ] TCI requires > 50 iterations consistently
- [ ] Rank exceeds hard cap and accuracy degrades > 10%
- [ ] Memory scaling is O(N) despite truncation
- [ ] GPU throughput < 100K flux/sec (10× too slow)

### Attestation

**Gaps identified:** Yes  
**Path to completion:** Clear  
**Estimated effort:** ~52 hours (7 + 19 + 17 + 9)  
**Implementation:** Python + PyTorch + Rust TCI Core + DLPack  

**Critical constraints:**
- Rank truncation after every TCI step (hard cap r ≤ 128)
- Batching: 10,000+ samples per TCI iteration (non-negotiable)
- Neighbor indices computed in Rust, not GPU (avoid binary carry divergence)
- QTT cores stored as contiguous tensor (enable torch.compile fusion)
- Sound speed must include sqrt (or CFL condition blows up)

**Deception in claims:** Forbidden going forward

---

*"The first principle is that you must not fool yourself — and you are the easiest person to fool."*  
— Richard Feynman

*"Sophistication over speed. Exceptionalism is our brand."*  
— HyperTensor Design Philosophy
