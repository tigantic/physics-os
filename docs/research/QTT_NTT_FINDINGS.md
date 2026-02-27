# QTT-NTT Research Findings

**Date:** January 19, 2026  
**Status:** Abandoned - Architectural Limitations Discovered  
**Archived Code:** `archive/rns_qtt_ntt_butterfly_mpo_deprecated.py`

---

## Executive Summary

We attempted to implement NTT (Number Theoretic Transform) using QTT (Quantized Tensor Train) representation with MPO (Matrix Product Operator) butterfly stages. After extensive investigation, we discovered **fundamental architectural limitations** that make this approach non-viable for production use.

**Key Finding:** Sequential MPO-QTT contractions do not compose correctly because each MPO expects to read raw bit indices from physical dimensions, but post-contraction QTT bond structures encode stage-specific state that doesn't semantically align with subsequent MPOs.

---

## What We Built

### 1. RNS (Residue Number System) Infrastructure ✅ WORKS

```python
# NTT-friendly primes for Baby Bear field
p₁ = 7,340,033 = 7 × 2²⁰ + 1   (safe rank ≤ 167)
p₂ = 13,631,489 = 13 × 2²⁰ + 1 (safe rank ≤ 48)
Product = 100,062,752,120,737 >> Baby Bear (2,013,265,921)
```

- **RNS decomposition:** Splits Baby Bear values into two channels
- **CRT reconstruction:** Recombines channels back to Baby Bear
- **Float64 exact:** Products of 23-bit numbers fit in 53-bit mantissa
- **Channel-preserving API:** Avoids lossy CRT roundtrips between stages

### 2. QTT Representation ✅ WORKS

```python
class QTTState:
    # LSB-first convention: core[k] corresponds to bit k
    # Index i = bit₀ + 2×bit₁ + 4×bit₂ + ...
    
    def from_dense(x, prime, max_rank)  # Uses bit-reversal for LSB-first
    def eval_at(index)                  # O(n × r²) per evaluation
    def eval_all()                      # Iterates without memory explosion
    def truncate(max_rank)              # rSVD for large matrices
```

- **Consistent bit ordering:** Fixed LSB-first throughout
- **rSVD integration:** `torch.svd_lowrank` for O(n²k) vs O(n³)
- **Memory safe:** No catastrophic `to_dense()` calls

### 3. Butterfly MPO Construction ✅ WORKS (in isolation)

```python
def build_butterfly_mpo(n_bits, stage, omega, prime):
    # Cores 0..stage-1: Accumulate twiddle index from input bits
    # Core stage: Apply butterfly [[1, ω^j], [1, -ω^j]]
    # Cores stage+1..n-1: Identity
```

- **Single-stage application:** Works correctly
- **N=4 full NTT:** Works correctly (2 stages)
- **N=8 stages 0,1:** Work correctly

### 4. MPO-QTT Contraction ❌ BREAKS ON COMPOSITION

```python
def apply_mpo_to_qtt(qtt, mpo, max_rank):
    # Einsum: contract physical index, Kronecker product of bonds
    result = torch.einsum('aobm,lbr->laorm', m_core, q_core)
    result = result.reshape(sq_l * sm_l, d_out, sq_r * sm_r)
```

- **First application:** Correct
- **Second application:** Correct  
- **Third application:** **WRONG** - Bond structure mismatch

---

## The Fundamental Problem

### Why Composition Fails

**Stage s MPO expects:**
- Physical indices at cores 0..s-1 represent original bit values
- Bond index at butterfly core encodes accumulated twiddle `j = Σ bit_k × 2^k`

**After stages 0..s-1:**
- QTT values are transformed (butterfly-mixed)
- QTT bonds encode stage-specific state, not raw bit accumulation
- Kronecker product `(qtt_bond × mpo_bond)` mixes semantics

**Result:**
- MPO's twiddle selection mechanism reads wrong bond index
- Output values are incorrect for indices where bit structure matters

### Concrete Example (N=8, Stage 2)

```
After stages 0,1:
  QTT values: [12, 5454950, 7340029, 1885075, 16, 5454950, 7340029, 1885075] ✓
  QTT bonds:  [1, 4, 2, 1] - encodes stage 0,1 transformation state

Stage 2 MPO expects:
  Bond index j = bit₀ + 2×bit₁ to select twiddle ω^j
  
What happens:
  Combined bond = QTT bond × MPO bond = (4×2, 2×4) = (8, 8)
  MPO twiddle selection gets mixed with QTT state
  
Result:
  Expected: [28, 3761513, 5454950, 191638, ...]
  Actual:   [28, 3993310, 5454950, 818426, ...]  ← Odd indices wrong
```

### Why Fresh QTT Works

When we create a fresh QTT from post-stage-1 values (instead of applying MPO):
- Bond structure is minimal rank (2, not 4)
- No semantic mixing with previous MPO state
- Stage 2 applies correctly

This proves the issue is **compositional**, not in individual components.

---

## Attempted Fixes (All Failed)

| Fix Attempted | Result |
|---------------|--------|
| Change einsum bond ordering | Same errors, different indices |
| Reverse core order after contraction | Bond dimension mismatch |
| Track bit indices through QTT bonds | Exponential bond growth |
| Truncate between stages | Loses critical information |
| Different twiddle accumulation scheme | Same fundamental issue |

---

## Why NTT Isn't the Right Target Anyway

### NTT is Memory-Bound, Not Compute-Bound

```
Standard GPU NTT (N = 2^24):
- Data: 128 MB
- 24 passes × 2 read/write = 6 GB transferred
- At 500 GB/s → 12 ms

QTT-MPO NTT (if it worked):
- MPO contraction is compute-bound
- Would need to materialize for next stage anyway
- Estimated: 50-100 ms (WORSE)
```

### Random Coefficients Don't Compress

QTT achieves compression for **structured** data. NTT input/output in ZK proofs:
- Polynomial coefficients: Essentially random
- No low-rank structure to exploit
- Compression ratio ≈ 1 (no benefit)

---

## What's Worth Keeping

### 1. RNS Infrastructure
The prime selection and CRT reconstruction work correctly. Useful for any finite field computation that needs float64 safety.

### 2. QTT Core Implementation
`from_dense`, `eval_at`, `eval_all`, `truncate` all work. Useful for structured data that actually compresses.

### 3. Lessons Learned
- MPO composition requires semantic alignment of bond indices
- Tensor network operators don't automatically compose like matrices
- Memory-bound operations don't benefit from compute-focused optimizations

---

## Recommended Direction

Tensor methods can still accelerate ZK proving, but not through QTT-NTT:

1. **Tensor Core Field Arithmetic** - RNS decomposition to INT8, use tensor cores for 20-100× batched mul
2. **QTT for Recursive Proofs** - Nova/HyperNova accumulators have structure that compresses
3. **Structured MSM** - When scalars come from polynomial evaluations, tensor methods apply
4. **Standard GPU NTT** - Use cuFFT-style implementation, it's already optimal

See: `docs/HYPERTENSOR_ZK_STACK.md` for the revised execution plan.

---

## References

- [QTT-Tucker decomposition](https://arxiv.org/abs/0912.2232)
- [Tensor networks for quantum simulation](https://arxiv.org/abs/1306.2164)
- [Baby Bear field (Plonky3)](https://github.com/Plonky3/Plonky3)
- [RNS-based cryptographic implementations](https://eprint.iacr.org/2016/1066)
