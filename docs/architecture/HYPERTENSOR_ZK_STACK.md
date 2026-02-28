# Ontic Engine ZK Stack

## Execution Plan for Tensor-Accelerated Zero-Knowledge Proving

**Version:** 1.1  
**Date:** January 19, 2026  
**Updated:** January 19, 2026 (Added FluidElite ZK Analysis)
**Target Hardware:** NVIDIA RTX 5070/5080/5090 (Blackwell), future tensor-core GPUs

---

## FluidElite ZK Integration (NEW)

### Discovery: FluidElite is Natively ZK-Friendly

We implemented a complete ZK circuit analysis for FluidElite inference:

```
fluidelite/zk/
├── __init__.py           # Module exports
├── circuit_analysis.py   # Constraint counting
├── proof_simulation.py   # Fiat-Shamir proof simulation
├── demo.py              # End-to-end demo
└── ZK_ANALYSIS.md       # Full findings document
```

### Key Results

| Config | Constraints/Token | GPU Prover/Token | Real-Time Viable? |
|--------|-------------------|------------------|-------------------|
| L=8, χ=32 | 16,384 | 8.2 ms | ✅ YES |
| L=16, χ=64 | 131,072 | 65.5 ms | ✅ YES |
| L=16, χ=128 | 524,320 | 262 ms | ❌ Batch only |

### FluidElite vs Transformer for ZK

| Sequence | FluidElite (linear) | Transformer | Winner |
|----------|---------------------|-------------|--------|
| 256 | 134M | 36M | Transformer |
| 1,024 | 537M | 579M | Crossover |
| 4,096 | 2.1B | 9.3B | FluidElite 4× |
| 16,384 | 8.6B | 148B | FluidElite 17× |

**Conclusion:** FluidElite dominates at long contexts (>1000 tokens) due to O(N) vs O(N²) scaling.

### The GELU Problem

GELU activation accounts for 75% of constraints. Solutions:

1. **ZK Linear Mode:** Skip GELU entirely (4× reduction)
2. **Lookup Tables:** Amortize nonlinearity cost
3. **Replace with ReLU:** Simpler polynomial (2× reduction)

**Recommendation:** For ZK, operate FluidElite as pure linear reservoir.

---

## Overview

A production-ready tensor-accelerated ZK proving stack that targets **real bottlenecks** with **practical speedups**.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ontic Engine ZK Stack                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Proof Aggregation           [5× speedup for IVC]     │
│           └─ QTT-compressed accumulators for Nova/HyperNova    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Commitment (MSM)             [10× for structured]    │
│           └─ Tensor-structured Pippenger bucket aggregation    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Polynomial Arithmetic        [1× - already optimal]  │
│           └─ Standard GPU NTT (memory-bound, can't improve)    │
│           └─ RNS for multi-prime fields                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Field Arithmetic             [20-100× for batched]   │
│           └─ Tensor core INT8 RNS decomposition                │
│           └─ Montgomery representation                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Tensor Core Field Arithmetic

### The Opportunity

Modern GPUs have 300+ TOPS of INT8 tensor core compute, but ZK uses 256-bit modular arithmetic. Bridge this gap with RNS decomposition.

### Architecture

```
256-bit field element
        │
        ▼
┌───────────────────────────────────────┐
│  RNS Decomposition (32-40 primes)     │
│  Each residue: 8-bit                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Tensor Core GEMM (INT8 × INT8)       │
│  330 TOPS on RTX 5070                 │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  CRT Reconstruction                   │
│  ~10% overhead                        │
└───────────────────────────────────────┘
        │
        ▼
256-bit result
```

### Implementation Plan

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| 1.1 | Select 32-40 NTT-friendly 8-bit primes | 1 day | Foundation |
| 1.2 | Implement RNS decomposition kernel | 2 days | Core |
| 1.3 | Batched INT8 GEMM via cuBLAS | 2 days | Core |
| 1.4 | CRT reconstruction kernel | 2 days | Core |
| 1.5 | Montgomery integration | 3 days | Optimization |
| 1.6 | Benchmark vs standard field mul | 1 day | Validation |

### Expected Performance

```python
# Batch of 10,000 field multiplications
# Standard: 10,000 × 1μs = 10 ms
# Tensor core: 
#   - Decompose: 0.1 ms
#   - GEMM: 0.01 ms  
#   - Reconstruct: 0.5 ms
#   - Total: ~0.6 ms
# Speedup: 16×
```

### Key Files

```
ontic/cuda/
├── tensor_field.py          # Main interface
├── rns_decompose.cu         # CUDA decomposition kernel
├── rns_reconstruct.cu       # CUDA CRT kernel  
└── field_ops.py             # High-level field operations
```

---

## Layer 2: Polynomial Arithmetic

### The Reality

NTT is memory-bound. Tensor tricks don't help. Use standard implementations.

### Architecture

```python
class PolynomialRing:
    """Standard polynomial operations with RNS for large primes."""
    
    def ntt(self, coeffs: Tensor) -> Tensor:
        """cuFFT-style radix-2 NTT. Memory-bound, already optimal."""
        
    def intt(self, evals: Tensor) -> Tensor:
        """Inverse NTT with N^{-1} scaling."""
        
    def mul(self, a: Tensor, b: Tensor) -> Tensor:
        """Polynomial multiply via NTT: O(n log n)."""
        
    def div(self, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """Polynomial division with remainder."""
```

### Implementation Plan

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| 2.1 | Wrap cuFFT for power-of-2 NTT | 2 days | Core |
| 2.2 | RNS multi-channel NTT | 2 days | Large primes |
| 2.3 | Twiddle factor precomputation | 1 day | Optimization |
| 2.4 | Coset NTT for Plonk | 2 days | Compatibility |

### Key Files

```
ontic/poly/
├── ntt.py                   # GPU NTT implementation
├── polynomial.py            # Polynomial ring operations
└── coset.py                 # Coset FFT for Plonk
```

---

## Layer 3: MSM (Multi-Scalar Multiplication)

### The Opportunity

MSM dominates ZK proving time. When scalars have structure (polynomial evaluations, Lagrange bases), tensor methods apply.

### Standard Pippenger

```
G = Σᵢ sᵢ × Pᵢ

1. Decompose scalars into windows
2. Sort points into buckets by window value
3. Aggregate buckets (many curve additions)
4. Combine windows with doubling
```

### Tensor-Structured Improvement

When scalars `sᵢ = f(ωⁱ)` for low-degree polynomial `f`:

```
┌─────────────────────────────────────────────────────────┐
│  Standard: N scalars → N bucket insertions             │
├─────────────────────────────────────────────────────────┤
│  Structured: Scalars form rank-r tensor                │
│  Bucket aggregation becomes tensor contraction         │
│  Effective work: O(N × r / N) = O(r) per bucket        │
└─────────────────────────────────────────────────────────┘
```

### Implementation Plan

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| 3.1 | Standard Pippenger baseline | 5 days | Foundation |
| 3.2 | Detect scalar structure (rank estimation) | 3 days | Routing |
| 3.3 | Tensor bucket aggregation | 5 days | Core speedup |
| 3.4 | BLS12-381 curve operations | 3 days | Production |
| 3.5 | BN254 curve operations | 2 days | Ethereum |

### Expected Performance

```python
# MSM with 2^24 points
# Random scalars: 300 ms (no improvement possible)
# Structured scalars (rank 64): 
#   - Detect structure: 5 ms
#   - Tensor bucket agg: 25 ms
#   - Curve work: 10 ms
#   - Total: 40 ms
# Speedup: 7.5× for structured, 1× for random
```

### Key Files

```
ontic/msm/
├── pippenger.py             # Standard implementation
├── structured_msm.py        # Tensor-accelerated version
├── bucket_tensor.py         # Tensor bucket operations
└── curves/
    ├── bls12_381.py
    └── bn254.py
```

---

## Layer 4: Proof Aggregation

### The Opportunity

Recursive proofs (Nova, HyperNova, ProtoStar) accumulate state across folds. This state often has structure that compresses.

### QTT Accumulators

```
┌─────────────────────────────────────────────────────────┐
│  Nova Folding Step                                      │
├─────────────────────────────────────────────────────────┤
│  Standard:                                              │
│    Accumulator A ∈ F^N stored densely                   │
│    Memory: O(N) per fold                                │
│    Folds before OOM: ~1000 (for N=2^24)                │
├─────────────────────────────────────────────────────────┤
│  QTT-Compressed:                                        │
│    Accumulator A as rank-r QTT                          │
│    Memory: O(n × r²) where N = 2^n                      │
│    Compression: 100-1000× for structured witnesses     │
│    Folds before OOM: 100,000+                          │
└─────────────────────────────────────────────────────────┘
```

### Implementation Plan

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| 4.1 | Port Nova accumulator to QTT | 5 days | Core |
| 4.2 | QTT arithmetic (add, scalar mul) | 3 days | Core |
| 4.3 | QTT-native cross-term computation | 5 days | Performance |
| 4.4 | Adaptive rank selection | 3 days | Efficiency |
| 4.5 | HyperNova multifolding support | 5 days | Advanced |

### Expected Performance

```python
# Nova folding (circuit size 2^20)
# Standard: 
#   - Memory per fold: 8 MB
#   - Compute: 100 ms
#   - Max folds in 16 GB: 2000
#
# QTT-Compressed (rank 32):
#   - Memory per fold: 80 KB (100× less)
#   - Compute: 80 ms (faster due to cache)
#   - Max folds in 16 GB: 200,000
# 
# Speedup: 1.25× per fold, 100× depth capacity
```

### Key Files

```
ontic/folding/
├── nova.py                  # Nova with QTT accumulators
├── hypernova.py             # HyperNova extension
├── qtt_accumulator.py       # QTT-specific accumulator ops
└── cross_terms.py           # Tensor-native cross-term
```

---

## Integration: Full Proving Stack

### Supported Proof Systems

| System | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Expected Speedup |
|--------|---------|---------|---------|---------|------------------|
| Groth16 | ✅ | ✅ | ✅ | ❌ | 1.3× |
| Plonk | ✅ | ✅ | ✅ | ❌ | 1.3× |
| Nova | ✅ | ✅ | ⚠️ | ✅ | 4× |
| HyperNova | ✅ | ✅ | ✅ | ✅ | 5× |
| ProtoStar | ✅ | ✅ | ✅ | ✅ | 5× |

### API Design

```python
from physics_os import ZKProver, Config

# Configure stack
config = Config(
    field="bls12_381",
    use_tensor_cores=True,     # Layer 1
    use_qtt_accumulators=True, # Layer 4
    max_qtt_rank=64,
)

# Create prover
prover = ZKProver("nova", config)

# Prove
proof = prover.prove(witness, public_inputs)

# Verify
assert prover.verify(proof, public_inputs)
```

---

## Milestones

### Phase 1: Foundation (Weeks 1-3)
- [ ] Layer 1: Tensor core field arithmetic
- [ ] Layer 2: Standard GPU NTT
- [ ] Benchmarks vs baseline

### Phase 2: MSM (Weeks 4-6)
- [ ] Layer 3: Standard Pippenger
- [ ] Layer 3: Structure detection
- [ ] Layer 3: Tensor bucket aggregation

### Phase 3: Recursion (Weeks 7-10)
- [ ] Layer 4: QTT accumulators for Nova
- [ ] Layer 4: Cross-term optimization
- [ ] Layer 4: HyperNova support

### Phase 4: Production (Weeks 11-12)
- [ ] Full integration testing
- [ ] Documentation
- [ ] Benchmark suite
- [ ] Release

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Field mul throughput | 20× vs baseline | ops/sec |
| MSM (structured) | 10× vs baseline | ms for 2^24 |
| Nova fold time | 4× vs baseline | ms/fold |
| Nova fold depth | 100× vs baseline | max folds in 16GB |
| Memory efficiency | 50× compression | bytes/proof |

---

## Dependencies

### Required
- CUDA 12.0+
- cuBLAS (tensor core GEMM)
- PyTorch 2.0+ (tensor operations)
- Python 3.10+

### Optional
- cuFFT (can use custom NTT instead)
- NCCL (multi-GPU, future)

---

## Repository Structure

```
HyperTensor-VM-main/
├── ontic/
│   ├── cuda/
│   │   ├── tensor_field.py      # Layer 1
│   │   └── kernels/             # CUDA kernels
│   ├── poly/
│   │   └── ntt.py               # Layer 2
│   ├── msm/
│   │   └── pippenger.py         # Layer 3
│   └── folding/
│       └── nova.py              # Layer 4
├── tests/
│   ├── test_field.py
│   ├── test_ntt.py
│   ├── test_msm.py
│   └── test_nova.py
├── benchmarks/
│   └── prove_benchmark.py
├── docs/
│   ├── QTT_NTT_FINDINGS.md      # Lessons learned
│   └── HYPERTENSOR_ZK_STACK.md  # This document
└── archive/
    └── rns_qtt_ntt_butterfly_mpo_deprecated.py
```

---

## Next Steps

1. **Immediate:** Implement Layer 1 (tensor core field arithmetic) - highest ROI
2. **Week 2:** Validate with benchmark against arkworks/blstrs
3. **Week 3:** Begin Layer 3 (MSM) foundation
4. **Parallel:** Research QTT accumulator math for Layer 4

---

## References

- [Nova: Recursive SNARKs without trusted setup](https://eprint.iacr.org/2021/370)
- [HyperNova: Recursive arguments for R1CS](https://eprint.iacr.org/2023/573)
- [Pippenger's algorithm](https://cr.yp.to/papers/pippenger.pdf)
- [cuBLAS tensor core GEMM](https://docs.nvidia.com/cuda/cublas/)
- [RNS in cryptography](https://eprint.iacr.org/2016/1066)
