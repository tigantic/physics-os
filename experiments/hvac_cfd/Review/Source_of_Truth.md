# HVAC CFD — Source of Truth

```
██╗  ██╗██╗   ██╗ █████╗  ██████╗     ██████╗███████╗██████╗ 
██║  ██║██║   ██║██╔══██╗██╔════╝    ██╔════╝██╔════╝██╔══██╗
███████║██║   ██║███████║██║         ██║     █████╗  ██║  ██║
██╔══██║╚██╗ ██╔╝██╔══██║██║         ██║     ██╔══╝  ██║  ██║
██║  ██║ ╚████╔╝ ██║  ██║╚██████╗    ╚██████╗██║     ██████╔╝
╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝     ╚═════╝╚═╝     ╚═════╝ 
        C R I T I C A L   I N F R A S T R U C T U R E   C F D
```

**Created**: January 6, 2026  
**Version**: 0.1.0  
**Authority**: Principal Investigator  
**Status**: 🏗️ **FOUNDATION PHASE**

---

## Preamble

This document serves as the **canonical source of truth** for all HVAC CFD development within Project HyperTensor. Every design decision, implementation pattern, and optimization strategy discovered during development SHALL be recorded here.

**High-end clients with critical dependencies demand elite engineering.**

---

## Part I: Inviolable Performance Mandates

### §1 — Dense is Anti-QTT

**"Dense" representations are the antithesis of QTT philosophy.**

- Dense tensors consume $O(N)$ memory where QTT achieves $O(r \log N)$
- Dense operations bypass the entire compression advantage of tensor trains
- **DO NOT** use dense representations unless:
  1. Explicit written justification exists
  2. No QTT-native alternative is mathematically feasible
  3. The operation is provably $O(1)$ in the pipeline (e.g., final scalar extraction)

**Violation**: Using `.to_dense()` or equivalent without documented exception.

---

### §2 — Python Loops Are Performance Poison

**Python loops are interpreted overhead that destroys throughput.**

- Every `for` loop over tensor elements is a code smell
- Prefer:
  1. **Vectorized PyTorch/NumPy operations** — C++/CUDA backend
  2. **`torch.einsum`** — Optimal contraction ordering
  3. **Batched operations** — Amortize kernel launch overhead
  4. **Custom CUDA kernels** — When PyTorch primitives are insufficient

**Optimization Hierarchy**:
```
CUDA kernel > torch.compile > einsum > broadcasting > list comprehension > for loop
                                                                            ↑
                                                                      NEVER HERE
```

**Violation**: Python `for`/`while` loops iterating over tensor dimensions without profiling justification.

---

### §3 — rSVD Over SVD

**Randomized SVD (rSVD) SHALL replace full SVD wherever applicable.**

- Full SVD: $O(mn \cdot \min(m,n))$ — prohibitive at scale
- rSVD: $O(mn \cdot k)$ where $k \ll \min(m,n)$ — tractable

**When to use rSVD**:
- Truncation is required (we only need top-$k$ singular values)
- Matrix dimensions exceed $1024 \times 1024$
- Real-time or interactive performance is required

**When full SVD is permitted**:
- $k$ approaches $\min(m,n)$ (no truncation benefit)
- Numerical precision demands exact decomposition
- Debugging/validation against rSVD results

**Implementation**: Use `torch.svd_lowrank()` or custom rSVD with oversampling parameter $p \geq 5$.

---

### §4 — Decompression is Anti-QTT

**"Decompression" (QTT → Dense) defeats the entire purpose of tensor trains.**

- If you decompress, you lose:
  1. Memory efficiency ($O(r \log N) \to O(N)$)
  2. Computational efficiency (operations scale with $N$, not $r$)
  3. The ability to handle resolutions beyond physical memory

**Decompression requires EXPLICIT AUTHORIZATION** with:
- Written justification in code comments
- Reference to this section (§4)
- Demonstration that no QTT-native alternative exists

**Approved decompression scenarios**:
1. Final visualization output (GPU texture upload)
2. Validation against dense reference (testing only)
3. Interface with legacy non-QTT libraries (documented boundary)

---

### §5 — CUDA Tensor-to-Tensor Architecture

**Custom CUDA wiring for QTT tensor-to-tensor operations may be optimal.**

#### Design Philosophy

Before implementing ANY custom CUDA kernel:

1. **Profile First**: Measure PyTorch baseline with `torch.profiler`
2. **Identify Bottleneck**: Is it memory bandwidth? Compute? Kernel launch?
3. **Batch Aggressively**: Amortize launch overhead across operations
4. **Think Deeply**: Premature CUDA is worse than no CUDA

#### Batching Strategy

HVAC simulations involve:
- Multiple rooms/zones (batch dimension)
- Multiple timesteps (temporal batching)
- Multiple physical fields (velocity, temperature, pressure, contaminants)

**Design for batched tensor-to-tensor operations from day one.**

#### Memory Layout Considerations

```
QTT Core Layout:
  cores[i]: (r_left, n_i, r_right)
  
Batched Layout (proposed):
  cores[i]: (batch, r_left, n_i, r_right)
```

**Kernel fusion opportunities**:
- Fuse contraction chains (avoid intermediate allocations)
- Fuse QTT arithmetic with truncation
- Fuse evaluation with colormap (for visualization)

#### Authorization Required

Custom CUDA implementations require:
1. Benchmark demonstrating >2× speedup over PyTorch
2. Test coverage matching core library standards (90%)
3. Fallback path for non-CUDA environments

---

### §6 — Compression-Rank Relationship

**Higher compression = Lower rank.**

- Compression ratio and TT-rank are **inversely related**
- Aggressive compression (high ratio) forces lower ranks, discarding fine-scale structure
- Low compression preserves detail but increases memory/compute

**Rank Selection Guidelines**:
```
Compression Level    TT-Rank Range    Use Case
─────────────────────────────────────────────────────
Extreme (>1000×)     r ≤ 4            Coarse screening, quick estimates
High (100-1000×)     r = 4-16         Production HVAC simulations
Moderate (10-100×)   r = 16-64        High-fidelity validation
Low (<10×)           r ≥ 64           Research, benchmark comparison
```

**Do not confuse compression ratio with accuracy** — a well-structured low-rank approximation can capture essential physics while a poorly chosen high-rank can miss key features.

---

### §7 — Simulation Time Requirements

**Minimum 100s simulation time required for physics to equilibrate.**

- HVAC flows are **recirculating and slow** — characteristic timescales of $\tau = L/U \approx 20$s
- Transient startup artifacts persist for $\sim 3$–$5\tau$
- **Steady-state validation requires $t \geq 100$s** of simulated physical time

**Runtime Budgeting**:
```
Physical Time    Iterations (dt=0.02s)    Purpose
─────────────────────────────────────────────────────────
t < 20s          < 1000                   Initial transient (discard)
t = 20-60s       1000-3000                Flow development
t = 60-100s      3000-5000                Approach to steady state
t > 100s         > 5000                   Validation-ready solution
```

**Wall-clock targets** (256×128 grid):
- t = 100s physical → <60s wall-clock (target)
- This implies **dt ≈ 0.02s**, **5000 iterations**, **~12ms per iteration**

**Violation**: Reporting validation metrics on solutions with $t < 100$s simulated time.

---

## Part II: Discovery Log

**Custom CUDA wiring for QTT tensor-to-tensor operations may be optimal.**

#### Design Philosophy

Before implementing ANY custom CUDA kernel:

1. **Profile First**: Measure PyTorch baseline with `torch.profiler`
2. **Identify Bottleneck**: Is it memory bandwidth? Compute? Kernel launch?
3. **Batch Aggressively**: Amortize launch overhead across operations
4. **Think Deeply**: Premature CUDA is worse than no CUDA

#### Batching Strategy

HVAC simulations involve:
- Multiple rooms/zones (batch dimension)
- Multiple timesteps (temporal batching)
- Multiple physical fields (velocity, temperature, pressure, contaminants)

**Design for batched tensor-to-tensor operations from day one.**

#### Memory Layout Considerations

```
QTT Core Layout:
  cores[i]: (r_left, n_i, r_right)
  
Batched Layout (proposed):
  cores[i]: (batch, r_left, n_i, r_right)
```

**Kernel fusion opportunities**:
- Fuse contraction chains (avoid intermediate allocations)
- Fuse QTT arithmetic with truncation
- Fuse evaluation with colormap (for visualization)

#### Authorization Required

Custom CUDA implementations require:
1. Benchmark demonstrating >2× speedup over PyTorch
2. Test coverage matching core library standards (90%)
3. Fallback path for non-CUDA environments

---

## Part II: Discovery Log

*Discoveries, insights, and lessons learned during HVAC CFD development will be recorded below.*

### Discovery Template

```markdown
### [YYYY-MM-DD] — Discovery Title

**Context**: What problem were we solving?

**Finding**: What did we discover?

**Implication**: How does this affect our design?

**Action**: What changes were made?
```

---

*This document is living. All significant discoveries SHALL be appended here.*

---

## Part III: Design Decisions

*Major architectural decisions will be recorded here with rationale.*

---

## Part IV: Validation Evidence

*Links to proofs, benchmarks, and validation artifacts will be collected here.*

---

## Part V: Client Scenarios

*The 5 escalated tier-scenarios and their solutions will be documented here.*

---

*Source of Truth — HVAC CFD — Project HyperTensor*
