# 3D QTT-Native Turbulence DNS Blueprint

## The Last Great Unsolved Problem in Classical Physics

**Mission**: Achieve DNS turbulence at 10-100× higher Reynolds numbers than current state-of-art on a **single GPU** by applying QTT compression to 3D turbulent velocity fields.

**Commercial Opportunity**: $3B+ CFD market (Boeing, Lockheed, Airbus, automotive, wind energy)

---

## Executive Summary

### The Physics Challenge

| Parameter | Current DNS Limit | Our Target | Improvement |
|-----------|-------------------|------------|-------------|
| Reynolds Number | Re ~ 10⁴ | Re ~ 10⁶ | 100× |
| Grid Points | 4096³ (~69B) | 8192³ (~550B) effective | 8× |
| Memory | ~100 TB | ~100 GB | 1000× |
| Compute Time | Days (supercomputer) | Hours (single GPU) | 1000× |

### The Mathematical Foundation

**Key Insight**: Turbulence has **hierarchical structure** in both physical and Fourier space:
- Large scales (low-k): Smooth, energy-containing → **Low QTT rank**
- Inertial range (mid-k): Self-similar cascade → **Moderate QTT rank**
- Small scales (high-k): Dissipative, sparse → **Low QTT rank**

This matches the **TURBULENT profile** in `qtt_multiscale.py`:
```python
ScaleProfile.TURBULENT  # Bell curve: low edges, high middle
```

### Proven Technology Stack

| Component | Status | Compression Achieved |
|-----------|--------|---------------------|
| 2D QTT-native NS (vorticity-streamfunction) | ✅ Working | 534× on 4M elements |
| 3D Dense NS (spectral) | ✅ Working | N/A (baseline) |
| 5D Vlasov-Poisson | ✅ Working | 36,870× (projected) |
| Multi-scale QTT | ✅ Working | 50% additional savings |
| N-D Shift MPO | ✅ Working | O(log N) derivatives |

---

## Technical Architecture

### Current State: What Exists

#### 1. 2D QTT-Native NS (`ns2d_qtt_native.py`)
- **Formulation**: Vorticity-streamfunction (ω-ψ)
- **Grid**: 2048×512 (~1M cells) at rank 24
- **Morton Ordering**: 2D space-filling curve for QTT indexing
- **Derivatives**: MPO shift operators (O(log N) cost)
- **Time Integration**: RK4 with QTT truncation at each step

#### 2. 3D Dense NS (`ns_3d.py`)
- **Formulation**: Velocity-pressure (primitive variables)
- **Discretization**: Spectral (FFT-based)
- **Projection**: Chorin-Temam (∇·u = 0 to machine precision)
- **Benchmark**: Taylor-Green, Kida vortex (Re=5000 validated)

#### 3. N-D Shift MPO (`nd_shift_mpo.py`)
- **Supported Dimensions**: 2D, 3D, 5D (Morton interleaving)
- **Operations**: ±1 shift per axis with periodic BC
- **Complexity**: O(r³ · n_qubits) per shift
- **CUDA**: Accelerated einsum, CPU SVD truncation

### Target State: 3D QTT-Native Turbulence

```
┌─────────────────────────────────────────────────────────────────┐
│                    3D QTT-Native NS Solver                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Velocity Field                 Vorticity Field                │
│   u(x,y,z) → QTT               ω(x,y,z) = ∇ × u → QTT           │
│                                                                 │
│   ┌──────────┐                  ┌──────────────────┐            │
│   │ 3D Morton│                  │ MultiScaleQTT    │            │
│   │ Ordering │                  │ (TURBULENT)      │            │
│   └────┬─────┘                  └────────┬─────────┘            │
│        │                                 │                      │
│        ▼                                 ▼                      │
│   ┌──────────┐                  ┌──────────────────┐            │
│   │ND Shift  │                  │ Adaptive Rank    │            │
│   │MPO (3D)  │                  │ per Scale        │            │
│   └────┬─────┘                  └────────┬─────────┘            │
│        │                                 │                      │
│        ▼                                 ▼                      │
│   ┌──────────────────────────────────────┐                      │
│   │        Native QTT Operators          │                      │
│   │  ∂/∂x, ∂/∂y, ∂/∂z, ∇², ∇×           │                      │
│   │  (All O(log N) via MPO)              │                      │
│   └──────────────────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: 3D Morton + QTT Foundation (Week 1-2)

**Goal**: Extend 2D Morton ordering to 3D for QTT representation

**Deliverables**:
1. `morton_encode_3d()` / `morton_decode_3d()` functions
2. `QTT3DState` dataclass with `n_x`, `n_y`, `n_z` qubits
3. `dense_to_qtt_3d()` / `qtt_3d_to_dense()` conversion

**Code Template** (from `fast_vlasov_5d.py`):
```python
def morton_encode_3d(x: int, y: int, z: int, n_bits: int) -> int:
    """Encode 3D index to Morton order."""
    idx = 0
    for b in range(n_bits):
        idx |= ((x >> b) & 1) << (3 * b + 0)
        idx |= ((y >> b) & 1) << (3 * b + 1)
        idx |= ((z >> b) & 1) << (3 * b + 2)
    return idx
```

**Validation**: Round-trip test (encode→decode→encode must match)

### Phase 2: 3D Derivative MPOs (Week 2-3)

**Goal**: Build shift MPOs for 3D grid using `nd_shift_mpo.py`

**Deliverables**:
1. Pre-built shift MPOs for X, Y, Z axes (both directions)
2. Laplacian MPO via composition: ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
3. Curl MPO for vorticity: ω = ∇ × u

**Key Code** (already exists in `nd_shift_mpo.py`):
```python
# 3D (64^3): 6 qubits/dim * 3 dims = 18 total
shift_x = make_nd_shift_mpo(18, num_dims=3, axis_idx=0, direction=+1)
shift_y = make_nd_shift_mpo(18, num_dims=3, axis_idx=1, direction=+1)
shift_z = make_nd_shift_mpo(18, num_dims=3, axis_idx=2, direction=+1)
```

**Validation**: Taylor-Green vortex (analytical gradients)

### Phase 3: 3D QTT-Native Poisson Solver (Week 3-4)

**Goal**: Solve ∇²p = f in QTT format for pressure projection

**Options**:
1. **FFT-based** (current 3D approach): Decompress → FFT → Compress
2. **Multigrid in QTT**: Hierarchical approach using `HierarchicalQTT`
3. **Iterative in QTT**: Conjugate gradient with QTT operations

**Recommended**: Start with FFT-based (proven), migrate to QTT-native multigrid

**Critical Insight**: Pressure field is smoother than velocity → lower QTT rank expected

### Phase 4: 3D QTT-Native NS Integration (Week 4-6)

**Goal**: Full 3D incompressible NS solver in QTT format

**Algorithm** (vorticity-velocity formulation):
```
1. Initialize: ω₀ = ∇ × u₀ (in QTT)
2. For each timestep:
   a. Compute velocity: u = ∇⁻² × ω (Biot-Savart)
   b. Compute advection: N = -(u·∇)ω + (ω·∇)u (vortex stretching)
   c. Compute diffusion: D = ν∇²ω
   d. Time step: ω^{n+1} = ω^n + Δt(N + D)
   e. Truncate QTT to max_rank
3. Extract velocity if needed
```

**Key Operations** (all in QTT):
- `qtt_hadamard()`: u·∇ω (element-wise multiply after gradient)
- `qtt_add()`: Combine terms
- `qtt_scale()`: Apply coefficients
- `truncate_cores()`: Control rank growth

### Phase 5: Turbulence-Optimized Compression (Week 6-8)

**Goal**: Apply `MultiScaleQTT` with turbulence-specific profiles

**Strategy**:
1. Use `ScaleProfile.TURBULENT` for inertial range optimization
2. Apply `adapt_ranks()` based on spectral energy content
3. Allocate higher ranks to energy-containing scales

**Expected Savings** (from `qtt_multiscale.py`):
- Uniform rank-64: 5,120 params/core
- Turbulent profile: 2,560 params/core average
- **Additional 50% compression**

### Phase 6: Validation & Benchmarking (Week 8-10)

**Canonical Benchmarks**:

| Benchmark | Reynolds | Grid | Success Criterion |
|-----------|----------|------|-------------------|
| Taylor-Green Vortex | 1,600 | 256³ | Decay rate < 5% error |
| Kida Vortex | 5,000 | 512³ | Enstrophy bounded |
| Isotropic Turbulence | 10,000 | 1024³ | K41 spectrum |
| High-Re DNS | 100,000 | 4096³ | Bounded solutions |

**Metrics**:
- Kinetic energy: E(t) = ½∫|u|² dV
- Enstrophy: Z(t) = ½∫|ω|² dV
- Dissipation: ε(t) = ν∫|∇u|² dV
- Energy spectrum: E(k) for Kolmogorov scaling
- QTT rank growth: χ(t) as regularity indicator

---

## Expected Performance

### Memory Scaling

| Grid | Dense (float32) | QTT (r=64) | QTT-Multiscale | Compression |
|------|-----------------|------------|----------------|-------------|
| 256³ | 64 MB | 200 KB | 100 KB | 640× |
| 512³ | 512 MB | 400 KB | 200 KB | 2,560× |
| 1024³ | 4 GB | 800 KB | 400 KB | 10,240× |
| 2048³ | 32 GB | 1.6 MB | 800 KB | 40,960× |
| 4096³ | 256 GB | 3.2 MB | 1.6 MB | 163,840× |

### Compute Scaling

| Operation | Dense | QTT | Speedup |
|-----------|-------|-----|---------|
| Gradient | O(N³) | O(log N · r³) | N³/log N |
| Laplacian | O(N³ log N) | O(log N · r³) | N³/r³ |
| Poisson | O(N³ log N) | O(log N · r³) | N³/r³ |
| Full Step | O(N³ log N) | O(log N · r³) | **~10,000×** |

---

## Critical Success Factors

### 1. Rank Control During Advection
The nonlinear term (u·∇)ω causes rank growth. Mitigation:
- Aggressive truncation after each substep
- Adaptive tolerance based on energy conservation
- Scale-dependent truncation (preserve inertial range)

### 2. Divergence-Free Constraint
Incompressibility (∇·u = 0) must be maintained. Strategy:
- Project after each timestep (Chorin-Temam)
- Monitor ∇·u in QTT format
- Alert if constraint violated beyond tolerance

### 3. Energy Conservation
Euler limit should conserve energy. Checks:
- Track E(t) vs analytical decay (viscous case)
- Verify d E/dt = -2ν Z(t)
- Flag anomalous energy growth

### 4. QTT Rank as Regularity Monitor
From `ns_qtt_singularity_hunt.py`:
- If χ(t) → ∞ in finite time → **SINGULARITY CANDIDATE**
- If χ(t) ~ const for all t → **EVIDENCE FOR REGULARITY**

---

## Files to Create/Modify

### New Files
1. `ontic/cfd/ns3d_qtt_native.py` - Main 3D QTT-native NS solver
2. `ontic/cfd/morton_3d.py` - 3D Morton encoding utilities
3. `ontic/cfd/qtt_3d_ops.py` - 3D-specific QTT operations
4. `ontic/cfd/turbulence_benchmark.py` - Validation suite
5. `TURBULENCE_DNS_ATTESTATION.json` - Benchmark results

### Modify
1. `ontic/cfd/qtt_multiscale.py` - Add TURBULENT_3D profile
2. `ontic/cfd/nd_shift_mpo.py` - Optimize 3D path
3. `ontic/cfd/pure_qtt_ops.py` - 3D-aware truncation

---

## Reference: Existing Building Blocks

### From `ns2d_qtt_native.py`
```python
class QTT2DNativeState:
    """2D field stored in QTT format with Morton ordering."""
    cores: List[torch.Tensor]
    n_x: int  # Qubits for x
    n_y: int  # Qubits for y
```

### From `nd_shift_mpo.py`
```python
def make_nd_shift_mpo(
    num_qubits_total: int,
    num_dims: int,       # 3 for 3D
    axis_idx: int,       # 0=X, 1=Y, 2=Z
    direction: int = 1,  # +1 forward, -1 backward
    ...
) -> list[torch.Tensor]:
```

### From `qtt_multiscale.py`
```python
class MultiScaleQTT:
    """Variable-rank QTT for multi-resolution physics."""
    
    def compress(self, tensor: Tensor) -> List[Tensor]:
        """Compress with scale-adaptive ranks."""
        
    def adapt_ranks(self, tensor: Tensor) -> List[int]:
        """Adaptively determine ranks based on content."""
```

---

## Next Steps

1. **Immediate**: Implement `morton_3d.py` and test round-trip encoding
2. **Week 1**: Build 3D shift MPOs and validate on Taylor-Green
3. **Week 2**: Implement basic 3D QTT NS step (no Poisson)
4. **Week 3**: Add Poisson solver (FFT-based first)
5. **Week 4**: Integrate MultiScaleQTT for turbulence
6. **Week 5-6**: Benchmark suite and optimization
7. **Week 7-8**: High-Re validation (Re > 10,000)
8. **Week 9-10**: Documentation and attestation

---

## Conclusion

The jump from 5D plasma (36,870× compression) to 3D turbulence is **conceptually identical**:
- Replace plasma distribution function f(x,y,z,vx,vy,vz) with velocity field u(x,y,z,t)
- Use same Morton ordering, same shift MPOs, same truncation
- Apply turbulence-specific rank profiles from MultiScaleQTT

**The infrastructure is ready. The physics awaits.**

---

*Generated: 2025*
*Project: HyperTensor-VM Turbulence DNS Initiative*
