# HyperFOAM: GPU-Native CFD for HVAC

## Strategic Moat

1. **GPU-native PyTorch solver** — 39× faster than OpenFOAM on same hardware
2. **HyperGrid tensor format** — Structured grid + immersed boundary. Keeps `torch.roll` speed while handling complex geometry. Competitors must adopt our format or spend years on their own.
3. **Differentiable by design** — Voxelization is continuous; geometry changes flow gradients. Unstructured mesh topology is discrete and breaks autograd.
4. **QTT compression** — Billion-cell grids on single GPU (nobody else has this)
5. **HVAC-specific** — Baked-in ASHRAE tables, comfort metrics, energy calcs

### ⚠️ The "Unstructured Trap" We Avoided

Switching to OpenFOAM-style polyhedral meshes would have killed our speed:
- **Structured:** `neighbor = cell[i+1]` → GPU prefetch works, 100% cache hit
- **Unstructured:** `neighbor = connectivity[i]` → Indirect access, 80% cache miss

We would have dropped from **39×** to **4-8×** speedup. Just another Fluent clone.

**Our solution:** Immersed Boundary / Fractional Volume method on structured grids.
Geometry becomes a tensor channel, not a connectivity graph.

---

## Phase 0: Foundation (COMPLETE)
**Status: ✓ DONE**

- [x] Cartesian Navier-Stokes solver
- [x] Upwind advection + central diffusion
- [x] Nielsen benchmark validated (6.7% error vs Aalborg data)
- [x] 2.1M cells @ 2000 steps/sec (float32 + torch.compile)
- [x] 39× speedup demonstrated

**Deliverables:**
- `qtt_ns_3d_v4.py` — Working Cartesian solver
- `qtt_nielsen_runner_v2.py` — Validation harness

---

## Phase 1: Voxelization & HyperGrid Format
**Timeline: 4-6 weeks**
**Goal: Convert CAD/STL geometry to GPU-native tensor format**

### ⚠️ CRITICAL ARCHITECTURAL DECISION

**WHY NOT UNSTRUCTURED MESHES:**
```
Structured:   neighbor = cell[i+1]           → GPU predicts fetch, 100% cache hit
Unstructured: neighbor = connectivity[i]    → Indirect access, 80% cache miss
```
Switching to OpenFOAM-style polyhedra would drop our speedup from **39×** to **4-8×**.
We would just be rewriting Fluent in Python — not enough to win.

**THE PIVOT:** Remain a **Structured Solver** that handles geometry via 
**Fractional Volume / Immersed Boundary**. The GPU keeps running at max FLOPs.

### 1.1 The HyperGrid Tensor Format (Week 1-2)

Instead of a graph (nodes/edges), geometry is a **4D Tensor**:

```python
# Shape: [Channels, Nx, Ny, Nz]
# All operations remain torch.roll / conv3d compatible

class HyperGrid:
    geo: Tensor  # Shape [5, Nx, Ny, Nz]
    
    # Channel 0: vol_frac   - Volume fraction (0.0=Solid, 1.0=Fluid)
    # Channel 1: area_x     - Open area fraction on X-faces (West/East)
    # Channel 2: area_y     - Open area fraction on Y-faces (South/North)  
    # Channel 3: area_z     - Open area fraction on Z-faces (Bottom/Top)
    # Channel 4: sdf        - Signed Distance to nearest wall (for wall functions)
```

**Why this wins:**
1. **Speed:** Keep using `torch.roll`. GPU runs at maximum bandwidth.
2. **Differentiability:** Voxelization is smooth. Moving a wall changes `vol_frac` 
   from 0.0→0.1 continuously — gradients flow. Unstructured topology changes are discrete.
3. **Moat:** Competitors can't just "port to PyTorch" — they need our format or 2+ years.

### 1.2 The Voxelizer (Week 2-4)

**Input:** STL file (from CAD) or OpenFOAM polyMesh (for validation)
**Process:** GPU ray-tracing / rasterization
**Output:** HyperGrid tensors

- [ ] STL parser (binary + ASCII)
- [ ] GPU raycasting for inside/outside classification
- [ ] Sub-cell sampling for accurate area fractions (anti-aliasing)
- [ ] SDF computation via jump flooding algorithm
- [ ] OpenFOAM polyMesh → sample to HyperGrid (for benchmarking vs OpenFOAM)

### 1.3 Primitive Geometry API (Week 3-4)

Fast path for common HVAC elements (no STL needed):
- [ ] `add_box(x_min, x_max, ...)` — Tables, walls, pillars
- [ ] `add_cylinder(center, radius, height)` — Ducts, columns
- [ ] `add_inlet_patch(x, y_range, z_range, velocity)` — Diffusers
- [ ] `add_outlet_patch(...)` — Returns
- [ ] Boolean operations: union, subtract, intersect

### 1.4 Validation (Week 5-6)
- [ ] Voxelize Nielsen room, compare to Cartesian baseline
- [ ] Voxelize room with table obstacle, verify vol_frac
- [ ] Benchmark: 256³ voxelization < 100ms
- [ ] Visualize in PyVista with transparency for partial cells

**Deliverables:**
- `hyper_grid.py` — HyperGrid data structure
- `voxelizer.py` — STL → HyperGrid conversion
- `primitives.py` — Box, cylinder, patch helpers
- `sdf.py` — Signed distance field computation

---

## Phase 2: Porous Media / Immersed Boundary Physics
**Timeline: 6-8 weeks**
**Goal: Modify FVM operators to respect geometry tensors**

### 2.1 Fractional Volume Navier-Stokes (Week 1-3)

The key insight: multiply fluxes by area fractions, divide by volume fractions.

```python
# Standard flux (what we have now):
flux_x = (phi[i+1] - phi[i]) / dx

# HyperGrid flux (geometry-aware):
flux_x = area_x[i] * (phi[i+1] - phi[i]) / dx
dphi_dt = (flux_x[i] - flux_x[i-1]) / (vol_frac[i] * dx)

# Solid cells (vol_frac=0) get masked out entirely
```

- [ ] Modify advection: `flux *= area_fraction`
- [ ] Modify diffusion: `flux *= area_fraction`  
- [ ] Modify time integration: `dphi_dt /= vol_frac` (with epsilon guard)
- [ ] Solid masking: zero velocity in cells where `vol_frac < threshold`

### 2.2 Wall Boundary Conditions (Week 3-5)

The SDF channel enables proper wall treatment:

```python
# Distance to wall from SDF
d_wall = grid.geo[4]  # Signed distance field

# Wall shear stress (log-law)
u_tau = kappa * u_parallel / log(E * y_plus)
tau_wall = rho * u_tau**2

# Apply as momentum source near walls
momentum_source = -tau_wall * wall_area / cell_volume
```

- [ ] No-slip from area fractions (flux blocked = no-slip)
- [ ] Wall functions using SDF for y+ calculation
- [ ] Automatic y+ field output for verification
- [ ] Slip walls (for symmetry planes)

### 2.3 Pressure-Velocity Coupling (Week 5-7)

SIMPLE algorithm adapted for fractional volumes:

```python
# Pressure Poisson with geometry:
# ∇·(area_frac * ∇p) = -∇·(rho * u) / dt

# The Laplacian stencil becomes:
lap_p = (area_x[i] * (p[i+1] - p[i]) - area_x[i-1] * (p[i] - p[i-1])) / dx²
      + (area_y[j] * (p[j+1] - p[j]) - area_y[j-1] * (p[j] - p[j-1])) / dy²
      + (area_z[k] * (p[k+1] - p[k]) - area_z[k-1] * (p[k] - p[k-1])) / dz²
```

- [ ] SIMPLE with area-weighted pressure gradients
- [ ] Rhie-Chow interpolation for cell-centered
- [ ] GPU Conjugate Gradient solver
- [ ] Multigrid preconditioner (geometric, grid-aligned)

### 2.4 Validation (Week 7-8)

- [ ] Lid-driven cavity with immersed cylinder
- [ ] Flow around cube (vs OpenFOAM with same geometry)
- [ ] Nielsen room with table obstacle
- [ ] Verify speedup remains >30× vs OpenFOAM

**Deliverables:**
- `fvm_porous.py` — Geometry-aware FVM operators
- `wall_model.py` — Wall functions using SDF
- `pressure_solver.py` — SIMPLE with fractional volumes
- `validation/immersed_cavity.py` — Benchmark case

---

## Phase 3: Turbulence Modeling
**Timeline: 4-6 weeks**
**Goal: RANS models for indoor airflow**

### 3.1 k-ε Model (Week 1-3)
```
Transport equations:
∂k/∂t + ∇·(Uk) = ∇·(ν_eff ∇k) + P_k - ε
∂ε/∂t + ∇·(Uε) = ∇·(ν_eff ∇ε) + C1 P_k ε/k - C2 ε²/k

Turbulent viscosity:
ν_t = C_μ k²/ε
```
- [ ] Standard k-ε with wall functions
- [ ] Realizable k-ε (better for jets)
- [ ] RNG k-ε (better for recirculation)

### 3.2 Wall Treatment (Week 3-4)
- [ ] Standard wall functions (y+ = 30-300)
- [ ] Enhanced wall treatment (y+ < 5)
- [ ] Automatic y+ calculation and switching

### 3.3 k-ω SST (Week 5-6)
- [ ] Menter's SST model (best for HVAC)
- [ ] Blending functions
- [ ] Low-Re damping

**Validation:**
- [ ] Nielsen benchmark with k-ε
- [ ] IEA Annex 20 test cases
- [ ] Compare to OpenFOAM results

**Deliverables:**
- `turbulence_models.py` — k-ε, k-ω SST
- `wall_functions.py` — Standard, enhanced
- `turbulence_bc.py` — Inlet turbulence intensity

---

## Phase 4: Thermal & Species Transport
**Timeline: 3-4 weeks**
**Goal: Temperature and contaminants**

### 4.1 Energy Equation (Week 1-2)
```
∂T/∂t + ∇·(UT) = ∇·(α_eff ∇T) + S_h

Buoyancy coupling:
ρ = ρ_ref (1 - β(T - T_ref))  [Boussinesq]
```
- [ ] Passive scalar transport
- [ ] Buoyancy source term in momentum
- [ ] Conjugate heat transfer (walls)

### 4.2 HVAC-Specific Features (Week 3-4)
- [ ] ASHRAE comfort metrics (PMV, PPD, DR)
- [ ] Age of air calculation
- [ ] Contaminant transport (CO2, particles)
- [ ] ACH (air changes per hour) computation

**Validation:**
- [ ] Mixed convection in cavity
- [ ] Displacement ventilation benchmark

**Deliverables:**
- `thermal.py` — Energy equation, buoyancy
- `comfort.py` — PMV, PPD, draft risk
- `air_quality.py` — Age of air, contaminants

---

## Phase 5: Differentiable Optimization
**Timeline: 3-4 weeks**
**Goal: Gradient-based design**

### 5.1 Adjoint Sensitivities (Week 1-2)
- [ ] Leverage PyTorch autograd (automatic for explicit schemes)
- [ ] Checkpointing for memory efficiency
- [ ] Sensitivity of comfort metrics to inlet conditions

### 5.2 Design Variables (Week 2-3)
- [ ] Inlet position/angle optimization
- [ ] Diffuser geometry parameters
- [ ] Supply temperature/velocity optimization

### 5.3 Optimization Loops (Week 3-4)
- [ ] Gradient descent on CFD objective
- [ ] Bayesian optimization for discrete choices
- [ ] Multi-objective (comfort vs energy)

**Deliverables:**
- `adjoint.py` — Sensitivity computation
- `optimizer.py` — Design optimization
- `objectives.py` — Comfort, energy, mixing

---

## Phase 6: Scale with QTT
**Timeline: 4-6 weeks**
**Goal: Billion-cell grids**

### 6.1 QTT Pressure Solver (Week 1-3)
- [ ] Pressure Poisson in QTT format
- [ ] TCI compression of solution
- [ ] O(log N) memory for pressure field

### 6.2 Hybrid Dense/QTT (Week 3-5)
- [ ] Dense velocity on GPU
- [ ] QTT pressure for memory efficiency
- [ ] Seamless conversion at coupling step

### 6.3 Multi-GPU Domain Decomposition (Week 5-6)
- [ ] Spatial decomposition
- [ ] Halo exchange with NCCL
- [ ] Weak scaling to 8+ GPUs

**Target:** 1 billion cells on single 80GB A100

**Deliverables:**
- `qtt_pressure.py` — QTT Poisson solver
- `distributed.py` — Multi-GPU support

---

## Phase 7: Production Hardening
**Timeline: 4-6 weeks**
**Goal: Reliable, documented, tested**

### 7.1 Testing & CI
- [ ] Unit tests for every operator
- [ ] Regression tests against OpenFOAM
- [ ] Nightly benchmarks

### 7.2 Documentation
- [ ] API reference (Sphinx)
- [ ] Theory guide (equations, discretization)
- [ ] Tutorial: room ventilation from scratch

### 7.3 CLI & GUI
- [ ] `hyperfoam run case/` command
- [ ] PyVista-based live visualization
- [ ] ParaView export

### 7.4 Packaging
- [ ] pip installable
- [ ] Docker image with CUDA
- [ ] Cloud deployment (AWS/GCP spot instances)

---

## Milestones & Success Criteria

| Milestone | Criteria | Target Date |
|-----------|----------|-------------|
| **M1: HyperGrid** | Voxelize STL, table/cylinder obstacles | +4 weeks |
| **M2: Immersed NS** | Nielsen with table obstacle, <10% error | +10 weeks |
| **M3: Turbulent** | k-ε on HyperGrid, wall functions from SDF | +16 weeks |
| **M4: Thermal** | Buoyancy, comfort metrics (PMV/PPD) | +20 weeks |
| **M5: Differentiable** | Gradient of comfort w.r.t. diffuser position | +24 weeks |
| **M6: Billion Cells** | 1B cell pressure solve via QTT + HyperGrid | +30 weeks |
| **M7: Production** | pip install, docs, validated | +36 weeks |

---

## Competitive Analysis

| Feature | OpenFOAM | Fluent | HyperFOAM |
|---------|----------|--------|-----------|
| License | Free (GPL) | $20-50k/yr | Free/Commercial |
| GPU Native | ❌ (bolted-on) | ✓ (limited) | ✓ (PyTorch) |
| Differentiable | ❌ | ❌ | ✓ (autograd) |
| Python API | ❌ (C++) | ❌ (Scheme) | ✓ (native) |
| HVAC-specific | ❌ | ❌ | ✓ (ASHRAE) |
| QTT Scale | ❌ | ❌ | ✓ (1B cells) |
| Speed (2M cells) | 1× | 5× | 39× |

---

## Resource Requirements

**Minimum:**
- 1 senior developer (you + AI assist)
- RTX 4090 or better for development
- A100 80GB for billion-cell validation

**Recommended:**
- 2-3 developers for parallel workstreams
- Small HPC cluster for regression testing
- Domain expert for HVAC validation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Pressure solver convergence | Start with direct solver, add multigrid |
| Mesh quality sensitivity | Strict quality checks at import |
| Turbulence model tuning | Validate against multiple benchmarks |
| GPU memory limits | QTT compression, out-of-core |
| Competitor response | Move fast, patent key innovations |

---

## Next Actions

1. [x] Create `hyper_grid.py` with HyperGrid data structure ✓
2. [x] Integrate HyperGrid into existing Nielsen solver ✓ (`fvm_porous.py`)
3. [x] Test with obstacle (verified: column blocks ceiling jet) ✓
4. [ ] Write STL voxelizer (GPU raycasting)
5. [ ] Add wall functions using SDF channel
6. [ ] Pressure-velocity coupling (SIMPLE algorithm)

**Phase 1.1 COMPLETE — hyper_grid.py working.**
**Phase 2.1 VALIDATED — Brinkman penalization blocks flow correctly.**

### Validation Results (2025-01-07)

| Test | Result |
|------|--------|
| Table at floor level | ✓ Velocity inside table = 0.0000 |
| Floor-to-ceiling column | ✓ Ceiling jet blocked at x=4.5m |
| Upstream velocity | ✓ Unaffected by downstream obstacle |
| Downstream velocity | ✓ Reduced after obstruction |
