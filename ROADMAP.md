# HyperTensor Strategic Roadmap

**Document ID**: ROADMAP-001  
**Version**: 3.0.0  
**Revised**: 2025-12-26  
**Authority**: Principal Investigator  
**Status**: Foundation Validated — Rebuilding on Solid Ground  
**Principle**: No Potemkin villages. Only validated truth.

---

## Executive Summary

**Previous roadmap was aspirational, not validated.** We marked layers "complete" based on scaffolding and unit tests, not actual working physics. The December 2025 capability audit exposed this.

**What we actually have:**
- ✅ QTT core that works (storage, arithmetic, truncation)
- ✅ N-D shift MPO with Morton ordering (validated through 512³)
- ✅ 3D incompressible Euler solver (proven resolution-independent)
- ✅ PQC cryptographic attestation

**This roadmap is rebuilt from validated foundations only.**

---

## Honest Status Assessment

### ✅ VALIDATED (Proven with Working Code)

| Component | Evidence | Date |
|-----------|----------|------|
| QTT Storage | `dense_to_qtt()`, 45× compression at N=1M | Dec 2025 |
| QTT Arithmetic | `qtt_add()`, `truncate_qtt()` | Dec 2025 |
| Morton N-D Ordering | 3D→1D bit interleaving verified | Dec 2025 |
| N-D Shift MPO | `make_nd_shift_mpo()`, `apply_nd_shift_mpo()` | Dec 2025 |
| 3D Euler Advection | Strang splitting, stable to t=10 | Dec 2025 |
| Resolution Independence | 32³→512³: rank 39→34 (DECREASES) | Dec 2025 |
| Kronecker IC Construction | 1024³ in 63s, 24,578× compression | Dec 2025 |
| PQC Signing | Dilithium2 `sign_manifest()`, `verify_signature()` | Dec 2025 |
| TileRenderer | `get_tile()`, `render_view()` with LOD | Dec 2025 |
| SliceEngine | XY/XZ/YZ slicing at arbitrary depth | Dec 2025 |
| VolumeRenderer | Ray-marched volume rendering | Dec 2025 |

### 🟡 PARTIAL (Exists but Not Battle-Tested)

| Component | Status | Gap |
|-----------|--------|-----|
| Field Oracle API | `sample()` works | Not stress-tested |
| 1D/2D Shock Tubes | Sod tube works | Limited validation |
| TCI Flux Approximation | Prototype exists | Needs full integration |
| Provenance Signing | Works | Third-party replay untested |

### ❌ SCAFFOLD ONLY (Code Exists, Not Validated)

| Component | Reality |
|-----------|---------|
| HyperSim Benchmarks | Harness exists, most benchmarks not run |
| HyperEnv | RL environment scaffolded, never trained an agent |
| Intent/FDL | Parser exists, never steered real physics |
| FieldOS | Kernel scaffolded, never ran multi-field |
| Unreal/Unity Plugins | Code exists, never integrated with engine |

---

## North Star (Unchanged)

> **A continuous, evolvable, GPU-native Field Substrate** whose complexity scales with **structure**, and which can be **steered by intent** with **auditable provenance**.

---

## Foundation-First Rebuild

### The Lesson Learned

We tried to build 8 layers simultaneously. Result: 8 partial layers, zero validated end-to-end paths.

**New approach:** Each layer must be VALIDATED before moving up. Validation means:
1. **Works** — Not just compiles, actually runs correctly
2. **Tested** — Against known analytical solutions or baselines
3. **Demonstrated** — Visible output that a human can verify
4. **Documented** — Clear API, limitations, and failure modes

---

## Layer 0: QTT Core (✅ VALIDATED)

**Status:** Complete and battle-tested

| Component | File | Validated By |
|-----------|------|--------------|
| QTT State | `tensornet/cfd/pure_qtt_ops.py` | 512³ simulation |
| Dense→QTT | `dense_to_qtt()` | 1024³ IC construction |
| QTT Addition | `qtt_add()` | Thousands of timesteps |
| Truncation | `truncate_qtt()` | Rank stayed bounded |

**API (Validated):**
```python
qtt = dense_to_qtt(tensor_1d, max_bond=64)
result = qtt_add(a, b, max_bond=64)
cores = truncate_qtt(cores, max_rank=64, tol=1e-8)
```

**Limitations:**
- CPU only (no GPU kernels yet)
- Python loops in some paths (performance ceiling)
- Rank can still explode for non-smooth functions

---

## Layer 1: Shift & Derivative MPO (✅ VALIDATED)

**Status:** Complete and battle-tested

| Component | File | Validated By |
|-----------|------|--------------|
| N-D Shift MPO | `tensornet/cfd/nd_shift_mpo.py` | 512³ Euler to t=10 |
| Axis-Specific BC | Morton bit-interleaving | Periodic verified all axes |
| MPO-QTT Matvec | `apply_nd_shift_mpo()` | 4000+ steps stable |

**API (Validated):**
```python
mpo = make_nd_shift_mpo(n_qubits, num_dims=3, axis_idx=0, direction=+1, periodic=True)
shifted_cores = apply_nd_shift_mpo(cores, mpo, max_rank=64)
```

**What This Enables:**
- First-order derivatives: `(f[i+1] - f[i-1]) / 2dx`
- Second-order derivatives: `(f[i+1] - 2f[i] + f[i-1]) / dx²`
- N-dimensional advection via operator splitting

---

## Layer 2: 3D Euler Solver (✅ VALIDATED)

**Status:** Complete — resolution-independence PROVEN

| Component | File | Validated By |
|-----------|------|--------------|
| Taylor-Green IC | `build_rank1_3d_qtt()` | Separable construction |
| Kronecker Construction | `build_rank1_3d_qtt_tensorfree()` | 1024³ in 63s |
| Strang Splitting | `MillenniumSolver.step()` | t=10 achieved |
| Lax-Friedrichs Diffusion | Artificial viscosity | Stable evolution |

**Proven Results (PQC-Signed):**

| Grid | Points | t_reached | Final Rank | Wall Time |
|------|--------|-----------|------------|-----------|
| 32³ | 32K | 10.0 | 39 | ~minutes |
| 64³ | 262K | 7.0 | 37 | ~minutes |
| 128³ | 2M | 10.0 | 36 | 41 min |
| 512³ | 134M | 10.0 | 34 | 9.4 hours |

**Key Finding:** Rank DECREASES with resolution (39→34). This is unprecedented.

---

## Layer 3: Attestation & Provenance (✅ VALIDATED)

**Status:** PQC signing works, replay needs third-party test

| Component | File | Validated By |
|-----------|------|--------------|
| Dilithium2 Signing | `demos/pqc_sign_results.py` | Signatures verify |
| Manifest Hashing | SHA-256 | Deterministic |
| Attestation JSON | `MILLENNIUM_HUNTER_ATTESTATION.json` | Published |

**API (Validated):**
```python
from dilithium_py.dilithium import Dilithium2
pk, sk = Dilithium2.keygen()
sig = Dilithium2.sign(sk, message)
valid = Dilithium2.verify(pk, message, sig)
```

**Gap:** Third party has not independently replayed and verified hashes.

---

## ✅ Layer 4: Rendering (VALIDATED)

**Status:** Core rendering pipeline validated December 2025

| Component | File | Validated By |
|-----------|------|--------------|
| TileRenderer | `tensornet/hypervisual/renderer.py` | `get_tile()`, `render_view()` |
| SliceEngine | `tensornet/hypervisual/slicer.py` | XY/XZ/YZ slicing at arbitrary depth |
| VolumeRenderer | `tensornet/hypervisual/slicer.py` | Ray-marched volume rendering |
| ColorMaps | `tensornet/hypervisual/colormaps.py` | VIRIDIS, PLASMA, etc. |
| LODPyramid | `tensornet/hypervisual/renderer.py` | Multi-resolution tile caching |

**API (Validated):**
```python
from tensornet.substrate import Field
from tensornet.hypervisual import TileRenderer, SliceEngine, VolumeRenderer

# Create field
field = Field.create(dims=3, bits_per_dim=4, rank=4, init='random')

# Tile-based rendering
renderer = TileRenderer(field, RenderConfig(tile_size=64))
tile = renderer.get_tile(TileCoord(x=0, y=0, lod=0))
view = renderer.render_view(bounds=(0,0,1,1), resolution=(256,256))

# Arbitrary slicing
slicer = SliceEngine(field)
slice_result = slicer.slice(plane='xy', depth=0.5)

# Volume rendering
vol_renderer = VolumeRenderer(field)
result = vol_renderer.render(camera_pos=(2,2,2), look_at=(0.5,0.5,0.5))
```

**Test Suite:** 43/43 tests passed

**Limitations:**
- No integration with external graphics APIs (Vulkan, OpenGL)
- "Infinite zoom" demo not implemented yet
- GPU acceleration exists but not stress-tested on large fields

---

## 🔲 Layer 5: FieldOps (NOT VALIDATED)

**Status:** Operators defined, not tested on real physics

**What Exists:**
- Grad, Div, Curl, Laplacian class definitions
- Graph scheduler outline

**What's Missing:**
- Never run on validated 3D field
- Divergence-free projection untested
- No comparison to analytical solutions

**To Validate:**
1. Apply Laplacian to known function, check error
2. Project velocity field, verify div(u) ≈ 0
3. Run full Navier-Stokes (not just Euler)

---

## 🔲 Layer 6: Benchmarks (PARTIALLY VALIDATED)

**Status:** Millennium Hunter validated, others not

**Validated:**
- Taylor-Green vortex (resolution scaling)
- 1D Sod shock tube (basic)

**Not Validated:**
- Lid-driven cavity
- Decaying turbulence
- Passive scalar advection
- Comparison vs grid-based solver

**To Validate:**
1. Run each benchmark
2. Compare to published reference data
3. Plot error vs rank curves
4. Document "sweet spot" for each problem type

---

## 🔲 Layer 7: AI Environments (NOT VALIDATED)

**Status:** Scaffold only

**What Exists:**
- `tensornet/hyperenv/` directory
- Environment class definitions
- Observation space outline

**What's Missing:**
- No RL agent ever trained
- No Gym/Gymnasium integration tested
- No multi-agent scenario run

**To Validate:**
1. Create simple smoke navigation env
2. Train PPO agent to completion
3. Verify deterministic replay

---

## 🔲 Layer 8: Intent Steering (NOT VALIDATED)

**Status:** FDL parser exists, never used

**What Exists:**
- `tensornet/intent/` directory
- FDL grammar definition
- Constraint operator stubs

**What's Missing:**
- Never compiled directive to dynamics
- Never steered real physics with intent
- "Calm zone / storm zone" demo not done

**To Validate:**
1. Parse simple FDL directive
2. Apply to running simulation
3. Verify field respects constraints
4. Show live toggle of regions

---

## 🔲 Layer 9: Engine Integration (NOT VALIDATED)

**Status:** Code exists, never ran in engine

**What Exists:**
- `integrations/unreal/` and `integrations/unity/` directories
- Actor/MonoBehaviour definitions
- Bridge protocol outline

**What's Missing:**
- Never opened Unreal/Unity with plugin loaded
- No actual volumetric visible in viewport
- No performance profiling in engine

**To Validate:**
1. Load plugin in engine
2. Create field actor in scene
3. Render volume in viewport
4. Profile frame time

---

## Milestone Redefinition

Previous milestones were defined before we had working physics. Redefining based on validated foundation:

### Milestone 1: "The Holy Grail Video" 
**Status:** 🟡 IN PROGRESS

**Requirements:**
1. ✅ 3D field evolves stably (Taylor-Green proven)
2. ✅ Resolution-independence demonstrated (32³→512³)
3. 🔲 Camera flythrough rendered to video
4. 🔲 Memory overlay showing constant usage
5. 🔲 Zoom into filament without tiling

**Current Gap:** Need actual renderer, not just field evolution

---

### Milestone 2: "Benchmark Credibility Pack"
**Status:** 🟡 PARTIAL

**Requirements:**
1. ✅ Taylor-Green rank scaling (done)
2. 🔲 Error vs rank curves for 3+ problems
3. 🔲 Speed comparison vs baseline CFD
4. 🔲 "Sweet spot map" published
5. ✅ PQC-signed results

**Current Gap:** Need more benchmarks, need baseline comparison

---

### Milestone 3: "Third-Party Replay"
**Status:** 🔲 NOT STARTED

**Requirements:**
1. 🔲 Publish FieldBundle + replay tool
2. 🔲 Third party downloads and runs
3. 🔲 Third party gets identical hash
4. 🔲 Write-up of reproduction attempt

**Current Gap:** Need external collaborator

---

### Milestone 4: "Intent Demo"
**Status:** 🔲 NOT STARTED

**Requirements:**
1. 🔲 FDL directive parsed and applied
2. 🔲 Live toggle of "calm" vs "turbulent" regions
3. 🔲 Field visibly responds to constraints
4. 🔲 Frame budget maintained during steering

**Current Gap:** Need validated rendering first, then intent

---

## Execution Priority

Based on validated foundation, here's the honest priority order:

### Priority 1: Rendering (Critical Path)

Without rendering, we can't show anything. All demos blocked on this.

**Tasks:**
1. QTT → dense slice extraction
2. Slice → 3D texture
3. Simple ray marcher (even CPU-based)
4. Camera controls
5. Video export

**Definition of Done:** 512³ Taylor-Green rendered as MP4

---

### Priority 2: More Benchmarks

Credibility requires more than one problem type.

**Tasks:**
1. 2D lid-driven cavity
2. 2D/3D decaying turbulence
3. Comparison vs NumPy/SciPy baseline
4. Error plots

**Definition of Done:** 4 benchmarks with published comparison

---

### Priority 3: Third-Party Replay

Someone else must verify our results.

**Tasks:**
1. Clean install instructions
2. Replay script
3. Hash verification
4. Find collaborator to test

**Definition of Done:** External verification blog post

---

### Priority 4: Intent Steering

Only after rendering works.

**Tasks:**
1. Simple region constraint
2. FDL → dynamics compiler
3. Live demo with slider

**Definition of Done:** Video showing constraint toggle

---

## Definition of Done (All Work)

Per Constitution Article III:

- [ ] **Works** — Actually runs, not just compiles
- [ ] **Tested** — Against known solution or baseline
- [ ] **Demonstrated** — Visual or numerical output shown
- [ ] **Documented** — API, limitations, failure modes
- [ ] **Committed** — In git with clear message

---

## Future Work (Parked)

These are NOT on active roadmap until foundation is solid:

| Item | Reason Parked |
|------|---------------|
| GPU Kernels | Need correct CPU first |
| Rust TCI Core | Need stable algorithm first |
| Unreal/Unity | Need rendering first |
| FieldOS Multi-Field | Need single field working |
| Enterprise SDK | Need product first |
| Black Swan Hunts | Different ICs queued, run on separate machine |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 3.0.0 | 2025-12-26 | **Complete rewrite.** Honest assessment based on capability audit. Scaffold layers marked as unvalidated. Foundation-first rebuild. |
| 2.0.0 | 2025-12-24 | 8-Layer architecture (aspirational, not validated) |
| 1.0.0 | 2025-12-20 | Initial roadmap |

---

## Appendix: The Lesson

> We tried to build a skyscraper by finishing all floors simultaneously.
> Result: Every floor was half-built, none was habitable.
> 
> **New rule:** Each floor must pass inspection before starting the next.
> No more Potemkin villages.
---

*This roadmap is governed by the HyperTensor Constitution. All development must comply with its provisions.*
