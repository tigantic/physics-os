# HyperTensor Strategic Roadmap

**Document ID**: ROADMAP-001  
**Version**: 1.0.0  
**Ratified**: 2025-12-24  
**Authority**: Principal Investigator  
**Status**: Active  
**Constitution Alignment**: Article VI, Section 6.1

---

## Executive Summary

This document establishes the strategic roadmap for HyperTensor's evolution from a tensor network library to a **Reality Representation Infrastructure** — a continuous, evolvable, GPU-native Field Substrate whose complexity scales with structure and which can be steered by intent with auditable provenance.

---

## North Star

> **A continuous, evolvable, GPU-native Field Substrate** whose complexity scales with **structure**, and which can be **steered by intent** with **auditable provenance**.

---

## Guiding Principles

### The One Rule

**One substrate. One Oracle API. One Directive layer.**

Everything else is a thin skin: render, sim, AI env, provenance.

### Don't-Fragment Rules

| Rule | Rationale |
|------|-----------|
| Don't build 4 products separately | Single substrate powers all consumers |
| Don't optimize before bounded mode | Frame budgets before micro-optimizations |
| Don't chase DNS-grade accuracy | Optimize for structure + controllability + reproducibility |
| Don't expose tensor internals | Users see intent + budgets + knobs |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CONSUMERS                                      │
│    Renderer     Simulator     AI Env     Audit     Training             │
│   (HyperVisual) (HyperSim)   (HyperEnv)  (Provenance)                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                      DIRECTIVE LAYER (Layer 6)                           │
│   Intent Operators   Field Directive Language   Control Loop            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                      FIELD ORACLE API (Layer 0)                          │
│   sample(points)  slice(spec)  step(dt)  stats()  serialize()          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                      FIELDOPS (Layer 1)                                  │
│   Grad  Div  Curl  Laplacian  Advect  Diffuse  Project  Forces         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                      QTT SUBSTRATE                                       │
│   Cores    Contractions    GPU Runtime    Bounded Mode    Cache         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Definitions

### Layer 0 — Substrate (The Spine)

**Status**: ✅ COMPLETE (2025-12-24)

The undeniable core that everything hangs on.

| Component | File | Status |
|-----------|------|--------|
| Field Oracle API | `tensornet/substrate/field.py` | ✅ Complete |
| FieldStats Telemetry | `tensornet/substrate/stats.py` | ✅ Complete |
| FieldBundle Format | `tensornet/substrate/bundle.py` | ✅ Complete |
| Bounded Mode | `tensornet/substrate/bounded.py` | ✅ Complete |

**API Surface**:
```python
Field.sample(points) -> values      # O(N × d × r²)
Field.slice(spec) -> buffer         # Rendering interface
Field.step(dt) -> Field             # Physics evolution
Field.stats() -> FieldStats         # Telemetry dashboard
Field.serialize() -> FieldBundle    # Reproducible storage
```

**Exit Criteria** (Per Constitution Article I):
- [x] One API powers: sim, rendering, AI envs, auditing
- [x] Bounded latency mode with rank caps
- [x] Error dashboard with explicit metrics
- [x] 19/19 unit tests passing

---

### Layer 1 — FieldOps (Physics Primitives)

**Status**: 🔲 NOT STARTED

Standard library of field operators.

| Operator | Purpose | Priority |
|----------|---------|----------|
| `Grad` | Gradient ∇f | P0 |
| `Div` | Divergence ∇·F | P0 |
| `Curl` | Curl ∇×F | P1 |
| `Laplacian` | ∇²f | P0 |
| `Advect` | Semi-Lagrangian transport | P0 |
| `Diffuse` | Heat equation | P0 |
| `Project` | Pressure Poisson + divergence-free | P0 |
| `Forces` | Impulses, buoyancy, attractors | P1 |

**FieldGraph System**:
```python
graph = FieldGraph()
graph.add_node('advect', Advect(velocity_field))
graph.add_node('diffuse', Diffuse(viscosity=0.01))
graph.add_node('project', Project())
graph.connect('advect', 'diffuse', 'project')
graph.execute(field, dt=0.01)
```

**Exit Criteria**:
- [ ] Express "fluids, smoke, EM, diffusion" as compositions
- [ ] Divergence bounded post-projection
- [ ] BCs: periodic + obstacle masks
- [ ] Graph scheduler with caching

---

### Layer 2 — HyperVisual (Graphics Primitive)

**Status**: 🔲 NOT STARTED

Make the Field a graphics primitive.

| Component | Description |
|-----------|-------------|
| Volume Renderer | Raymarch against `sample()` or slice-to-bricks |
| Clipmap System | Multi-resolution, view-dependent bricks |
| Artist Controls | Rank ↔ detail, dissipation ↔ softness, vorticity ↔ drama |
| Engine Plugins | Unreal TensorField Volume, Unity equivalent |

**Rendering Paths**:
1. **Slice → 3D Texture → Raymarch** (practical, ship first)
2. **Direct Raymarch via sample()** (cleanest, later)

**Exit Criteria**:
- [ ] Real-time volumetric smoke/clouds
- [ ] Zoom forever without memory growth
- [ ] No tiling artifacts
- [ ] Artist can "direct" without PDEs

---

### Layer 3 — HyperSim (Benchmark Credibility)

**Status**: 🔲 NOT STARTED

Credibility pack for engineering buyers.

| Benchmark | Purpose |
|-----------|---------|
| Lid-driven cavity (2D) | Recirculating flow validation |
| Taylor-Green vortex | Energy decay, vorticity |
| Passive scalar advection | Transport accuracy |
| Poisson/projection | Divergence control |
| Decaying turbulence | Spectral behavior |

**Deliverables**:
- Error vs rank curves
- Stability vs dt curves
- Performance vs baseline charts
- "Sweet spot map" for buyers

**Exit Criteria**:
- [ ] 4+ canonical tests with plotted metrics
- [ ] Clear documentation of where HyperTensor wins
- [ ] Comparison against grid-based baselines

---

### Layer 4 — HyperEnv (AI Environments)

**Status**: 🔲 NOT STARTED

Field-native worlds for AI.

| Feature | Description |
|---------|-------------|
| Query Sensors | Agents probe via `sample()` at chosen points/rays |
| Bandwidth Budgets | Sensor bandwidth as RL constraint |
| Multi-Agent | Shared field state, deterministic stepping |
| Differentiable | Train policies against substrate, not pixels |

**Exit Criteria**:
- [ ] RL environment where resolution is agent choice
- [ ] Deterministic multi-agent stepping
- [ ] Optional differentiable hooks

---

### Layer 5 — Provenance & Attestation

**Status**: 🔲 NOT STARTED  
**Constitution**: Article IX (Reproducibility)

Trust layer for defense/science buyers.

| Component | Description |
|-----------|-------------|
| Deterministic Replay | Seeds, operator graph, hashes per step |
| Signed FieldBundles | Cryptographic attestation |
| Audit Viewer | Replay and verify tool |
| SBOM | Build-time software bill of materials |

**Exit Criteria**:
- [ ] Anyone can reproduce "this field evolved from X under Y"
- [ ] Third-party replay produces identical hashes
- [ ] Defense buyers stop flinching

---

### Layer 6 — Intentional Fields (The Bottom)

**Status**: 🔲 NOT STARTED

Steer structure, not pixels. This is the category jump.

#### 6.1 Intent Operators

| Operator | Purpose |
|----------|---------|
| Rank Bias Fields | Allocate detail where needed |
| Constraint Potentials | "Avoid region", "preserve calm zone" |
| Energy Shaping | Encourage/discourage turbulence |
| Attractors | Field drifts toward target structure |
| Budget Governors | Maintain frame-time caps |

#### 6.2 Field Directive Language (FDL)

```
region A: calm
region B: turbulent
detail_budget: 16ms
no_tiling: true
preserve_coherence: high
avoid: obstacle_mask
```

Compiles to: operator weights, rank caps, constraint potentials, schedule rules.

#### 6.3 Optimization-in-the-Loop

Find field evolution satisfying physics + intent + budgets.

**Exit Criteria**:
- [ ] Intent compiles into dynamics
- [ ] "Make it ominous and heavy" is a field constraint
- [ ] Worlds become solved objects under constraints

---

### Layer 7 — Field Operating System

**Status**: 🔲 NOT STARTED

Multi-field, multi-user, multi-process.

| Component | Description |
|-----------|-------------|
| Field Scheduler | Multiple fields sharing GPU with QoS |
| Memory Budgeting | Rank budgets per field |
| Cross-App Composition | One world state drives visuals + AI + sim |

---

### Layer 8 — Reality Representation Infrastructure

**Status**: 🔲 TERMINAL STATE

The endpoint:
- Reality modeled as **continuous fields**
- Fields are **compressed, evolvable, steerable**
- Everything is **query-based**
- Everything is **auditable**
- Renderers, solvers, AI are **clients** of the substrate

---

## Execution Timeline

### Phase A: Weeks 1-2 — Substrate Hardening ✅

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Oracle API finalization | ✅ | sample/slice/step/stats/serialize |
| Bounded mode | ✅ | Rank caps + budgets |
| Telemetry dashboard | ✅ | Rank, error, energy, timings |
| 19 unit tests | ✅ | All passing |

### Phase B: Weeks 3-4 — FieldOps MVP

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Advect/Diffuse/Project | 🔲 | Stable evolution |
| Graph scheduler | 🔲 | Topological execution with caching |
| BCs: periodic + obstacles | 🔲 | Mask-based boundaries |
| Divergence tests | 🔲 | Bounded post-projection |

### Phase C: Weeks 5-6 — HyperVisual MVP

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Slice-to-bricks + raymarch | 🔲 | Real-time volumetrics |
| Clipmap v0 | 🔲 | View-dependent LOD |
| Artist controls v0 | 🔲 | Detail/softness/drama knobs |

### Phase D: Weeks 7-8 — Benchmarks

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Benchmark harness | 🔲 | Automated test runner |
| 4+ canonical tests | 🔲 | With plotted metrics |
| Sweet spot map | 🔲 | Where HyperTensor wins |

### Phase E: Weeks 9-10 — FieldBundle + Replay

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Deterministic serialization | 🔲 | Reproducible to bit |
| Replay tool | 🔲 | Third-party verification |
| Hash chain | 🔲 | Integrity verification |

### Phase F: Weeks 11-14 — Intent Layer v1

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Intent operators | 🔲 | Region-based constraints |
| FDL compiler v0 | 🔲 | Directive → dynamics |
| Control loop | 🔲 | Budget-aware steering |

### Phase G: Weeks 15+ — Scale-Out

| Task | Status | Definition of Done |
|------|--------|-------------------|
| Unreal plugin | 🔲 | TensorField Volume actor |
| Unity plugin | 🔲 | Equivalent integration |
| RL environment | 🔲 | Query-native sensors |
| Enterprise packaging | 🔲 | SDK, docs, licensing |

---

## Milestone Acceptance Criteria

### Milestone 1: "Killer Visual Demo"

**Script**:
1. Start with calm smoke volume
2. Fly camera through it
3. Zoom into one filament repeatedly
4. Show constant memory usage
5. Increase "detail budget" live (rank rises, detail sharpens)
6. No tiling, no texture swapping

**Pass/Fail**: Viewers say "how is it not repeating?"

---

### Milestone 2: "Benchmark Credibility Pack"

**Script**:
1. Show Taylor-Green vortex error vs rank curves
2. Show divergence control
3. Show speed/memory vs baseline at multiple sizes
4. End with "sweet spot map"

**Pass/Fail**: Engineers stop arguing, start asking for access.

---

### Milestone 3: "Audit Replay"

**Script**:
1. Ship FieldBundle + replay tool
2. Third party replays, gets identical hashes

**Pass/Fail**: Someone else reproduces without you.

---

### Milestone 4: "Intent Demo"

**Script**:
1. Draw two regions: calm zone + storm zone
2. Toggle directives live
3. Show field obeys constraints while keeping frame budget

**Pass/Fail**: Feels like "directing reality."

---

## Definition-of-Done Checklist

Per Constitution Article III (Testing Protocols):

### Substrate ✅
- [x] Oracle API implemented + used everywhere
- [x] Bounded latency mode with rank caps
- [x] Contraction plans cached
- [x] Telemetry dashboard
- [x] 19/19 tests passing

### FieldOps
- [ ] advect/diffuse/project stable
- [ ] divergence bounded post-projection
- [ ] BCs: periodic + obstacle mask
- [ ] graph scheduler + caching

### Rendering
- [ ] slice-to-bricks + raymarch
- [ ] clipmap streaming
- [ ] zoom demo without tiling
- [ ] artist control panel

### Benchmarks
- [ ] 4+ tests with plotted metrics
- [ ] sweet spot map
- [ ] performance/memory comparisons

### Provenance
- [ ] FieldBundle schema versioned
- [ ] deterministic replay tool
- [ ] optional signature + hash chain

### Intent
- [ ] region-based constraints
- [ ] directive language compiler
- [ ] budget-aware control loop

---

## Module Mapping

Per Constitution Article II, Section 2.1:

| Layer | Module Location |
|-------|----------------|
| 0 - Substrate | `tensornet/substrate/` |
| 1 - FieldOps | `tensornet/fieldops/` |
| 2 - Rendering | `tensornet/visual/` |
| 3 - Benchmarks | `benchmarks/` |
| 4 - AI Envs | `tensornet/envs/` |
| 5 - Provenance | `tensornet/provenance/` |
| 6 - Intent | `tensornet/intent/` |
| 7 - Scheduler | `tensornet/runtime/` |

---

## Risk Register

| Risk | Mitigation | Owner |
|------|------------|-------|
| GPU memory limits | Bounded mode + rank caps | Substrate |
| Frame budget misses | Adaptive rank controller | Bounded Mode |
| Accuracy vs speed tradeoff | Sweet spot map documentation | Benchmarks |
| Adoption friction | Artist control layer | HyperVisual |
| Defense buyer trust | Provenance + attestation | Layer 5 |

---

## Amendment History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-24 | Initial ratification |

---

*This roadmap is governed by the HyperTensor Constitution. All development must comply with its provisions.*
