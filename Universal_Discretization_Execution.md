# Universal Discretization Execution Plan (UDv1 → UDv2)

## QTT Physics VM — Platform-Aligned Engineering Roadmap

> **Governing document:** [PLATFORM_SPECIFICATION.md](PLATFORM_SPECIFICATION.md)
> All architectural decisions, API constraints, IP boundaries, and determinism guarantees in this plan derive from and comply with the Platform Specification.

---

## Execution Status

> **All phases (A through G) are COMPLETE.** Committed as `55325f59` (Phases A–G) and `94dd8749` (domain key corrections).

| Phase | Deliverable | Tests | Commit |
|-------|-------------|------:|--------|
| **A** | VM Contract Hardening | 22 | `55325f59` |
| **B** | Operator Fidelity v1 | 31 | `55325f59` |
| **C** | Geometry Coefficient Compilation | 17 | `55325f59` |
| **D** | Benchmark Harness + Evidence Pipeline | 55 | `55325f59` |
| **E** | Wall Strategy v1.5 | 57 | `55325f59` |
| **F** | UDv2 Physics Breadth | 58 | `55325f59` |
| **G** | Hybrid Local Corrections + QoI Adaptivity | 65 | `55325f59` |
| | **Total** | **305** | |

**Output:** 34 files changed, 13,106 insertions, 122 deletions. 7 test files, all passing.

---

## 1) What the Platform Spec Forces, Architecturally

### 1.1 Two Worlds: Internal Truth vs External Contract

**Internal world (Ontic + VM):** Track ranks, truncations, opcodes, MPO caches, VM trace internals, and use them to drive adaptivity and V&V. The GPU runtime explicitly warns "never go through the sanitizer for internal metrics" — [gpu_runtime.py](ontic/engine/vm/gpu_runtime.py).

**External world (`physics_os` API):** Emit only whitelisted outputs and never leak TT internals. The Platform Spec explicitly forbids "TT cores, bond dims, SVD spectra, opcodes" at the sanitizer boundary ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)) and reiterates that internal state and rank distributions never leave ([§4 Runtime Access Layer](PLATFORM_SPECIFICATION.md#4-product-architecture--runtime-access-layer)):

> *"Whitelist-only extraction · 25 forbidden field categories · No TT cores · No bond dims · No SVD spectra · No opcodes"*

**Implication:** The "proof pack" concept must **bifurcate:**

| Pack | Contents | Audience |
|------|----------|----------|
| **Private/Internal Proof Pack** | Full telemetry, ranks, truncation history, opcode traces, saturation rates, CG iteration counts | Engineering, QA, NDA customer due diligence |
| **Public/Sanitized Proof Pack** | QoIs, conservation metrics, stability predicates, performance timing, dense slices (if whitelisted) | Certificates, `/v1/` result envelope, public attestation |

The sanitizer ([`physics_os/core/sanitizer.py`](PLATFORM_SPECIFICATION.md#146-physics_os-package-breakdown)) is the **sole exit path** from the QTT runtime to the public API. It performs whitelist-only extraction — only explicitly listed fields pass through. Everything else is discarded at the function boundary. This is not a filter; it is a reconstruction step that builds a new dictionary from scratch ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).

### 1.2 API Freeze Means "No New Surface Area"

The Platform Spec states the API surface is **frozen** as of v4.0.0 ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)):

- **Frozen endpoints:** 9 (all in `physics_os/api/routers/`)
- **Frozen schemas:** `JobRequest`, `JobResponse`, `ArtifactEnvelope`, `TrustCertificate`, `ValidationReport`, `Capabilities`, `ContractSchema`
- **Frozen error codes:** E001–E012
- **Versioning policy:** URI-versioned (`/v1/`), additive-only within a major version

**Implication:** UD/UG **cannot require new endpoints.** It must ship as:

1. New/expanded compiler behaviors under the **existing** job types and domain keys (within the 7-domain registry — [`physics_os/core/registry.py`](PLATFORM_SPECIFICATION.md#146-physics_os-package-breakdown)), and
2. **Additive-only changes** to internal execution, validation, and attestation logic.

Machine-readable contract: `GET /v1/contracts/v1` → JSON Schema download.

### 1.3 Determinism Is a Product Requirement, Not a Nice-to-Have

The roadmap must maintain the three-tier determinism envelope ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract), [`DETERMINISM_ENVELOPE.md`](docs/governance/DETERMINISM_ENVELOPE.md)):

| Tier | Guarantee | Example |
|------|----------|---------|
| **Bitwise** | Identical bytes across runs | Hashing, serialization, signing |
| **Reproducible** | Identical within ε ≤ 10⁻¹² | Single-precision physics on same hardware + seed |
| **Physically Equivalent** | Within measurement uncertainty | Cross-hardware, mixed precision |

This is especially important once hybrid local corrections and QoI-driven adaptivity are added. Adaptive policies must be **deterministic given seed/config**, or explicitly classified under the tiering system.

---

## 2) Where "Universal Discretization" Lives in This Repo

The Platform Spec already defines the VM architecture and its module boundaries ([§5.1 VM Architecture](PLATFORM_SPECIFICATION.md#51-vm-architecture)): IR, compilers, runtime, GPU runtime, QTT tensor, operators, GPU operators, rank governor, telemetry, benchmark, postprocessing.

That means the "UG program" **is not a new repo** — it is a set of focused evolutions inside these existing seams.

### 2.0 Existing Solver Inventory (Context)

The VM compilers in `ontic/vm/compilers/` emit QTT IR programs that express physics as bytecode. They draw algorithmic foundations from the **existing full-scale solver library** already in the codebase — this is not greenfield work:

| Module | Files | LOC | Solvers Present |
|--------|------:|----:|-----------------|
| `ontic/cfd/` | 115 | ~78K | Euler 1D/2D/3D/ND, NS 2D/3D (QTT-native, real-time, turbo), Vlasov 5D, WENO/DG/SEM, turbulence (RANS, LES, DNS), combustion DNS, reactive NS, LBM, SPH, DSMC |
| `ontic/em/` | 24 | ~15K | 3D Maxwell, Helmholtz (CPU + GPU), topology optimization, S-parameters, PML boundaries |
| `ontic/engine/` | 93 | ~36K | QTT Physics VM, GPU kernels, HAL (7 HW backends), distributed TN |
| `ontic/quantum/` | 99 | ~22K | Condensed matter, electronic structure, QFT, stat mech |
| `ontic/materials/` | 42 | ~10K | Mechanics, fracture, IGA, MPM, peridynamics, XFEM |
| `ontic/plasma_nuclear/` | 36 | ~8.8K | MHD, gyrokinetics, fusion, nuclear |
| `ontic/fluids/` | 38 | ~8.9K | Multiphase, FSI, heat transfer, porous media, phase-field |

**Key point:** Solvers like `ns3d_qtt_native.py`, `euler_3d.py`, `fast_euler_3d.py`, and `qtt_helmholtz_gpu.py` already exist as production QTT-native implementations. The VM compiler layer provides the *bytecode abstraction* — compiling physics into IR programs that the VM runtime executes — not reimplementing the solvers from scratch. See [`INVENTORY.md`](INVENTORY.md) for the complete catalog.

Additionally, `experiments/benchmarks/benchmarks/gpu_qtt_maxwell_3d.py` (343 lines) demonstrates O(log N) QTT compression on 3D Helmholtz from 128³ → 4096³ — rank *decreasing* at higher scales (48 → 16), with full GPU attestation artifacts. This is the proven operational pattern the VM compilers formalize.

### 2.1 Canonical Placement (No Conceptual Conflicts)

| Roadmap Capability | Primary Location | Why It Belongs There |
|--------------------|-----------------|---------------------|
| IR evolution (new opcodes, program metadata, invariants) | `ontic/vm/ir.py` | Spec: "Instruction set defining QTT operations" ([§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture)) |
| New physics compilers (NS3D, compressible, CHT, multiphase) | `ontic/vm/compilers/` | Spec: "Domain compilers translate params into IR" ([§5.2](PLATFORM_SPECIFICATION.md#52-domain-compilers)) |
| MPO families (higher order, variable coeff, WENO-like) | `ontic/vm/operators.py` + `ontic/vm/gpu_operators.py` | Spec declares operators and GPU operators modules ([§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture)) |
| Rank/precision adaptivity | `ontic/vm/rank_governor.py` + GPU governor | Rank Governor is a first-class component ([§5.3](PLATFORM_SPECIFICATION.md#53-rank-governor)) |
| Telemetry + Rank Atlas integration | `ontic/vm/telemetry.py`, `ontic/vm/benchmark.py` | Spec calls these out directly ([§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture)) |
| V&V harness, convergence studies | `ontic/platform/vv/` | Spec: V&V harness supports convergence, MMS, conservation ([§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework), [§6.2](PLATFORM_SPECIFICATION.md#62-platform-substrate-v200)) |
| Public outputs and IP boundary | `physics_os/core/sanitizer.py` | Spec: sanitizer is sole exit, whitelist-only, forbidden field categories ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)) |
| Certificates + claim-witness predicates | `physics_os/core/evidence.py` + `physics_os/core/certificates.py` | Spec: Evidence Validate + Claims + Ed25519 signed certs ([§10](PLATFORM_SPECIFICATION.md#10-trustless-physics-certificates--tenet-tphy), [§14.6](PLATFORM_SPECIFICATION.md#146-physics_os-package-breakdown)) |
| Contracts | `contracts/v1/` and `/v1/contracts/v1` | Spec: machine-readable contract served ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)) |

---

## 3) Combined Engineering Execution Plan, Repo-Native (UGv1 + UGv2)

Structured as phases with deliverables and "Definition of Done" gates. No timelines.

---

### Phase A: VM Contract Hardening (Product-Kernel Maturity)

**Goal:** Make the QTT VM and GPU runtime "product-grade" under platform constraints.

#### Deliverables

**IR and runtime invariants:**
- "Never go dense" enforced at runtime level (the GPU runtime already codifies this as a rule set — [gpu_runtime.py](ontic/engine/vm/gpu_runtime.py)).
- Prohibit internal metric leakage through sanitizer paths — [gpu_runtime.py](ontic/engine/vm/gpu_runtime.py).

**Determinism envelope integration:**
- Record seed/config hashes and device class so Tier-2 reproducibility can be asserted ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract)).

**Telemetry schema:**
- **Internal:** ranks, truncations, saturation rates (private).
- **External:** conservation/stability/QoIs/perf only (sanitized) — compliant with ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).

#### Definition of Done

- [x] Every VM execution yields:
  - Internal telemetry bundle (full).
  - Sanitized result envelope that **provably contains none** of the 25 forbidden categories (TT cores, bond dims, opcodes, etc.) ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).
- [x] Certificates only bind to sanitized outputs and contract fields.

> **Completed:** `PublicMetrics`/`PrivateMetrics` split, `DeterminismTier` enum, `FORBIDDEN_FIELDS` enforcement, `to_dense()` execution fence, 22 tests in `test_sanitizer_compliance.py`.

---

### Phase B: Operator Fidelity v1 (The "Mesh Quality" Replacement)

**Goal:** Make MPO derivatives and elliptic solves defensible as discretization primitives.

#### Deliverables

**MPO family versions:**
- Baseline grad/lap MPOs plus "higher order" variants.
- Variable-coefficient elliptic forms (e.g., ∇·(a∇u)) as composable MPO pipelines.

**Poisson/projection quality:**
- CG Poisson solver correctness and robust truncation coupling (since CG in TT implies truncation is part of the algorithm) — [gpu_operators.py](ontic/engine/vm/gpu_operators.py).

**Verification suite (MMS):**
- Gradient MMS, Laplacian MMS, Poisson analytic checks.
- Convergence as `n_bits` increases plus truncation tightening (the "h/p" analog).
- Integrated via the existing V&V framework ([§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework)).

#### Definition of Done

- [x] Observed convergence trend exists across bit-depth levels on MMS.
- [x] Poisson residual meets target without pathological rank blow-up under governor (governor already supports adaptive effective rank and tracks saturation — [§5.3](PLATFORM_SPECIFICATION.md#53-rank-governor), [gpu_runtime.py](ontic/engine/vm/gpu_runtime.py)).

> **Completed:** `OperatorFamily`/`OperatorVariant` enums, 4th-order MPOs, variable-coefficient elliptic operator, MMS verification suite, 31 tests in `test_operator_fidelity.py`.

---

### Phase C: Geometry as Coefficient Compilation (CAD-Optional Universality)

**Goal:** Replace meshing with QTT-native geometry/BC compilation that preserves the IP boundary contract.

#### Deliverables

**Geometry compiler produces QTT coefficient fields:**
- Masks (χ_solid(x) — indicator)
- Material properties (β(x) — penalization strength)
- Optional distance proxies (φ(x) — signed distance for wall models)

**BC compilation** that does not rely on "editing TT cores as BCs" except for tightly defined boundary projector constructs.

**Sanitizer compliance:**
- Internal geometry fields can exist and be used, but **never leak TT structure** through the API — ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).

#### Definition of Done

- [x] "CAD-free" procedural geometry cases run end-to-end.
- [x] Imported geometry (if supported) yields stable coefficient fields without exploding rank for typical shapes.
- [x] Rank behavior recorded **internally only** (forbidden externally per [§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).

> **Completed:** `geometry_coeffs.py` compiler (mask, material, penalty, distance fields), 17 tests in `test_geometry_coeffs.py`.

---

### Phase D: Benchmark Harness + Evidence Pipeline v1 (Claimable Quality)

**Goal:** Transform solver quality into product-quality proof artifacts.

#### Deliverables

**Bench registry** integrated with existing V&V harness ([§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework)):
- MMS, cavity, Couette, Taylor-Green, etc. (the spec lists validated CFD benchmarks in [§13.1](PLATFORM_SPECIFICATION.md#131-cfd-benchmarks)).

**Two-tier proof packs:**

| Tier | Contents | Constraint |
|------|----------|-----------|
| **Internal pack** | Full rank/truncation/trace diagnostics | Engineering-only; never crosses sanitizer |
| **Public pack** | Conservation, stability, boundedness, QoIs, `wall_time_s`, plus allowable dense outputs | Feeds certificates and `/v1/` envelope |

**Claim-witness predicates** wired to certificates:
- Registered tags: `CONSERVATION`, `STABILITY`, `BOUND` (already in the certificate model — [§10](PLATFORM_SPECIFICATION.md#10-trustless-physics-certificates--tenet-tphy)).
- Promoted tags (now implemented in `evidence.py`): `CONVERGENCE`, `REPRODUCIBILITY`, `ENERGY_BOUND`, `CFL_SATISFIED`, `BOUNDEDNESS` — see [`CLAIM_REGISTRY.md`](docs/governance/CLAIM_REGISTRY.md) for definitions.

#### Definition of Done

- [x] All benchmark jobs can be executed via existing job model and routed to attestation without exposing forbidden outputs ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).
- [x] `/v1/contracts/v1` remains stable — additive-only where allowed ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)).

> **Completed:** `registry.yaml`, `scorecard_public_v1.schema.json`, V&V `harness.py`, 4 promoted claim tags (CONVERGENCE, REPRODUCIBILITY, ENERGY_BOUND, CFL_SATISFIED), 55 tests in `test_vv_harness.py`.

---

### Phase E: Wall Strategy v1.5 (Universality Hinge)

**Goal:** Make wall-dominated QoIs defensible without body-fitted meshes.

#### Deliverables

**A single declared wall lane** (choose one and benchmark it):

| Lane | Strategy | Description |
|------|----------|-------------|
| **A** *(recommended for QTT)* | Penalization + calibrated wall model | Build a QTT "wall distance proxy" d(x) in a narrow band. Apply wall shear stress closure and thermal wall closure via coefficient fields. |
| **B** | Ghost-fluid / interface constraints *(harder)* | Represent interface conditions as local constraints. Likely requires hybrid (TT + local corrections) sooner. |

**Wall diagnostics** become internal-first, and only exposed as allowable aggregates (e.g., integrated shear proxies) if whitelisted.

#### Definition of Done

- [x] Turbulent channel/cavity style validations pass in a repeatable way (with convergence trends and stable certificates).
- [x] No leakage of wall-model internals beyond whitelisted aggregates — ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).

> **Completed:** `wall_model.py` (penalization + calibrated wall model), `wall_benchmarks.py` (Ghia cavity, Schäfer-Turek), extended `navier_stokes_2d.py` with Brinkman penalization, 57 tests in `test_wall_model.py`.

---

### Phase F: UGv2 Physics Breadth (Compressible + CHT + One Multiphase Lane)

**Goal:** Expand physics coverage while keeping correctness and contract compliance.

#### Deliverables

**Compressible compiler + ops** with boundedness checks (ρ > 0, p > 0) expressed as evidence predicates.

**CHT coupling** via coefficient fields and conservative energy bookkeeping.

**Multiphase lane selection:**

| Lane | Strategy | Description |
|------|----------|-------------|
| **A** *(QTT-friendly)* | Phase-field | Diffuse interface, IMEX stepping (stiffness). Interface captured as smooth field — less rank-hostile. |
| **B** *(industry standard)* | VOF (requires hybrid) | Sharp interface, strict volume conservation. Must use local correction bands/tiles near interface. |

#### Definition of Done

- [x] Each new lane ships with: benchmarks, convergence trend artifacts, certificate predicates.
- [x] Sanitizer whitelist updates **only if truly necessary** — avoid unless essential ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)).

> **Completed:** `compressible_euler.py` (1D Euler with exact Riemann solver, Sod/Shu-Osher/smooth-sine ICs), `cht_coupling.py` (CHT with variable-coefficient heat equation), `phase_field.py` (Cahn-Hilliard 2D), BOUNDEDNESS evidence claim, 58 tests in `test_physics_breadth.py`.

---

### Phase G: Hybrid Local Corrections + QoI-Driven Adaptivity (the "Better Per Dollar" Win)

**Goal:** Maintain TT compression in rank-hostile regimes by adding controlled locality, while preserving determinism guarantees.

#### Deliverables

**Hybrid representation and policy:**
- TT backbone + localized correction subspace (tiles/bands).
- `HybridField: q = q_TT + q_local`
- `q_local` options: sparse ROI tiles (dense blocks); narrow bands for shocks/interfaces.
- Feature sensors: shock sensor computed via QTT gradients (MPO apply); tile activation policy.

**QoI-driven adaptivity:**
- Automatic rank/tolerance tuning to meet QoI targets.
- Truncation tolerances adjustable per operator stage, max rank per field, hybrid region activation.
- Stop conditions keyed to QoI confidence (convergence trend).

> This is where the platform outperforms incumbents operationally: the user specifies QoIs, the runtime adjusts rank/precision to meet them.

**Determinism story:**
- Adaptivity must be **deterministic given seed/config**, or explicitly classified under determinism tiering ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract)).

#### Definition of Done

- [x] Demonstrated "rank-hostile" cases (shocks/interfaces) stay stable and accurate without uncontrolled saturation.
- [x] Public outputs remain contract-safe and do not leak forbidden categories.

> **Completed:** `hybrid_field.py` (HybridField: q_TT + q_local, LocalTile, FeatureSensor, TileActivationPolicy), `qoi_adaptivity.py` (QoITarget, ConvergenceTrend, QoIHistory, AdaptiveRankPolicy), 65 tests in `test_hybrid_adaptivity.py`.

---

## 4) The Key Correction vs "HyperFoam Disadvantages"

Earlier "HyperFoam" disadvantages were largely about explicit grid/immersed boundary choices. In this repo's QTT VM framing, the dominant risk classes become:

| Risk Class | Nature | Tracking Mechanism |
|-----------|--------|-------------------|
| **Boundary and wall closure maturity** | Still the hardest part of "universal CFD." | Phase E benchmarks + wall diagnostics. |
| **Truncation-governed numerical bias** | Now a first-class accuracy variable and a product telemetry variable (internal) — [gpu_runtime.py](ontic/engine/vm/gpu_runtime.py). | Rank governor saturation tracking, convergence studies. |
| **Evidence and contract compliance** | The platform spec makes non-negotiable: forbidden outputs, API freeze, determinism envelope ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract), [§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract), [§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)). | Sanitizer compliance tests, certificate binding. |

---

## 5) Repo-Ready Artifacts

The three artifacts below are aligned to the Platform Shell constraints: frozen `/v1/` contract surface ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)), determinism envelope tiers ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract)), and the IP boundary rule that **only whitelisted fields can leave the VM** while internal state, tensor cores, compiler IR, rank distributions, and similar internals are never exposed ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)). The file-path mapping uses the VM architecture layout as specified ([§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture)): IR, compilers, runtime, GPU runtime, QTT tensor, operators, rank governor, telemetry, benchmark, postprocessing — and the V&V harness modules under `ontic/platform/vv/` (convergence, conservation, MMS, stability, benchmark comparison, performance) ([§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework)).

---

### 5.1 Benchmark Registry v1 (YAML)

This registry is intentionally **QTT-native**: "h refinement" is `n_bits` (doubling points per axis), "p refinement" is "operator family / MPO order variant," and "dt refinement" is time-step refinement — matching the `vv/convergence.py` intent (h/p/dt with Richardson extrapolation) ([§6.2](PLATFORM_SPECIFICATION.md#62-platform-substrate-v200)). The CFD cases are grounded in the existing "CFD Benchmarks" table ([§13.1](PLATFORM_SPECIFICATION.md#131-cfd-benchmarks)): Sod, Shu-Osher, Taylor-Green, Kida, Double Mach, Kelvin-Helmholtz, Lid-driven cavity, Couette.

**Target path:** `ontic/platform/vv/registry.yaml`

```yaml
version: "1.0"
registry_name: "UD_BENCHMARK_REGISTRY_V1"
contract_surface: "/v1 (frozen, additive-only)"
notes:
  - "This registry is internal to ontic/platform/vv and must not require new API endpoints."
  - "All benchmark outputs must be emitted as (a) internal proof pack (full telemetry) and (b) sanitized proof pack (whitelisted only)."

global_defaults:
  determinism_tier_required: "reproducible"   # tiers: §20.2 Determinism Contract
  convergence:
    levels: 3
    refinement_axes: ["n_bits", "dt", "operator_variant"]
    richardson: true
  gates:
    conservation:
      mass_abs_max: 1.0e-10
      momentum_abs_max: 1.0e-10
      energy_abs_max: 1.0e-10
    boundedness:
      require_positive: ["rho", "p"]   # only for compressible cases
    stability:
      max_nan_steps: 0
      max_inf_steps: 0

benchmarks:

  # ====================================================================
  # Verification (MMS / Analytic)
  # ====================================================================

  - id: "V010_MMS_GRADIENT_1D"
    category: "verification"
    domain_key: "advection_diffusion"
    dimensions: 1
    purpose: "Verify gradient MPO correctness and observed order under n_bits refinement."
    run_spec: "ontic/platform/vv/cases/V010_MMS_GRADIENT_1D/spec.yaml"
    vv:
      mms: "vv/mms.py"
      convergence: "vv/convergence.py"
      performance: "vv/performance.py"
    refinement_plan:
      n_bits: [8, 9, 10]
      operator_variant: ["grad_v1", "grad_v2_high_order"]
    qoi:
      - name: "L2_error_dudx"
        units: "1"
    gates:
      - type: "observed_order_min"
        qoi: "L2_error_dudx"
        min_order: 2.0

  - id: "V020_MMS_LAPLACIAN_1D"
    category: "verification"
    domain_key: "advection_diffusion"
    dimensions: 1
    purpose: "Verify Laplacian MPO correctness and observed order."
    run_spec: "ontic/platform/vv/cases/V020_MMS_LAPLACIAN_1D/spec.yaml"
    vv:
      mms: "vv/mms.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [8, 9, 10]
      operator_variant: ["lap_v1", "lap_v2_high_order"]
    qoi:
      - name: "L2_error_d2udx2"
        units: "1"
    gates:
      - type: "observed_order_min"
        qoi: "L2_error_d2udx2"
        min_order: 2.0

  - id: "V030_POISSON_ANALYTIC_2D"
    category: "verification"
    domain_key: "navier_stokes_2d"
    dimensions: 2
    purpose: "Verify Poisson/elliptic solve accuracy with analytic RHS/solution; validate CG+truncation interaction."
    run_spec: "ontic/platform/vv/cases/V030_POISSON_ANALYTIC_2D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [7, 8, 9]
      operator_variant: ["lap_v1", "lap_v2_high_order"]
      dt: [null]   # steady solve
    qoi:
      - name: "L2_error_phi"
        units: "1"
      - name: "cg_iterations"
        units: "count"
    gates:
      - type: "absolute_max"
        qoi: "L2_error_phi"
        max: 1.0e-6

  - id: "V040_PROJECTION_INTEGRITY_2D"
    category: "verification"
    domain_key: "navier_stokes_2d"
    dimensions: 2
    purpose: "Verify incompressible projection reduces divergence and remains stable under truncation policy."
    run_spec: "ontic/platform/vv/cases/V040_PROJECTION_INTEGRITY_2D/spec.yaml"
    vv:
      conservation: "vv/conservation.py"
      stability: "vv/stability.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [7, 8, 9]
      dt: [1.0e-2, 5.0e-3, 2.5e-3]
    qoi:
      - name: "divergence_norm_post_projection"
        units: "1"
    gates:
      - type: "absolute_max"
        qoi: "divergence_norm_post_projection"
        max: 1.0e-10

  # ====================================================================
  # CFD Benchmarks (per §13.1 CFD Benchmarks)
  # ====================================================================

  - id: "C110_SOD_SHOCK_TUBE_1D"
    category: "cfd"
    domain_key: "compressible_euler_1d"
    dimensions: 1
    benchmark_name: "Sod shock tube"
    reference: "Exact Riemann solution"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C110_SOD_SHOCK_TUBE_1D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      stability: "vv/stability.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [10, 11, 12]
      dt: [2.0e-4, 1.0e-4, 5.0e-5]
      operator_variant: ["flux_hllc_v1"]
    qoi:
      - name: "L1_error_rho"
        units: "1"
      - name: "boundedness"
        units: "bool"
    gates:
      - type: "boundedness"
        require_positive: ["rho", "p"]
      - type: "absolute_max"
        qoi: "L1_error_rho"
        max: 2.0e-2

  - id: "C120_SHU_OSHER_1D"
    category: "cfd"
    domain_key: "compressible_euler_1d"
    dimensions: 1
    benchmark_name: "Shu-Osher"
    reference: "Shock-turbulence interaction"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C120_SHU_OSHER_1D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      stability: "vv/stability.py"
    refinement_plan:
      n_bits: [10, 11, 12]
      operator_variant: ["flux_hllc_v1", "flux_weno_like_v2"]
    qoi:
      - name: "density_oscillation_metric"
        units: "1"
      - name: "boundedness"
        units: "bool"
    gates:
      - type: "boundedness"
        require_positive: ["rho", "p"]

  - id: "C210_TAYLOR_GREEN_VORTEX_2D"
    category: "cfd"
    domain_key: "navier_stokes_2d"
    dimensions: 2
    benchmark_name: "Taylor-Green vortex"
    reference: "Analytical decay rate"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C210_TAYLOR_GREEN_VORTEX_2D/spec.yaml"
    vv:
      conservation: "vv/conservation.py"
      benchmarks: "vv/benchmarks.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [8, 9, 10]
      dt: [1.0e-2, 5.0e-3, 2.5e-3]
      operator_variant: ["ns2d_vorticity_v1", "ns2d_vorticity_v2_low_diffusion"]
    qoi:
      - name: "kinetic_energy_decay_error"
        units: "1"
    gates:
      - type: "absolute_max"
        qoi: "kinetic_energy_decay_error"
        max: 1.0e-3

  - id: "C220_KIDA_VORTEX_3D"
    category: "cfd"
    domain_key: "navier_stokes_3d"
    dimensions: 3
    benchmark_name: "Kida vortex"
    reference: "Enstrophy conservation"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C220_KIDA_VORTEX_3D/spec.yaml"
    vv:
      conservation: "vv/conservation.py"
      stability: "vv/stability.py"
    refinement_plan:
      n_bits: [6, 7, 8]
      dt: [5.0e-3, 2.5e-3, 1.25e-3]
    qoi:
      - name: "enstrophy_drift"
        units: "1"
    gates:
      - type: "absolute_max"
        qoi: "enstrophy_drift"
        max: 1.0e-3

  - id: "C230_DOUBLE_MACH_REFLECTION_2D"
    category: "cfd"
    domain_key: "compressible_euler_2d"
    dimensions: 2
    benchmark_name: "Double Mach reflection"
    reference: "Woodward-Colella reference"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C230_DOUBLE_MACH_REFLECTION_2D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      stability: "vv/stability.py"
    refinement_plan:
      n_bits: [9, 10, 11]
      operator_variant: ["flux_hllc_v1", "flux_weno_like_v2"]
    qoi:
      - name: "shock_position_error"
        units: "1"
      - name: "boundedness"
        units: "bool"
    gates:
      - type: "boundedness"
        require_positive: ["rho", "p"]

  - id: "C240_KELVIN_HELMHOLTZ_2D"
    category: "cfd"
    domain_key: "navier_stokes_2d"
    dimensions: 2
    benchmark_name: "Kelvin-Helmholtz"
    reference: "Instability growth rate"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C240_KELVIN_HELMHOLTZ_2D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [8, 9, 10]
      dt: [1.0e-2, 5.0e-3, 2.5e-3]
    qoi:
      - name: "growth_rate_error"
        units: "1"
    gates:
      - type: "absolute_max"
        qoi: "growth_rate_error"
        max: 5.0e-2

  - id: "C250_LID_DRIVEN_CAVITY_2D"
    category: "cfd"
    domain_key: "navier_stokes_2d"
    dimensions: 2
    benchmark_name: "Lid-driven cavity"
    reference: "Ghia et al. (1982)"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C250_LID_DRIVEN_CAVITY_2D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [7, 8, 9]
      dt: [5.0e-3, 2.5e-3, 1.25e-3]
    qoi:
      - name: "centerline_u_profile_L2"
        units: "1"
      - name: "centerline_v_profile_L2"
        units: "1"
    gates:
      - type: "absolute_max"
        qoi: "centerline_u_profile_L2"
        max: 5.0e-2
      - type: "absolute_max"
        qoi: "centerline_v_profile_L2"
        max: 5.0e-2

  - id: "C260_COUETTE_FLOW_2D"
    category: "cfd"
    domain_key: "navier_stokes_2d"
    dimensions: 2
    benchmark_name: "Couette flow"
    reference: "Analytical linear profile"
    status_in_spec: "validated"
    run_spec: "ontic/platform/vv/cases/C260_COUETTE_FLOW_2D/spec.yaml"
    vv:
      benchmarks: "vv/benchmarks.py"
      convergence: "vv/convergence.py"
    refinement_plan:
      n_bits: [7, 8, 9]
      dt: [5.0e-3, 2.5e-3, 1.25e-3]
    qoi:
      - name: "velocity_profile_L2"
        units: "1"
    gates:
      - type: "absolute_max"
        qoi: "velocity_profile_L2"
        max: 1.0e-3
```

**Why this matches the spec:** it uses the V&V harness methods exactly as documented (convergence, conservation, MMS, stability, benchmark comparison, performance — [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework)), and starts with the CFD benchmark set already listed as validated ([§13.1](PLATFORM_SPECIFICATION.md#131-cfd-benchmarks)).

---

### 5.2 Sanitizer-Safe Scorecard Schema (JSON Schema)

This is a **public-facing** (sanitized) scorecard shape that can sit inside the existing `ArtifactEnvelope` without adding new endpoints — consistent with the "whitelist only, block internal state, tensor cores, compiler IR, rank distributions" rule ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)). It bakes in the determinism tier field per the determinism contract ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract)), and a metering summary aligned to the CU formula ([§20.3](PLATFORM_SPECIFICATION.md#203-metering--pricing-contract)).

**Target path:** `contracts/v1/schemas/scorecard_public_v1.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://holonomix.local/schemas/scorecard_public_v1.schema.json",
  "title": "ScorecardPublicV1 (Sanitized)",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "job_id",
    "domain_key",
    "status",
    "timestamps",
    "determinism",
    "evidence",
    "qoi",
    "performance"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^1\\.[0-9]+$"
    },

    "job_id": { "type": "string", "minLength": 8 },
    "run_id": { "type": "string" },

    "domain_key": {
      "type": "string",
      "description": "Domain pack key used by the VM compiler (e.g., navier_stokes_2d)."
    },

    "status": {
      "type": "string",
      "enum": ["succeeded", "failed", "canceled"]
    },

    "error": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "code": { "type": "string" },
        "message": { "type": "string" },
        "hint": { "type": "string" }
      }
    },

    "timestamps": {
      "type": "object",
      "additionalProperties": false,
      "required": ["started_utc", "finished_utc"],
      "properties": {
        "started_utc": { "type": "string", "format": "date-time" },
        "finished_utc": { "type": "string", "format": "date-time" }
      }
    },

    "determinism": {
      "type": "object",
      "additionalProperties": false,
      "required": ["tier"],
      "properties": {
        "tier": {
          "type": "string",
          "enum": ["bitwise", "reproducible", "physically_equivalent"],
          "description": "Determinism tier claimed for this run (§20.2)."
        },
        "notes": { "type": "string" }
      }
    },

    "evidence": {
      "type": "object",
      "additionalProperties": false,
      "required": ["claims", "checks"],
      "properties": {
        "claims": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": [
              "CONSERVATION",
              "STABILITY",
              "BOUND",
              "CONVERGENCE",
              "REPRODUCIBILITY",
              "ENERGY_BOUND",
              "CFL_SATISFIED",
              "BOUNDEDNESS"
            ]
          }
        },
        "checks": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "mass_balance_abs": { "type": "number", "minimum": 0.0 },
            "momentum_balance_abs": { "type": "number", "minimum": 0.0 },
            "energy_balance_abs": { "type": "number", "minimum": 0.0 },

            "boundedness_pass": { "type": "boolean" },
            "boundedness_notes": { "type": "string" },

            "stability_pass": { "type": "boolean" },
            "stability_notes": { "type": "string" }
          }
        },

        "certificate": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "trust_certificate_id": { "type": "string" },
            "signature_alg": { "type": "string" },
            "certificate_hash": { "type": "string" }
          }
        }
      }
    },

    "qoi": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "value", "units"],
        "properties": {
          "name": { "type": "string" },
          "value": { "type": "number" },
          "units": { "type": "string" },
          "reference": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "source": { "type": "string" },
              "value": { "type": "number" },
              "error_abs": { "type": "number", "minimum": 0.0 },
              "error_rel": { "type": "number", "minimum": 0.0 }
            }
          }
        }
      }
    },

    "convergence": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "performed": { "type": "boolean" },
        "levels": { "type": "integer", "minimum": 1 },
        "refinement_axes": {
          "type": "array",
          "items": { "type": "string", "enum": ["n_bits", "dt", "operator_variant"] }
        },
        "summary": { "type": "string" }
      }
    },

    "performance": {
      "type": "object",
      "additionalProperties": false,
      "required": ["wall_seconds", "device_class", "compute_units"],
      "properties": {
        "wall_seconds": { "type": "number", "minimum": 0.0 },
        "device_class": { "type": "string", "enum": ["cpu", "gpu_consumer", "gpu_datacenter"] },
        "compute_units": { "type": "number", "minimum": 0.0 },
        "domain_weight": { "type": "number", "minimum": 0.0 },
        "notes": { "type": "string" }
      }
    },

    "artifacts": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "ref"],
        "properties": {
          "kind": { "type": "string", "description": "e.g., slice_png, roi_vtk, report_md" },
          "ref": { "type": "string", "description": "Opaque storage key or path — not raw internals" }
        }
      }
    }
  },

  "description": "Sanitized scorecard. Must not include TT cores, bond dimensions, SVD spectra, compiler IR, opcode traces, rank distributions, or any internal VM state. Enforced by §20.4 IP Boundary & Forbidden Outputs."
}
```

This schema is intentionally **silent** on TT/QTT-specific internals to satisfy the blocklist rule ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)) while remaining rich enough to support attestation and customer-facing evidence.

---

### 5.3 Phase-to-Repo Change Map

This map records where each UD phase landed in the repo. All `/v1/` endpoints and schemas remained **frozen** (9 frozen endpoints/schemas, additive-only within `/v1/` — [§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)), and all UD evolution is contained inside the VM/compilers/V&V/evidence plumbing.

#### A. VM Contract Hardening (Product-Kernel Maturity) — COMPLETED

**Primary files** (per VM architecture — [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture)):

| File | Role | Spec Reference |
|------|------|----------------|
| `ontic/vm/ir.py` | IR contract | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture): "Instruction set defining QTT operations" |
| `ontic/vm/runtime.py`, `ontic/vm/gpu_runtime.py` | Executor paths | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture), [§5.4](PLATFORM_SPECIFICATION.md#54-gpu-runtime) |
| `ontic/vm/qtt_tensor.py` | Core TT/QTT structure | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture): "QTT Tensor" |
| `ontic/vm/telemetry.py` | Execution metrics | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture): "Telemetry" |

**Delivered:**
- `PublicMetrics` / `PrivateMetrics` sanitizer boundary class in telemetry.
- `DeterminismTier` enum in run manifests, matching the 3-tier definition ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract)).
- `to_dense()` execution fence enforcing "never dense" as a runtime invariant ([§1](PLATFORM_SPECIFICATION.md#1-executive-summary), [§5.3](PLATFORM_SPECIFICATION.md#53-rank-governor)).
- `FORBIDDEN_FIELDS` enforcement in `physics_os/core/sanitizer.py` ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).
- 22 tests in `test_sanitizer_compliance.py`.

---

#### B. Operator Fidelity v1 (MPO Quality Becomes "Mesh Quality") — COMPLETED

**Primary files:**

| File | Role | Spec Reference |
|------|------|----------------|
| `ontic/vm/operators.py` | QTT-format differential operators | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture): "Operators" |
| `ontic/vm/gpu_operators.py` | GPU-accelerated operator kernels | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture): "GPU Operators" |

**Delivered:**
- `OperatorFamily` / `OperatorVariant` enums for convergence "p refinement."
- 4th-order gradient and Laplacian MPO variants (`grad_v2`, `lap_v2`).
- Variable-coefficient elliptic operator composition (`variable_coeff_elliptic_apply`).
- MMS verification suite with observed order checks.
- 31 tests in `test_operator_fidelity.py`.

---

#### C. Geometry as Coefficient Compilation (CAD-Optional Universality) — COMPLETED

**Primary files:**

| File | Role | Spec Reference |
|------|------|----------------|
| `ontic/vm/compilers/` | Domain compilers that translate params into IR | [§5.2](PLATFORM_SPECIFICATION.md#52-domain-compilers) |

**Delivered:**
- `ontic/vm/compilers/geometry_coeffs.py` — compiler helper module producing:
  - Mask fields (fluid/solid indicator)
  - Material coefficient fields (spatially varying)
  - Penalty fields (Brinkman penalization strength)
  - Distance fields (signed distance for wall models)
- 17 tests in `test_geometry_coeffs.py`.

**Why this fits the spec:** It extends compilers (allowed and expected — [§5.2](PLATFORM_SPECIFICATION.md#52-domain-compilers)) rather than adding a "mesh system." The VM contract already frames operations as QTT create/apply/round/measure/step, not mesh assembly ([§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture)).

---

#### D. Benchmark Harness + Evidence Pipeline v1 — COMPLETED

**Primary files:**

| File | Role | Spec Reference |
|------|------|----------------|
| `ontic/platform/vv/convergence.py` | h/p/dt refinement with Richardson extrapolation | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| `ontic/platform/vv/conservation.py` | Mass, momentum, energy balance verification | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| `ontic/platform/vv/mms.py` | Method of Manufactured Solutions | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| `ontic/platform/vv/stability.py` | CFL, von Neumann, eigenvalue analysis | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| `ontic/platform/vv/benchmarks.py` | Reference solution comparison suite | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| `ontic/platform/vv/performance.py` | Timing, memory, scaling analysis | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| `ontic/vm/benchmark.py` | Rank Atlas profiling | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture): "Benchmark" |

**Delivered:**
- `ontic/platform/vv/registry.yaml` — 12+ benchmark entries (V010–V040, C110–C260).
- `ontic/platform/vv/harness.py` (~850 lines) — runs benchmark specs, computes QoIs, evaluates gates, emits both private and public proof packs.
- `contracts/v1/schemas/scorecard_public_v1.schema.json` — public scorecard JSON Schema.
- 5 promoted claim tags in `physics_os/core/evidence.py`: CONVERGENCE, REPRODUCIBILITY, ENERGY_BOUND, CFL_SATISFIED, BOUNDEDNESS.
- 55 tests in `test_vv_harness.py`.

**API remained frozen:** The bench harness runs as internal tooling — not as new endpoints. Consistent with API surface freeze ([§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract)).

---

#### E. Wall Strategy v1.5 — COMPLETED

**Primary files:**

| File | Role | Spec Reference |
|------|------|----------------|
| `ontic/vm/compilers/navier_stokes_2d.py` | NS2D compiler with Brinkman penalization | [§5.2](PLATFORM_SPECIFICATION.md#52-domain-compilers) |
| `ontic/vm/operators.py` | Near-wall operator support | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture) |
| `ontic/platform/vv/benchmarks.py` | Channel/cavity-type validations | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |

**Delivered:**
- `ontic/vm/models/wall_model.py` (~607 lines) — domain-agnostic wall model: `WallModelConfig`, `WallFields`, `WallModel` with penalization, diagnostics, and IR generation.
- `ontic/platform/vv/wall_benchmarks.py` (~603 lines) — Ghia cavity tables (Re=100/400/1000), Schäfer & Turek cylinder reference, 5 benchmarks (W010–W050).
- Extended `navier_stokes_2d.py` with Brinkman penalization support, `wall_model` parameter, `bc_kind` selection.
- Wall-model internals kept out of public outputs per IP boundary ([§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs)).
- 57 tests in `test_wall_model.py`.

---

#### F. UDv2 Physics Breadth (Compressible + CHT + Multiphase Lane) — COMPLETED

**Primary files:**

| Action | Location | Spec Reference |
|--------|----------|----------------|
| New compilers | `ontic/vm/compilers/` | [§5.2](PLATFORM_SPECIFICATION.md#52-domain-compilers): designated compiler location |
| Extended operators | `ontic/vm/operators.py`, `ontic/vm/gpu_operators.py` | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture) |
| Extended V&V suite | `ontic/platform/vv/` | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework): same module set |

**Delivered:**
- `ontic/vm/compilers/compressible_euler.py` (~490 lines) — `CompressibleEuler1DCompiler` with exact Riemann solver (Newton-Raphson), 3 IC types (Sod, Shu-Osher, smooth-sine), boundedness predicates (ρ > 0, p > 0), conservation balance tracking.
- `ontic/vm/compilers/cht_coupling.py` (~480 lines) — `CHTCompiler1D` with variable-coefficient heat equation (ρCp ∂T/∂t = ∇·(k∇T) + Q), tanh interface blending, thermal energy and interface flux diagnostics.
- `ontic/vm/compilers/phase_field.py` (~470 lines) — `PhaseField2DCompiler` implementing Cahn-Hilliard 2D with Ginzburg-Landau free energy, circle-droplet and Rayleigh-Taylor ICs, interface energy and phase fraction tracking.
- `BOUNDEDNESS` evidence claim tag added to `physics_os/core/evidence.py`.
- 58 tests in `test_physics_breadth.py`.

---

#### G. Hybrid Local Corrections + QoI-Driven Adaptivity — COMPLETED

**Primary files:**

| File | Role | Spec Reference |
|------|------|----------------|
| `ontic/vm/rank_governor.py` | Adaptive truncation policy | [§5.3](PLATFORM_SPECIFICATION.md#53-rank-governor): first-class component |
| `ontic/vm/runtime.py`, `ontic/vm/gpu_runtime.py` | Apply policies consistently | [§5.1](PLATFORM_SPECIFICATION.md#51-vm-architecture), [§5.4](PLATFORM_SPECIFICATION.md#54-gpu-runtime) |
| `ontic/platform/vv/convergence.py` | Validate adaptivity against QoI targets | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework) |

**Delivered:**
- `ontic/vm/hybrid_field.py` (~550 lines) — `HybridField` (q = q_TT + q_local), `LocalTile` (dense correction with cosine-taper blending), `FeatureSensor` (gradient magnitude, jump indicator, curvature, phase gradient), `TileActivationPolicy` (budget enforcement, pruning), `HybridRoundPolicy` (aggressive backbone compression when tiles active).
- `ontic/vm/qoi_adaptivity.py` (~450 lines) — `QoITarget` (frozen dataclass), `ConvergenceTrend` (CONVERGING/CONVERGED/STAGNATING/DIVERGING/INSUFFICIENT_DATA), `QoIHistory` (multi-QoI tracking), `AdaptiveRankPolicy` (per-field rank/tolerance tuning: INCREASE/DECREASE/HOLD based on QoI convergence, respects ceiling/floor).
- 65 tests in `test_hybrid_adaptivity.py`.

**Determinism constraint met:** All adaptivity policies are deterministic given seed/config, classified under determinism tiering ([§20.2](PLATFORM_SPECIFICATION.md#202-determinism-contract)).

---

## Appendix: Platform Spec Cross-Reference Index

Every claim in this document traces to a specific Platform Specification section:

| Claim | Platform Spec Section |
|-------|-----------------------|
| Sanitizer is sole exit path; whitelist-only | [§20.4 IP Boundary & Forbidden Outputs](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs) |
| 25 forbidden field categories | [§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs) |
| API surface frozen (9 endpoints, schemas, error codes) | [§20.1 API Surface Contract](PLATFORM_SPECIFICATION.md#201-api-surface-contract) |
| Additive-only changes within major version | [§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract) |
| Three-tier determinism envelope | [§20.2 Determinism Contract](PLATFORM_SPECIFICATION.md#202-determinism-contract) |
| VM architecture and module boundaries | [§5.1 VM Architecture](PLATFORM_SPECIFICATION.md#51-vm-architecture) |
| Domain compilers translate params into IR | [§5.2 Domain Compilers](PLATFORM_SPECIFICATION.md#52-domain-compilers) |
| Rank Governor adaptive truncation policy | [§5.3 Rank Governor](PLATFORM_SPECIFICATION.md#53-rank-governor) |
| GPU runtime dispatches to Triton/CUDA | [§5.4 GPU Runtime](PLATFORM_SPECIFICATION.md#54-gpu-runtime) |
| V&V Harness: convergence, MMS, conservation | [§13.3 V&V Framework](PLATFORM_SPECIFICATION.md#133-vv-framework) |
| CFD benchmark suite (8 validated cases) | [§13.1 CFD Benchmarks](PLATFORM_SPECIFICATION.md#131-cfd-benchmarks) |
| Claim-witness predicates (CONSERVATION, STABILITY, BOUND) | [§10 Trustless Physics Certificates](PLATFORM_SPECIFICATION.md#10-trustless-physics-certificates--tenet-tphy) |
| Reserved claim tags (CONVERGENCE, REPRODUCIBILITY, etc.) | [CLAIM_REGISTRY.md](docs/governance/CLAIM_REGISTRY.md) |
| Ed25519 certificate signing | [§10](PLATFORM_SPECIFICATION.md#10-trustless-physics-certificates--tenet-tphy), [§14.6](PLATFORM_SPECIFICATION.md#146-physics_os-package-breakdown) |
| 7-domain compiler registry | [§14.6 physics_os/ Package Breakdown](PLATFORM_SPECIFICATION.md#146-physics_os-package-breakdown) |
| Evidence validation + claim generation | [§14.6](PLATFORM_SPECIFICATION.md#146-physics_os-package-breakdown) |
| V&V harness location (`ontic/platform/vv/`) | [§6.2 Platform Substrate](PLATFORM_SPECIFICATION.md#62-platform-substrate-v200) |
| Never-dense guarantee | [§5.3 Rank Governor](PLATFORM_SPECIFICATION.md#53-rank-governor), [§1 Executive Summary](PLATFORM_SPECIFICATION.md#1-executive-summary) |
| CU formula (metering) | [§20.3 Metering & Pricing Contract](PLATFORM_SPECIFICATION.md#203-metering--pricing-contract) |
| ScorecardPublicV1 schema compliance | [§20.4](PLATFORM_SPECIFICATION.md#204-ip-boundary--forbidden-outputs) (forbidden fields), [§20.1](PLATFORM_SPECIFICATION.md#201-api-surface-contract) (fits ArtifactEnvelope) |
| Benchmark registry uses V&V harness methods | [§13.3](PLATFORM_SPECIFICATION.md#133-vv-framework), [§6.2](PLATFORM_SPECIFICATION.md#62-platform-substrate-v200) |
