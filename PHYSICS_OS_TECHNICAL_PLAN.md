<div align="center">

```
████████╗██╗  ██╗███████╗    ██████╗ ██╗  ██╗██╗   ██╗███████╗██╗ ██████╗███████╗     ██████╗ ███████╗
╚══██╔══╝██║  ██║██╔════╝    ██╔══██╗██║  ██║╚██╗ ██╔╝██╔════╝██║██╔════╝██╔════╝    ██╔═══██╗██╔════╝
   ██║   ███████║█████╗      ██████╔╝███████║ ╚████╔╝ ███████╗██║██║     ███████╗    ██║   ██║███████╗
   ██║   ██╔══██║██╔══╝      ██╔═══╝ ██╔══██║  ╚██╔╝  ╚════██║██║██║     ╚════██║    ██║   ██║╚════██║
   ██║   ██║  ██║███████╗    ██║     ██║  ██║   ██║   ███████║██║╚██████╗███████║    ╚██████╔╝███████║
   ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝ ╚═════╝╚══════╝     ╚═════╝ ╚══════╝
```

### Transformation Technical Plan

*From physics-os to The Physics OS — Powered by The Ontic Engine*

**Tigantic Holdings LLC** · **DBA: HolonomiX** · **February 2026**

**Status: ACTIVE** · **Document Version: 1.0.0**

</div>

---

## §1 Naming Taxonomy

The complete naming hierarchy from legal entity to code symbol:

```
┌─────────────────────────────────────────────────────────────┐
│  TIGANTIC HOLDINGS LLC                                      │  Legal entity
│  └── DBA: HolonomiX                                         │  Brand
│       └── The Physics OS                                    │  Platform / product
│            ├── The Ontic Engine                              │  Core runtime
│            │   ├── IR Compiler Pipeline                      │  QTT instruction set
│            │   ├── CPU + GPU Runtime                         │  Execution substrate
│            │   ├── Conservation Enforcer                     │  Physics constraints
│            │   ├── Rank Governor                             │  Compression control
│            │   └── Evidence Generator                        │  Trust certificates
│            ├── Physics Libraries (168 nodes)                 │  Domain physics
│            │   ├── CFD · EM · Plasma · Quantum · Bio · ...  │
│            │   └── Domain Packs (20 industries)              │
│            ├── Rust Substrate                                │  Performance layer
│            │   ├── Ontic Core (Rust)                         │  QTT / MPO ops
│            │   ├── Ontic Bridge (IPC)                        │  Python ↔ Rust
│            │   ├── Proof Bridge (ZK)                         │  Trust pipeline
│            │   └── QTT Solvers (CEM/FEA/OPT)                │  Native solvers
│            ├── Client Surfaces                               │  Access layer
│            │   ├── REST API                                  │  9 endpoints
│            │   ├── Python SDK                                │  sync + async
│            │   ├── CLI                                       │  5 commands
│            │   └── MCP Server                                │  11 AI tools
│            ├── Applications                                  │  Vertical products
│            │   ├── Glass Cockpit                             │  Flight viz
│            │   ├── Global Eye                                │  Weather platform
│            │   ├── FluidElite                                │  ZK-verified CFD
│            │   ├── The Compressor                            │  Compression tool
│            │   └── Sovereign Oracle                          │  Trustless oracle
│            └── Infrastructure                                │  Deployment
│                ├── Containerization                           │
│                ├── Telemetry                                  │
│                └── CI/CD (11 workflows)                       │
└─────────────────────────────────────────────────────────────┘
```

### Term Definitions

| Term | Definition | Scope |
|------|-----------|-------|
| **Tigantic Holdings LLC** | Delaware LLC. Legal owner of all IP. | Corporate |
| **HolonomiX** | DBA. The public-facing brand. | Brand |
| **The Physics OS** | The complete platform — everything a user touches. | Product |
| **The Ontic Engine** | *"The internal factory floor where physics are compiled, executed, and constrained by conservation laws."* The runtime that takes physics specifications and produces verified results. Formerly "The Ontic Engine." | Core |
| **Ontic** | "Of or relating to real existence" (ontology → ontic). Physics deals in what *is* — not simulations of reality, but computational structures that obey the same conservation laws as reality. | Etymology |
| **Physics OS** | The operating-system metaphor: physics modules are programs, the Ontic Engine is the kernel, client surfaces are the shell, and trust certificates are the audit log. | Metaphor |

---

## §2 Current Architecture → New Architecture

### 2.1 Current State (Pre-Transformation)

```
physics-os (monorepo)
├── physics_os/          Python product layer  (API, SDK, CLI, MCP, core)
│   ├── api/              REST API (FastAPI)
│   ├── sdk/              Python SDK
│   ├── cli/              CLI
│   ├── mcp/              MCP Server
│   ├── billing/          CU metering
│   ├── core/             Executor, registry, sanitizer, certificates, evidence
│   ├── contracts/        Pydantic schemas
│   └── jobs/             Job state machine
├── ontic/            Python engine layer  (VM + 100+ physics modules)
│   ├── engine/vm/        QTT IR, compilers, runtime, GPU runtime
│   ├── engine/*/         Adaptive, distributed, gateway, GPU, hardware, realtime
│   ├── cfd/              77K LOC — CFD solvers
│   ├── genesis/          42K LOC — Genesis layers
│   ├── packs/            26K LOC — Domain packs
│   ├── discovery/        25K LOC — Discovery engine
│   └── [93 more dirs]    Physics domains (acoustics → zk)
├── crates/               Rust substrate  (15 crates, 151K LOC)
│   ├── ontic_core/       Core QTT/MPO ops
│   ├── ontic_bridge/     Python↔Rust IPC
│   ├── proof_bridge/     ZK proof pipeline
│   ├── fluidelite_*/     FluidElite family (4 crates)
│   ├── qtt_*/            Native solvers (CEM, FEA, OPT)
│   └── tci_core*/        TCI implementation
├── apps/                 15 applications
├── products/             4 product lines
├── contracts/            Solidity smart contracts
├── proofs/               Lean 4 formal proofs
├── docs/                 Comprehensive documentation
├── tests/                Test suite (370+ tests)
├── deploy/               Container + telemetry
├── tools/                Development tooling
└── challenges/           Civilization Challenge pipelines
```

**Scale:** ~1,989K LOC · 5,882 files · 19 languages · 168 physics taxonomy nodes · 20 industries

### 2.2 New Architecture (The Physics OS)

The directory structure stays monorepo — no premature decomposition. The transformation is **naming, boundaries, and documentation**, not filesystem fragmentation.

```
The Physics OS (monorepo)
│
├── ╔══════════════════════════════════════════════════════════╗
│   ║  THE ONTIC ENGINE                                       ║
│   ║  "Where physics are compiled, executed, and constrained" ║
│   ╠══════════════════════════════════════════════════════════╣
│   ║                                                          ║
│   ║  ontic/                    Python engine package         ║
│   ║  ├── engine/vm/            IR + Compilers + Runtime      ║
│   ║  │   ├── ir.py             QTT instruction set           ║
│   ║  │   ├── compilers/        7 domain compilers            ║
│   ║  │   ├── runtime.py        CPU execution engine          ║
│   ║  │   ├── gpu_runtime.py    GPU execution (CUDA+Triton)   ║
│   ║  │   ├── rank_governor.py  Compression control           ║
│   ║  │   └── qtt_tensor.py     QTT tensor format             ║
│   ║  ├── engine/adaptive/      Adaptive refinement           ║
│   ║  ├── engine/distributed/   Distributed execution         ║
│   ║  ├── engine/gpu/           GPU acceleration              ║
│   ║  ├── engine/realtime/      Real-time streaming           ║
│   ║  ├── engine/substrate/     Low-level substrate           ║
│   ║  ├── cfd/                  CFD solvers (77K LOC)         ║
│   ║  ├── [100+ physics dirs]   168 taxonomy nodes            ║
│   ║  └── packs/                20 industry domain packs      ║
│   ║                                                          ║
│   ║  crates/                   Rust substrate (151K LOC)     ║
│   ║  ├── ontic_core/           Core QTT/MPO (renamed)        ║
│   ║  ├── ontic_bridge/         Python↔Rust IPC (renamed)     ║
│   ║  ├── proof_bridge/         ZK proof pipeline             ║
│   ║  ├── fluidelite_*/         FluidElite family             ║
│   ║  └── qtt_*/                Native solvers                ║
│   ║                                                          ║
│   ║  proofs/                   Lean 4 formal proofs          ║
│   ╚══════════════════════════════════════════════════════════╝
│
├── ╔══════════════════════════════════════════════════════════╗
│   ║  PLATFORM SHELL                                          ║
│   ║  "How users interact with the Ontic Engine"              ║
│   ╠══════════════════════════════════════════════════════════╣
│   ║                                                          ║
│   ║  physics_os/               Python platform package       ║
│   ║  ├── api/                  REST API (FastAPI)            ║
│   ║  ├── sdk/                  Python SDK (sync + async)     ║
│   ║  ├── cli/                  CLI (5 commands)              ║
│   ║  ├── mcp/                  MCP Server (11 AI tools)      ║
│   ║  ├── core/                 Execution fabric              ║
│   ║  │   ├── executor.py       Job orchestration             ║
│   ║  │   ├── registry.py       Domain registry               ║
│   ║  │   ├── sanitizer.py      IP boundary (whitelist)       ║
│   ║  │   ├── certificates.py   Trust certificate engine      ║
│   ║  │   └── evidence.py       Evidence generation           ║
│   ║  ├── billing/              CU metering                   ║
│   ║  ├── contracts/            API schemas (Pydantic)        ║
│   ║  └── jobs/                 Job state machine             ║
│   ╚══════════════════════════════════════════════════════════╝
│
├── ╔══════════════════════════════════════════════════════════╗
│   ║  APPLICATIONS                                            ║
│   ║  "Vertical products built on the Physics OS"             ║
│   ╠══════════════════════════════════════════════════════════╣
│   ║                                                          ║
│   ║  apps/                                                   ║
│   ║  ├── glass_cockpit/        Fighter cockpit viz           ║
│   ║  ├── global_eye/           Global weather platform       ║
│   ║  ├── sovereign_api/        Sovereign oracle API          ║
│   ║  ├── trustless_verify/     Certificate verifier          ║
│   ║  └── [10 more]                                           ║
│   ║                                                          ║
│   ║  products/                                               ║
│   ║  ├── fluidelite/           FluidElite (ZK-verified CFD)  ║
│   ║  ├── fluidelite-zk/        FluidElite ZK layer           ║
│   ║  ├── the_compressor/       Compression tool              ║
│   ║  └── facial_plastics/      Surgical planning             ║
│   ╚══════════════════════════════════════════════════════════╝
│
├── contracts/               Solidity smart contracts
├── deploy/                  Containerization + telemetry
├── docs/                    Platform documentation
├── tests/                   Test suite
├── tools/                   Development tooling
├── challenges/              Civilization Challenge pipelines
└── experiments/             R&D experiments
```

---

## §3 The Ontic Engine — Detailed Specification

### 3.1 What It Is

The Ontic Engine is the computational heart of the Physics OS. It is the **execution environment** where:

1. **Physics specifications are compiled** — Domain compilers (Burgers, Maxwell, Schrödinger, Navier-Stokes, etc.) translate physics problems into the QTT Intermediate Representation
2. **Conservation laws are enforced** — Every time step verifies mass, momentum, and energy conservation to machine precision (Δ < 10⁻¹⁵)
3. **Results are cryptographically attested** — Lean 4 proofs → Halo2 ZK circuits → Ed25519 trust certificates

### 3.2 Boundary Definition

```
                ╔═══════════════════════════════════════════╗
                ║           THE ONTIC ENGINE                 ║
                ║                                           ║
 Job Request ──▶║  IR Compiler ─▶ Runtime ─▶ Conservation  ║──▶ Attested Result
  (physics       ║       │            │          Check       ║    (sanitized +
   spec)         ║       ▼            ▼            │         ║     signed)
                ║  Domain       GPU/CPU         Rank        ║
                ║  Packs        Kernels       Governor      ║
                ║       │            │            │         ║
                ║       ▼            ▼            ▼         ║
                ║  168 Physics   CUDA/Triton   Compression  ║
                ║  Modules       Operators     Control      ║
                ╚═══════════════════════════════════════════╝
                                    │
                                    ▼
                          Evidence + Certificate
```

**Inside the engine boundary:**
- QTT IR definition and instruction set
- Domain compilers (7 registered, extensible)
- CPU runtime + GPU runtime (CUDA, Triton)
- All 168 physics taxonomy nodes and their implementations
- Rank governor and compression control
- Conservation law enforcement
- Rust substrate (ontic_core, ontic_bridge, QTT solvers)
- Lean 4 formal proofs and Halo2 ZK circuits

**Outside the engine boundary (in the Platform Shell):**
- Authentication + authorization
- Rate limiting
- IP sanitization
- CU metering / billing
- Job lifecycle management
- Client surfaces (API, SDK, CLI, MCP)
- Applications and products

### 3.3 The "Never Go Dense" Guarantee

The Ontic Engine's defining architectural constraint: **all operations remain in Tensor Train format**. Dense materialization is structurally blocked, not merely discouraged. This is what enables 10¹² degrees of freedom on commodity hardware with O(log N) memory scaling.

### 3.4 The Trust Pipeline

```
Physics Problem
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LAYER A     │────▶│  LAYER B     │────▶│  LAYER C     │
│  Correctness │     │  Computation │     │  Provenance  │
│              │     │              │     │              │
│  Lean 4      │     │  Halo2 ZK    │     │  Ed25519     │
│  57+ theorems│     │  Circuits    │     │  Certificates│
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  "The equations       "This specific       "This output was
   are correct"        computation ran       produced by this
                       as claimed"           engine at this time"
```

---

## §4 Transformation Phases

### Phase Overview

| Phase | Name | Scope | Breaking Changes | Est. Effort |
|:-----:|------|-------|:----------------:|:-----------:|
| **0** | Identity Layer | Brand assets, README, docs, LICENSE, headers | None | 1 session |
| **1** | Python Package Rename | `ontic/` → `ontic/`, `physics_os/` → `physics_os/` | **YES** | 2-3 sessions |
| **2** | Rust Crate Rename | `ontic_core` → `ontic_core`, `ontic_bridge` → `ontic_bridge` | **YES** | 1 session |
| **3** | Repository Rename | `physics-os` → `physics-os` on GitHub | **YES** | 1 session |
| **4** | Documentation Overhaul | Full rebrand of all 30+ docs | None (content) | 1-2 sessions |
| **5** | CI/CD & Deployment | Workflow names, badge URLs, container tags | Operational | 1 session |

### 4.0 Phase 0 — Identity Layer (THIS SESSION)

**Goal:** Every human-facing surface says "The Physics OS" and "The Ontic Engine." No import paths change, no code breaks.

| # | Task | File(s) | Risk |
|:-:|------|---------|:----:|
| 0.1 | Update ASCII banner | README.md, PLATFORM_SPECIFICATION.md | None |
| 0.2 | Update tagline + descriptions | README.md, pyproject.toml | None |
| 0.3 | Update LICENSE header | LICENSE | None |
| 0.4 | Update ARCHITECTURE.md | ARCHITECTURE.md | None |
| 0.5 | Update ROADMAP.md | ROADMAP.md | None |
| 0.6 | Update CITATION.cff | CITATION.cff | None |
| 0.7 | Update VERSION banner | VERSION | None |
| 0.8 | Update Cargo.toml workspace header | Cargo.toml | None |
| 0.9 | Update .zenodo.json | .zenodo.json | None |
| 0.10 | Update physics_os/__init__.py docstring | physics_os/__init__.py | None |
| 0.11 | Update ontic/__init__.py docstring | ontic/__init__.py | None |
| 0.12 | Create this technical plan | PHYSICS_OS_TECHNICAL_PLAN.md | None |

**Invariant:** `pytest tests/ -x` must pass before and after Phase 0. Zero broken imports.

### 4.1 Phase 1 — Python Package Rename

**Goal:** Rename Python packages to match the new brand.

| Current | New | Size |
|---------|-----|------|
| `ontic/` | `ontic/` | ~471K LOC, 100+ modules |
| `physics_os/` | `physics_os/` | ~4K LOC, 8 modules |

**Migration Strategy:**

1. **Create compatibility shim packages** that re-export everything:
   ```python
   # ontic/__init__.py (shim)
   """Backward compatibility — use 'ontic' for new code."""
   import warnings
   warnings.warn("'tensornet' is deprecated. Use 'ontic'.", DeprecationWarning, stacklevel=2)
   from ontic import *  # noqa: F401,F403
   ```

2. **Rename directories** in a single atomic commit:
   ```bash
   git mv tensornet ontic
   git mv hypertensor physics_os
   ```

3. **Bulk update all imports** with `sed` + manual review:
   ```bash
   find . -name '*.py' -exec sed -i 's/from ontic/from ontic/g; s/import ontic/import ontic/g' {} +
   find . -name '*.py' -exec sed -i 's/from physics_os/from physics_os/g; s/import physics_os/import physics_os/g' {} +
   ```

4. **Create shim packages** at old paths for backward compatibility:
   ```
   ontic/         (shim → ontic)
   physics_os/       (shim → physics_os)
   ```

5. **Update pyproject.toml:**
   ```toml
   [project]
   name = "physics-os"
   description = "The Physics OS — Powered by The Ontic Engine"

   [tool.setuptools.packages.find]
   include = ["ontic*", "physics_os*", "tensornet*", "hypertensor*"]

   [project.scripts]
   physics-os = "physics_os.cli:main"
   ontic = "ontic.platform.cli:main"
   ```

6. **Update Rust crate names** and PyO3 module names.

7. **Run full test suite** — every test must pass.

**Risk:** HIGH. This is the most dangerous phase. Every import in 1,425 files changes.

**Mitigation:** 
- Compatibility shims ensure old imports work with deprecation warnings
- Atomic commit ensures git bisect works
- Full test suite runs before merge

### 4.2 Phase 2 — Rust Crate Rename

| Current | New |
|---------|-----|
| `crates/ontic_core/` | `crates/ontic_core/` |
| `crates/ontic_bridge/` | `crates/ontic_bridge/` |
| `crates/ontic_gpu_py/` | `crates/ontic_gpu_py/` |

Update `Cargo.toml` workspace members, all `use` statements, and the PyO3 module registration.

### 4.3 Phase 3 — Repository Rename

1. Rename on GitHub: `tigantic/physics-os` → `tigantic/physics-os`
2. GitHub auto-redirects old URLs
3. Update all badge URLs, CI workflow references, documentation links
4. Update `pyproject.toml` URLs section
5. Update `.github/` workflow files

### 4.4 Phase 4 — Documentation Overhaul

Full rebrand of all documentation. Every instance of "The Ontic Engine" becomes either "The Physics OS" (platform context) or "The Ontic Engine" (runtime context).

**Priority order:**
1. README.md (first thing anyone sees)
2. PLATFORM_SPECIFICATION.md (2,106 lines — the bible)
3. ARCHITECTURE.md (189 lines — system overview)
4. docs/ directory (30+ files)
5. Inline code comments and docstrings
6. Challenge documents
7. Remaining markdown files

### 4.5 Phase 5 — CI/CD & Deployment

- Rename GitHub Actions workflow display names
- Update container image tags (`ghcr.io/tigantic/physics-os`)
- Update badge URLs
- Update mkdocs.yml site name
- Update deploy/config/ references

---

## §5 Package Identity Map

### 5.1 Python Packages

| Package | Purpose | PyPI Name | Import Path |
|---------|---------|-----------|-------------|
| **ontic** | The Ontic Engine — physics VM + libraries | `ontic-engine` | `import ontic` |
| **physics_os** | Platform shell — API, SDK, CLI, MCP | `physics-os` | `import physics_os` |
| **tensornet** | Backward compatibility shim → ontic | (alias) | `import tensornet` |
| **hypertensor** | Backward compatibility shim → physics_os | (alias) | `import physics_os` |

### 5.2 Rust Crates

| Crate | Purpose | crates.io Name |
|-------|---------|----------------|
| **ontic_core** | Core QTT/MPO operations | `ontic-core` |
| **ontic_bridge** | Python↔Rust IPC (mmap) | `ontic-bridge` |
| **ontic_gpu_py** | GPU PyO3 bridge | `ontic-gpu-py` |
| **proof_bridge** | ZK proof pipeline | `proof-bridge` |
| **tci_core** | TCI Rust implementation | `tci-core` |
| **fluidelite_*** | FluidElite family (4 crates) | (unchanged) |
| **qtt_*** | Native solvers (CEM/FEA/OPT) | (unchanged) |

### 5.3 Version Strategy

| Dimension | Current | Post-Transform |
|-----------|---------|----------------|
| Release | 4.0.0 | **5.0.0** (major bump — brand change) |
| Platform | 3.0.0 | **4.0.0** |
| Package | 40.0.1 | **50.0.0** |
| API Contract | 1 | 1 (frozen — no breaking API changes) |

The major version bump signals to all consumers that this is a new era. The API contract stays frozen at v1 — existing integrations continue to work.

---

## §6 Brand Assets

### 6.1 ASCII Banner (New)

```
████████╗██╗  ██╗███████╗     ██████╗ ███╗   ██╗████████╗██╗ ██████╗
╚══██╔══╝██║  ██║██╔════╝    ██╔═══██╗████╗  ██║╚══██╔══╝██║██╔════╝
   ██║   ███████║█████╗      ██║   ██║██╔██╗ ██║   ██║   ██║██║     
   ██║   ██╔══██║██╔══╝      ██║   ██║██║╚██╗██║   ██║   ██║██║     
   ██║   ██║  ██║███████╗    ╚██████╔╝██║ ╚████║   ██║   ██║╚██████╗
   ╚═╝   ╚═╝  ╚═╝╚══════╝     ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝

███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗
██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝
█████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  
██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  
███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗
╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝
```

### 6.2 Short Descriptions

| Context | Text |
|---------|------|
| **One-liner** | The Physics OS — Powered by The Ontic Engine |
| **Tagline** | *Where physics are compiled, executed, and constrained by conservation laws.* |
| **Technical** | Compression-native computational physics platform with 10¹² DoF on commodity hardware and cryptographic trust certificates. |
| **README Hero** | One codebase. ~2 million lines of code. 20 industries. Cryptographic proof that the physics is real. |
| **The Ontic Engine** | The internal factory floor where physics are compiled into QTT IR, executed on GPU/CPU runtimes, and constrained by conservation laws to machine precision. |

### 6.3 Naming Rules

1. **"The Physics OS"** — Always capitalize "The." It is not "physics OS" or "PhysicsOS." The article is part of the name.
2. **"The Ontic Engine"** — Same rule. Always "The Ontic Engine," never "ontic engine" in prose.
3. **In code:** Package names are lowercase: `ontic`, `physics_os`. Module references follow Python convention.
4. **In docs:** Use the full name with article on first mention, then "the engine" or "the platform" for subsequent references.
5. **HolonomiX** — Note the capital X. This is the public brand. Use in marketing, website, legal. Not in code.
6. **"The Ontic Engine"** — Legacy name. Deprecated. Backward compatibility shims use it. No new usage.

---

## §7 Migration Tooling

### 7.1 Automated Rename Script

Phase 1 will be driven by a migration script (`tools/scripts/rebrand.py`) that:

1. Walks every `.py`, `.rs`, `.toml`, `.md`, `.yml`, `.yaml`, `.json` file
2. Applies context-aware replacements:
   - `from tensornet` → `from ontic`
   - `import tensornet` → `import ontic`
   - `from hypertensor` → `from physics_os`
   - `import physics_os` → `import physics_os`
   - `"tensornet` (in pyproject.toml strings) → `"ontic`
   - `The Ontic Engine` (in docs/comments) → Context-dependent: "The Physics OS" or "The Ontic Engine"
3. Generates a diff report for manual review
4. Preserves git blame via `git mv` where possible

### 7.2 Verification Gate

After each phase, the following must pass:

```bash
# Python tests
PYTHONPATH="$PWD:$PYTHONPATH" python3 -m pytest tests/ -x --timeout=120

# Rust tests
cargo test --workspace

# Import verification
python3 -c "import ontic; print(physics_os.__version__)"
python3 -c "import physics_os; print(physics_os.__version__)"
python3 -c "import tensornet; print(tensornet.__version__)"  # shim
python3 -c "import physics_os; print(physics_os.__version__)"  # shim

# Lint
ruff check .
```

---

## §8 Risk Register

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Mass rename breaks imports | **Critical** | Compatibility shim packages + full test suite |
| Git history becomes hard to trace | High | Use `git mv` for directory renames; `git log --follow` works |
| CI/CD badge URLs break | Medium | GitHub auto-redirects + update in Phase 5 |
| External consumers hit broken imports | Medium | Shim packages emit deprecation warnings, not errors |
| Cargo.toml crate references break | High | Atomic commit for Rust rename; workspace-wide test |
| Documentation goes stale | Medium | Phase 4 is a dedicated doc overhaul pass |
| Solidity contract names mismatch | Low | Contracts are deployment artifacts, not runtime |

---

## §9 Decision Log

| # | Date | Decision | Rationale |
|:-:|------|----------|-----------|
| 1 | 2026-02-28 | Keep monorepo structure | Premature decomposition adds complexity without value at current scale |
| 2 | 2026-02-28 | Phase the rename (5 phases) | Atomic "rename everything" risks catastrophic breakage |
| 3 | 2026-02-28 | Provide backward compatibility shims | Zero breaking changes for existing consumers |
| 4 | 2026-02-28 | Major version bump (v4 → v5) | Signals brand era change; API contract stays frozen at v1 |
| 5 | 2026-02-28 | Doc rename before package rename | Human-facing surfaces change first; code changes follow |
| 6 | 2026-02-28 | "Ontic" as engine name | Etymology: "of or relating to real existence" — physics deals in what *is* |

---

## §10 Execution Timeline

```
Week 0 (now)     Phase 0 — Identity Layer
                  ├── Brand assets, README, docs headers
                  ├── This technical plan
                  └── Commit: "brand: introduce The Physics OS / The Ontic Engine"

Week 1           Phase 4 — Documentation Overhaul (can run in parallel with Phase 1 prep)
                  ├── Full rebrand of all 30+ docs
                  └── PLATFORM_SPECIFICATION.md rewrite

Week 2           Phase 1 — Python Package Rename
                  ├── Create rebrand.py migration script
                  ├── Rename tensornet → ontic
                  ├── Rename hypertensor → physics_os
                  ├── Create backward compatibility shims
                  └── Full test verification

Week 3           Phase 2 — Rust Crate Rename
                  ├── Rename crates
                  ├── Update Cargo.toml
                  └── cargo test --workspace

Week 4           Phase 3 + 5 — Repository Rename + CI/CD
                  ├── GitHub repo rename
                  ├── Update all URLs
                  ├── CI/CD workflow updates
                  └── Final verification sweep
```

---

## §11 Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| All tests pass | `pytest tests/ -x` = 0 failures |
| No "The Ontic Engine" in brand surfaces | `grep -rn "The Ontic Engine" README.md ARCHITECTURE.md PLATFORM_SPECIFICATION.md` = 0 hits |
| Backward compatibility works | `import tensornet` and `import physics_os` succeed with deprecation warnings |
| New imports work | `import ontic` and `import physics_os` succeed |
| Documentation consistent | Every doc uses "The Physics OS" / "The Ontic Engine" terminology |
| CI/CD green | All 11 workflows pass |
| Version bumped | Release = 5.0.0, Package = 50.0.0, Platform = 4.0.0 |

---

## §12 Related Documents

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture (to be rebranded) |
| [PLATFORM_SPECIFICATION.md](PLATFORM_SPECIFICATION.md) | Complete platform specification |
| [API_SURFACE_FREEZE.md](API_SURFACE_FREEZE.md) | Frozen API contract |
| [ROADMAP.md](ROADMAP.md) | Release roadmap |
| [LICENSE](LICENSE) | Proprietary license |
| [CHANGELOG.md](CHANGELOG.md) | Release changelog |

---

<div align="center">

*This document is the source of truth for the Ontic Engine → Physics OS transformation.*
*All naming decisions, phasing, and migration strategies are governed by this plan.*

**Tigantic Holdings LLC** · **DBA: HolonomiX** · **Proprietary & Confidential**

</div>
