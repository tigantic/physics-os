<div align="center">

```
██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ 
██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
██║  ██║   ██║   ██║     ███████╗██║  ██║   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
```

### **The Planetary Operating System for Physics Computation**

*One codebase. 1.5 million lines of code. 20 industries. Cryptographic proof that the physics is real.*

<br/>

[![CI](https://github.com/tigantic/HyperTensor-VM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tigantic/HyperTensor-VM/actions/workflows/ci.yml)
[![Audit Gates](https://github.com/tigantic/HyperTensor-VM/actions/workflows/audit-gates.yml/badge.svg?branch=main)](https://github.com/tigantic/HyperTensor-VM/actions/workflows/audit-gates.yml)
[![Hardening](https://github.com/tigantic/HyperTensor-VM/actions/workflows/hardening.yml/badge.svg?branch=main)](https://github.com/tigantic/HyperTensor-VM/actions/workflows/hardening.yml)
[![Docs](https://github.com/tigantic/HyperTensor-VM/actions/workflows/docs.yml/badge.svg)](https://github.com/tigantic/HyperTensor-VM/actions/workflows/docs.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/tigantic/HyperTensor-VM/badge)](https://scorecard.dev/viewer/?uri=github.com/tigantic/HyperTensor-VM)

[![Release](https://img.shields.io/github/v/release/tigantic/HyperTensor-VM?style=for-the-badge&color=blue)](https://github.com/tigantic/HyperTensor-VM/releases/latest)
[![LOC](https://img.shields.io/badge/LOC-1.51M-blue?style=for-the-badge)](PLATFORM_SPECIFICATION.md)
[![Tests](https://img.shields.io/badge/Tests-370%2B_Passing-brightgreen?style=for-the-badge)](https://github.com/tigantic/HyperTensor-VM/actions/workflows/ci.yml)
[![V&V](https://img.shields.io/badge/V%26V-ASME_10--2019-gold?style=for-the-badge)](PLATFORM_SPECIFICATION.md)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-471K_LOC-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![Rust](https://img.shields.io/badge/Rust-151K_LOC-000000?style=flat-square&logo=rust&logoColor=white)]()
[![Lean 4](https://img.shields.io/badge/Lean_4-57%2B_Theorems-purple?style=flat-square)]()
[![CUDA](https://img.shields.io/badge/CUDA-GPU_Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)]()
[![OpenSSF Best Practices](https://img.shields.io/badge/OpenSSF-Best_Practices-green?style=flat-square&logo=opensourceinitiative&logoColor=white)](https://github.com/tigantic/HyperTensor-VM/blob/main/SECURITY.md)

</div>

---

## The Problem

Physics simulation at scale requires supercomputers. A single high-fidelity CFD run at 10⁹ grid points consumes terabytes of RAM, days of wall time, and six-figure cloud bills. This gatekeeps the most valuable engineering decisions on Earth — aircraft design, fusion reactor optimization, drug molecule simulation, surgical planning — behind institutions that can afford the compute.

## The Solution

**HyperTensor** replaces dense numerical arrays with **Quantized Tensor Trains (QTT)** — a mathematical compression that reduces memory from O(N³) to O(log N). A simulation that requires 1 TB of RAM in dense format fits in 64 MB. A grid of **4.4 × 10¹² degrees of freedom** runs on a single workstation.

This isn't an approximation. Conservation laws are verified to machine precision (Δ < 10⁻¹⁵). Every simulation produces a cryptographic trust certificate — a Lean 4-verified, Halo2-proven, Ed25519-signed proof that the physics actually ran correctly.

| | Traditional CFD | HyperTensor |
|---|:---:|:---:|
| **Maximum Grid** | ~10⁹ points | **4.4 × 10¹² points** |
| **Memory Scaling** | O(N³) | **O(log N)** |
| **Hardware** | HPC cluster | **Single workstation** |
| **Formal Verification** | None | **Lean 4 proofs** |
| **Proof of Computation** | None | **Halo2 ZK circuits** |
| **Trust Certificates** | None | **Ed25519-signed** |
| **Time-to-Insight** | Days–weeks | **Minutes** |

---

## The Product: Runtime Access Layer

The **HyperTensor Runtime** (`hypertensor/`, 31 files, 3,965 LOC) is the commercial surface over the physics engine — exposing simulation-as-a-service through four access surfaces while protecting all intellectual property behind a whitelist-only sanitization boundary.

### Four Ways In

```
                     ┌──────────────────────────────────────────┐
                     │           HyperTensor Runtime             │
                     │                                          │
   REST API ────────▶│  Auth · Rate Limit · Job Router          │
   Python SDK ──────▶│  IP Sanitizer · Evidence Generator       │──▶ Trust Certificate
   CLI ─────────────▶│  Ed25519 Certificates · CU Metering      │
   MCP (AI Agents) ─▶│  9 Endpoints · 7 Physics Domains         │
                     │                                          │
                     └──────────────────────────────────────────┘
```

| Surface | Description |
|---------|-------------|
| **REST API** | 9 frozen OpenAPI endpoints — submit jobs, poll results, retrieve signed certificates |
| **Python SDK** | Sync + async typed client with auth, polling, and local certificate validation |
| **CLI** | `hypertensor run` · `validate` · `attest` · `verify` · `serve` |
| **MCP Server** | 11 tools for AI agent workflows — agents can autonomously run physics simulations |

### Six-State Job Lifecycle

```
queued ──▶ running ──▶ succeeded ──▶ validated ──▶ attested
                   ╲
                    ──▶ failed
```

Every job that completes successfully produces three cryptographic artifacts:
1. **Sanitized Result** — physics output with all IP-sensitive internals removed
2. **Validation Report** — conservation, stability, and bound verification
3. **Trust Certificate** — Ed25519-signed, SHA-256-bound, independently verifiable

### Quick Start — API

```bash
# Start the server
uvicorn hypertensor.api.app:create_app --factory --host 0.0.0.0 --port 8000

# Submit a physics job
curl -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $HYPERTENSOR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"domain": "cfd", "job_type": "full_pipeline", "parameters": {"n_bits": 7, "n_steps": 100}}'

# Retrieve the signed trust certificate
curl http://localhost:8000/v1/jobs/{job_id}/certificate \
  -H "Authorization: Bearer $HYPERTENSOR_API_KEY"
```

### Quick Start — Python SDK

```python
from hypertensor.sdk.client import HyperTensorClient

client = HyperTensorClient(base_url="http://localhost:8000", api_key="...")

# Submit, wait, get certificate — one call
result = client.run_and_wait(
    domain="cfd",
    job_type="full_pipeline",
    parameters={"n_bits": 7, "n_steps": 100}
)

print(f"Job:         {result.job_id}")
print(f"Status:      {result.status}")           # "attested"
print(f"Certificate: {result.certificate.signature[:32]}...")
print(f"Verified:    {client.verify(result.certificate)}")  # True
```

### Quick Start — CLI

```bash
# Run a simulation end-to-end
hypertensor run --domain cfd --n-bits 7 --n-steps 100

# Verify a trust certificate
hypertensor verify certificate.json

# Start the API server
hypertensor serve --port 8000
```

---

## The Engine: 471K Lines of Physics

The core physics engine (`tensornet/`, 1,082 files, 471,534 LOC) implements a register-based bytecode VM for QTT computation. Domain specifications compile to an intermediate representation (IR) of QTT operations — no dense arrays are ever materialized.

### Seven Physics Domains

| Domain | Key | Governing Equation |
|--------|-----|-------------------|
| Viscous Burgers | `burgers` | u_t + uu_x = νu_xx |
| Maxwell TE | `maxwell` | ∂_t E = c∂_x H, ∂_t H = c∂_x E |
| Maxwell 3D | `maxwell_3d` | Full curl–curl system |
| Schrödinger | `schrodinger` | iℏψ_t = -(ℏ²/2m)ψ_xx + Vψ |
| Advection-Diffusion | `advection_diffusion` | u_t + cu_x = κu_xx |
| Vlasov-Poisson | `vlasov_poisson` | f_t + vf_x + (q/m)Ef_v = 0 |
| Navier-Stokes 2D | `navier_stokes_2d` | Incompressible vorticity-streamfunction |

### The Never-Dense Guarantee

Every operation stays in TT/QTT format. Dense materialization is architecturally blocked:

- **Rank Governor**: Adaptive bond dimension control after every rank-growing operation
- **Memory**: O(r² · log N) where r is the controlled bond dimension
- **GPU**: Auto-detect CUDA → Triton JIT for hot paths → graceful CPU fallback
- **Exascale**: Demonstrated at **16,384³ = 4.4 × 10¹² DOF** on commodity hardware

### SDK — WorkflowBuilder

```python
from tensornet.sdk import WorkflowBuilder

result = (
    WorkflowBuilder("sod_shock")
    .domain(shape=(400,), extent=((0.0, 1.0),))
    .field("rho", ic="step")
    .solver("PHY-II.2")
    .time(0.0, 0.2, dt=5e-4)
    .export("vtu", path="output/")
    .build()
    .run()
)
print(f"Solved in {result.wall_time:.2f}s")
```

---

## Trustless Physics — Cryptographic Proof of Computation

Three verification layers that establish mathematical certainty, not statistical confidence:

| Layer | Technology | What It Proves |
|:-----:|------------|---------------|
| **A** | **Lean 4** formal proofs | The governing equations are mathematically correct |
| **B** | **Halo2** ZK circuits | The computation trace matches the equations without revealing internals |
| **C** | **Ed25519** signed certificates | The simulation output is bound to the input and untampered |

### Phases 0–3: Complete

| Phase | Scope | Tests | Key Deliverable |
|:-----:|-------|:-----:|-----------------|
| 0 | Foundation | 25/25 | TPC binary format, computation trace, proof bridge, certificate generator |
| 1 | Single-Domain | 24/24 | Euler 3D Halo2 circuit, 12+ Lean 4 theorems |
| 2 | Multi-Domain | 45/45 | NS-IMEX circuit, 20+ Lean 4 theorems, REST API |
| 3 | Production Scale | 40/40 | Prover pool, Gevulot decentralized network, multi-tenant, WAL persistence |

**134/134 Python tests · 299/299 Rust tests · 57+ Lean 4 theorems**

---

## Genesis Layers — QTT Meta-Primitives

Eight mathematical extensions that push QTT into domains no one has compressed before. **40,836 LOC across 80 files**, all with dedicated gauntlet validation.

| Layer | Primitive | Scale Achieved |
|:-----:|-----------|---------------|
| 20 | **QTT-OT** — Optimal Transport | Trillion-point Wasserstein distances in O(r³ log N) |
| 21 | **QTT-SGW** — Spectral Graph Wavelets | Billion-node graph signal processing |
| 22 | **QTT-RMT** — Random Matrix Theory | Dense-free eigenvalue statistics |
| 23 | **QTT-TG** — Tropical Geometry | All-pairs shortest paths in log-space |
| 24 | **QTT-RKHS** — Kernel Methods | Trillion-sample Gaussian processes |
| 25 | **QTT-PH** — Persistent Homology | Betti numbers at scale |
| 26 | **QTT-GA** — Geometric Algebra | Cl(50) in KB, not PB |
| 27 | **QTT-Aging** — Biological Aging | Aging as tensor rank growth; reversal as rank reduction |

**Cross-Primitive Pipeline**: OT → SGW → RKHS → PH → GA — end-to-end, zero densification.

---

## 20 Industry Verticals

Every vertical has dedicated domain packs, validation gauntlets, and signed attestation artifacts.

| # | Industry | Application | # | Industry | Application |
|:-:|----------|-------------|:-:|----------|-------------|
| 1 | **Aerospace** | Hypersonic re-entry heat shields | 11 | **Medical** | Arterial hemodynamics |
| 2 | **Wind Energy** | Wake modeling, turbine twins | 12 | **Motorsport** | Dirty air / slipstream CFD |
| 3 | **Grid Energy** | Frequency response, SCADA | 13 | **Ballistics** | Long-range 6-DOF trajectory |
| 4 | **Fusion Energy** | Tokamak MHD confinement | 14 | **Emergency** | Wildfire spread prediction |
| 5 | **Finance** | Order book NS physics | 15 | **Agriculture** | Vertical farm microclimate |
| 6 | **Urban Planning** | Street canyon pollution | 16 | **Biology** | Epigenetic aging dynamics |
| 7 | **Agriculture** | HVAC + LED thermal | 17 | **Electromagnetics** | CEM-QTT Maxwell FDTD |
| 8 | **Materials** | Superconductor at 300K | 18 | **Structural** | FEA-QTT Hex8 elasticity |
| 9 | **Quantum** | Surface code error correction | 19 | **Optimization** | SIMP topology + inverse |
| 10 | **Drug Design** | QM/MM binding free energy | 20 | **Facial Plastics** | Surgical simulation (43K LOC, 941 tests) |

---

## Engineering Infrastructure

### Codebase Metrics

| Metric | Value |
|--------|------:|
| **Total Authored Lines** | 1,513,108 |
| **Source Files** | 3,679 |
| **Languages** | 19 |
| **Tests Passing** | 370+ |
| **Validation Gauntlets** | 33 |
| **Attestation JSONs** | 125+ |
| **Physics Taxonomy Nodes** | 168 |
| **Domain Packs** | 20 |

### Six Integrated Platforms

| # | Platform | Size | Purpose |
|:-:|----------|-----:|---------|
| 1 | **HyperTensor Runtime** | 3,965 LOC | Commercial API + SDK + CLI + MCP |
| 2 | **Physics VM** (`tensornet/`) | 471K LOC | QTT compute engine — 105 modules, 1,082 files |
| 3 | **FluidElite** (`crates/fluidelite*/`) | 57K LOC | Production tensor engine + ZK prover |
| 4 | **QTeneT** (`apps/qtenet/`) | 10K LOC | Enterprise QTT SDK |
| 5 | **Platform Substrate** | 13.7K LOC | Unified simulation API V2.0.0 |
| 6 | **Sovereign Compute** | 3K LOC | Decentralized physics via Gevulot + Substrate |

### Rust Workspace — 19 Members

| Crate | LOC | Purpose |
|-------|----:|---------|
| `fluidelite_zk` | 31,325 | ZK prover engine — Halo2, prover pool, Gevulot, multi-tenant |
| `glass_cockpit` | 30,608 | Flight visualization — wgpu, 18 WGSL shaders |
| `fluidelite_circuits` | 21,342 | Halo2 constraint system definitions |
| `fluidelite_infra` | 8,542 | Persistence, networking, deployment infrastructure |
| `hyper_bridge` | 5,917 | Python↔Rust IPC — mmap + protobuf, 9ms latency |
| + 14 more | ~33K | CEM, FEA, OPT solvers · TCI · GPU bindings · formal proofs |

### CI/CD — 11 Workflows

| Workflow | Function |
|----------|----------|
| `ci.yml` | Full quality gate: ruff lint → mypy strict → pytest → Rust clippy + test + fmt |
| `release.yml` | Tag-driven 4-stage pipeline: validate → build → publish (PyPI OIDC) → GitHub Release |
| `docs.yml` | MkDocs Material → GitHub Pages (27 doc directories, 25 ADRs) |
| `nightly.yml` | Nightly Rust benchmarks, performance regression tracking |
| `audit-gates.yml` | Version sync enforcement, forbidden output scanning |
| `hardening.yml` | Log security, certificate integrity, production gates |
| + 5 more | Contracts CI, exploit engine, facial plastics, ledger validation, V&V framework |

### Makefile — 30+ Targets

```bash
make check          # All quality gates (Python + Rust)
make test-unit      # Python unit tests
make test-physics   # Physics validation gates
make rs-check       # Rust: cargo fmt + clippy + test
make docs           # Build MkDocs Material site
make version-check  # Validate version sync (7 checkpoints)
make dep-graph      # Dependency graph (16 nodes, 34 edges)
make container      # Build production container
make package        # Build Python + Rust packages
```

### Observability

Production Prometheus + Grafana stack under `deploy/telemetry/`:
- **8 alert rules** across API, Jobs, and System groups
- Pre-provisioned Grafana dashboard (request rate, latency, job status, CU consumption)
- `docker compose -f deploy/telemetry/docker-compose.yml up -d`

### Supply Chain

| Control | Implementation |
|---------|----------------|
| **Dependabot** | Weekly updates for pip, cargo, and GitHub Actions |
| **CODEOWNERS** | 278 path-to-owner mappings — every PR requires domain-expert review |
| **Pre-commit** | ruff, bare-except detection, large file prevention, YAML/JSON validation |
| **PEP 561** | `tensornet/py.typed` — downstream type checker support |
| **Feature Flags** | 16 pip extras for domain-specific installation |

---

## Validation & Benchmarks

### CFD Benchmarks — 8/8 Passing

| Benchmark | Reference | Verified |
|-----------|-----------|:--------:|
| Sod Shock Tube | Sod (1978) | ✅ |
| Shu-Osher | Shu & Osher (1989) | ✅ |
| Taylor-Green Vortex | Taylor & Green (1937) | ✅ |
| Kida Vortex (3D Euler) | Brachet et al. (1983) | ✅ |
| Double Mach Reflection | Woodward & Colella (1984) | ✅ |
| Kelvin-Helmholtz | Chandrasekhar (1961) | ✅ |
| Lid-Driven Cavity | Ghia et al. (1982) | ✅ |
| Couette Flow | Analytical linear profile | ✅ |

### PWA — Badui (2020) Eq. 5.48 — 10/10 Passing

Full partial wave analysis replication from Badui (2020), *"Extraction of Spin Density Matrix Elements,"* Indiana University dissertation (165 pp). Convention reduction at machine precision, 14× Gram acceleration, beam asymmetry 85× improvement, bootstrap 100% convergence.

### 33 Validation Gauntlets

Each produces a **cryptographically signed attestation JSON** with SHA-256 commit binding:

> Hellskin (re-entry) · Tomahawk (missile CFD) · Starheart (fusion) · Chronos (TDVP) · Orbital Forge (trajectory) · Prometheus (combustion) · LaLuH₆ Odin (superconductor @ 300K) · Femto Fabricator (< 0.1Å) · Proteome (protein folding) · QTT-Native · Sovereign Genesis · 7× Genesis Layers · 4× Trustless Physics Phases · SNHFF · ADE V1/V2 · Production Hardening · + more

### V&V Framework — ASME V&V 10-2019 Aligned (3,755 LOC)

A dedicated verification and validation framework implementing the full ASME V&V 10-2019 methodology. Not a wrapper — a ground-up implementation with 8 specialized modules:

| Module | LOC | Methodology |
|--------|----:|-------------|
| `mms.py` | 305 | **Method of Manufactured Solutions** — inject known analytic solutions, verify solver converges to them |
| `conservation.py` | 358 | **Conservation tracking** — mass, momentum, energy balance at every timestep (Δ < 10⁻¹⁵) |
| `convergence.py` | 363 | **h/p/dt convergence** — Richardson extrapolation, observed-order computation, asymptotic range verification |
| `stability.py` | 490 | **Stability analysis** — CFL condition, von Neumann spectral radius, eigenvalue bounds |
| `performance.py` | 385 | **Performance regression** — wall time, throughput, memory high-water-mark, CI-gated |
| `benchmarks.py` | 410 | **Benchmark suite** — 8 published CFD references with L1/L2/L∞ norms against exact/reference data |
| `vv.py` | 832 | **V&V orchestrator** — pipeline runner, report generator, pass/fail adjudication |
| `vertical_vv.py` | 505 | **Industry-specific V&V** — per-vertical validation requirements with domain-expert review gates |

> **Every solver assertion in this codebase traces back to a V&V module.** Conservation violations, convergence regression, and stability breaches are CI-blocking — they cannot be merged.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   Client Tier                                                                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────────────────┐  │
│   │ REST API │  │  Python  │  │   CLI    │  │  MCP Server (11 AI tools)     │  │
│   └─────┬────┘  └─────┬────┘  └─────┬────┘  └───────────────┬────────────────┘  │
│         └──────────────┴─────────────┴───────────────────────┘                   │
│                                      │  HTTPS / Bearer Auth                       │
│   ┌──────────────────────────────────▼────────────────────────────────────────┐  │
│   │           HyperTensor Runtime Access Layer (v4.0.0, 3,965 LOC)            │  │
│   │   Auth · Rate Limit · Job Router · IP Sanitizer · Certificates · Metering │  │
│   └──────────────────────────────────┬────────────────────────────────────────┘  │
│                                      │  IP Boundary                               │
│   ┌──────────────────────────────────▼────────────────────────────────────────┐  │
│   │           Physics VM — QTT Execution Engine (29 files, 9.9K LOC)          │  │
│   │   IR · 7 Domain Compilers · Runtime · GPU Runtime · Rank Governor         │  │
│   └──────────────────────────────────┬────────────────────────────────────────┘  │
│                                      │                                            │
│   ┌──────────────────────────────────▼────────────────────────────────────────┐  │
│   │           tensornet/ Physics Engine (1,082 files, 471K LOC)               │  │
│   │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │  │
│   │   │  CFD   │ │Genesis │ │Exploit │ │ Packs  │ │Discover│ │Platform│    │  │
│   │   │  77K   │ │  42K   │ │  28K   │ │  26K   │ │  25K   │ │  15K   │    │  │
│   │   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │  │
│   │   + 99 more domain-specific modules                                      │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│   Rust Substrate (19 workspace members, 151K LOC)                                │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│   │FluidElite-ZK │ │Glass Cockpit │ │ Hyper Bridge │ │CEM / FEA /   │          │
│   │  31K LOC     │ │  31K LOC     │ │   6K LOC     │ │OPT  5K LOC   │          │
│   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘          │
│                                                                                  │
│   GPU Compute: CUDA Kernels (11) · Triton JIT (3) · WGSL Shaders (18)          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone
git clone https://github.com/tigantic/HyperTensor-VM.git
cd HyperTensor-VM

# Install (pick your scope)
pip install -e ".[all]"             # Everything
pip install -e ".[cfd,quantum]"     # CFD + Quantum only
pip install -e ".[dev,docs]"        # Development + documentation

# Verify
python -c "import tensornet; import hypertensor; print(f'tensornet {tensornet.__version__} | hypertensor {hypertensor.__version__}')"
# → tensornet 40.0.1 | hypertensor 40.0.1

# Run tests
make check                          # Full quality gate (Python + Rust)
pytest tests/ -v                    # 370+ tests
```

### 16 Feature Flags (pip extras)

```bash
pip install tensornet[cfd]              # CFD solvers
pip install tensornet[quantum]          # Quantum many-body
pip install tensornet[plasma]           # Plasma physics
pip install tensornet[materials]        # Condensed matter
pip install tensornet[aerospace]        # Aerospace & guidance
pip install tensornet[em]              # Electromagnetics
pip install tensornet[ml]             # ML surrogates
pip install tensornet[physics-all]     # All physics domains
pip install tensornet[dev]             # ruff, mypy, pytest, coverage
pip install tensornet[docs]            # MkDocs Material + mkdocstrings
pip install tensornet[viz]             # matplotlib, plotly
pip install tensornet[io]              # h5py, netCDF4, vtk
```

---

## Project Structure

```
HyperTensor-VM/
├── hypertensor/                    # Runtime Access Layer (31 files, 3,965 LOC)
│   ├── api/                        #   FastAPI server — 9 frozen endpoints
│   │   ├── routers/                #     jobs, validate, capabilities, contracts, health
│   │   ├── auth.py                 #     Bearer token + rate limiting (60 rpm)
│   │   └── config.py               #     Environment configuration
│   ├── core/                       #   Business logic
│   │   ├── hasher.py               #     SHA-256 content-addressed hashing
│   │   ├── registry.py             #     7-domain compiler registry
│   │   ├── executor.py             #     VM execution bridge
│   │   ├── sanitizer.py            #     Whitelist-only IP boundary
│   │   ├── evidence.py             #     Claim-witness predicate generator
│   │   └── certificates.py         #     Ed25519 signing + verification
│   ├── jobs/                       #   6-state machine + thread-safe store
│   ├── sdk/                        #   Typed sync + async client
│   ├── cli/                        #   CLI (run, validate, attest, verify, serve)
│   └── mcp/                        #   MCP server — 11 AI-agent tools
├── tensornet/                      # Physics Engine (1,082 files, 471K LOC)
│   ├── vm/                         #   Register-based QTT VM (IR, compilers, rank governor)
│   ├── cfd/                        #   Computational Fluid Dynamics (77K LOC)
│   ├── genesis/                    #   QTT Meta-Primitives — 8 layers (42K LOC)
│   ├── exploit/                    #   Smart contract vulnerability analysis (28K LOC)
│   ├── packs/                      #   20 Domain Packs, 168 taxonomy nodes (26K LOC)
│   ├── discovery/                  #   Autonomous Discovery Engine (25K LOC)
│   ├── platform/                   #   Platform Substrate V2.0.0 (15K LOC)
│   │   └── vv/                     #     V&V framework (MMS, conservation, convergence)
│   ├── sdk/                        #   WorkflowBuilder + recipes
│   └── ... (93 more modules)       #   Quantum, plasma, fusion, materials, ...
├── crates/                         # Rust workspace — 19 members (151K LOC)
│   ├── fluidelite_zk/              #   ZK prover (Halo2, Gevulot, multi-tenant)
│   ├── hyper_bridge/               #   Python↔Rust IPC (mmap + protobuf)
│   ├── qtt_cem/                    #   Maxwell FDTD solver (Q16.16)
│   ├── qtt_fea/                    #   Hex8 static elasticity (Q16.16)
│   ├── qtt_opt/                    #   SIMP topology optimization
│   └── ...                         #   + 14 more crates
├── apps/                           # Standalone applications
│   ├── glass_cockpit/              #   Flight visualization (Rust + WGSL, 31K LOC)
│   ├── qtenet/                     #   Enterprise QTT SDK (10K LOC)
│   └── trustless_verify/           #   Standalone certificate verifier
├── proofs/                         # Formal verification (Lean 4)
├── products/                       # Shipped products
│   └── facial_plastics/            #   Surgical simulation (43K LOC, 941 tests)
├── deploy/telemetry/               # Prometheus + Grafana observability stack
├── docs/                           # 27 subdirectories, 25 ADRs
├── tests/                          # 370+ tests across integration + unit suites
├── tools/                          # sync_versions.py, dep_graph.py, 75+ scripts
├── .github/workflows/              # 11 CI/CD workflows
├── .github/ISSUE_TEMPLATE/         # Bug report, feature request, template chooser
├── .github/PULL_REQUEST_TEMPLATE.md # PR checklist with physics-specific gates
├── .github/FUNDING.yml             # Sponsorship / investment links
├── .github/dependabot.yml          # Automated dependency updates (3 ecosystems)
├── .pre-commit-config.yaml         # 7 hook categories (ruff, secrets, conventional commits)
├── ARCHITECTURE.md                 # System architecture with Mermaid diagrams
├── ROADMAP.md                      # Product milestones and research frontiers
├── NOTICE                          # Third-party software attributions
├── CODEOWNERS                      # 278 domain-expert review mappings
├── Makefile                        # 30+ orchestration targets (uv auto-detect)
├── VERSION                         # Single source of truth for all versions
├── PLATFORM_SPECIFICATION.md       # 2,052-line master specification
└── Cargo.toml                      # Rust workspace manifest
```

---

## Version Integrity

All version numbers are validated by `tools/sync_versions.py` across 7 checkpoints:

| Namespace | Version | Source |
|-----------|:-------:|--------|
| Release | v4.0.1 | Infrastructure-hardened baseline |
| Package (tensornet) | 40.0.1 | `tensornet.__version__` |
| Package (hypertensor) | 40.0.1 | `hypertensor.__version__` |
| API Version | 2.0.0 | `hypertensor.API_VERSION` |
| Runtime Version | 1.0.0 | `hypertensor.RUNTIME_VERSION` |
| Cargo Workspace | 4.0.0 | `Cargo.toml` |
| CITATION | 4.0.0 | `CITATION.cff` |

```bash
$ python tools/sync_versions.py
All versions in sync.       # 7/7 OK
```

---

## Documentation

> **[Documentation Site](https://tigantic.github.io/HyperTensor-VM)** — Full API reference, guides, and ADRs powered by MkDocs Material.

| Document | Description |
|----------|-------------|
| [`PLATFORM_SPECIFICATION.md`](PLATFORM_SPECIFICATION.md) | **Master specification** — 24 sections, 7 appendices, 2,052 lines, every metric validated |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture — Mermaid diagrams, data flow, IP boundary, ADR index |
| [`ROADMAP.md`](ROADMAP.md) | Product roadmap — milestones, research frontiers, honest status assessment |
| [`CHANGELOG.md`](CHANGELOG.md) | Full semantic versioning history |
| [`NOTICE`](NOTICE) | Third-party software attributions and licenses |
| [`docs/adr/`](docs/adr/) | 25 Architecture Decision Records |
| [`docs/governance/`](docs/governance/) | Constitution, API freeze, determinism envelope, metering policy, forbidden outputs |
| [`docs/operations/`](docs/operations/) | Operations runbook, launch gate matrix, security operations |
| [`docs/product/`](docs/product/) | Pricing model, release notes, certificate test matrix, launch readiness |
| [`docs/strategy/`](docs/strategy/) | Commercial execution plan (7 phases complete), IP strategy |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines, PR process, review requirements |
| [`SECURITY.md`](SECURITY.md) | Security policy, vulnerability reporting |

---

## Security & Compliance Posture

### Standards Alignment

| Standard / Framework | Status | Implementation |
|----------------------|:------:|----------------|
| **ASME V&V 10-2019** | ✅ Aligned | 3,755 LOC dedicated V&V framework — 8 modules, all CI-gated |
| **OpenSSF Scorecard** | ✅ Active | Automated security health metrics — [live badge](https://scorecard.dev/viewer/?uri=github.com/tigantic/HyperTensor-VM) |
| **OpenSSF Best Practices** | ✅ Implemented | Security policy, vulnerability disclosure, code review, CI/CD |
| **Supply Chain (SLSA-adjacent)** | ✅ Enforced | Dependabot (3 ecosystems), CODEOWNERS (278 rules), pre-commit, pinned Actions |
| **NIST SP 800-218 (SSDF)** | ✅ Partial | Secure development lifecycle, automated testing, change management |

### Defense-in-Depth

| Layer | Control | Enforcement |
|:-----:|---------|-------------|
| **API** | Bearer tokens + HMAC-SHA256 per-client rate limiting (60 rpm) | Runtime |
| **IP Boundary** | Whitelist-only sanitization — 25 forbidden field categories | Every exit path |
| **Signing** | Ed25519 keypair — server-side only, never exported | Certificate generation |
| **Certificates** | SHA-256 content-addressed, independently verifiable offline | Every completed job |
| **Input** | Pydantic V2 strict mode on all API inputs — no coercion | Request ingestion |
| **Logs** | Zero API keys, zero signing material in any log output | CI-enforced scan |
| **Dependencies** | Dependabot weekly (pip + cargo + Actions), lockfile pinning | Automated PRs |
| **Review** | CODEOWNERS with 278 path-to-owner rules — domain-expert approval required | Every PR |
| **Secrets** | Pre-commit hooks with detect-secrets + bare-except prevention | Pre-push gate |
| **Formal** | Lean 4 theorem proving (57+ theorems) — mathematical correctness proofs | Build artifact |
| **ZK** | Halo2 circuits — computation integrity without revealing internals | Attestation layer |

Full policy: [`SECURITY.md`](SECURITY.md) · [`SECURITY_OPERATIONS.md`](docs/operations/SECURITY_OPERATIONS.md) · [`FORBIDDEN_OUTPUTS.md`](FORBIDDEN_OUTPUTS.md)

---

## References

1. Gourianov, N. et al., "A quantum-inspired approach to exploit turbulence structures," [Nature Computational Science (2022)](https://doi.org/10.1038/s43588-022-00351-9)
2. Oseledets, I. V., "Tensor-Train Decomposition," SIAM J. Sci. Comput. **33**, 2295 (2011)
3. White, S. R., "Density matrix formulation for quantum renormalization groups," Phys. Rev. Lett. **69**, 2863 (1992)
4. Vidal, G., "Efficient simulation of one-dimensional quantum many-body systems," Phys. Rev. Lett. **93**, 040502 (2004)
5. Sod, G. A., "A survey of several finite difference methods...," J. Comp. Phys. **27**, 1 (1978)
6. Ghia, U. et al., "High-Re solutions for incompressible flow...," J. Comp. Phys. **48**, 387 (1982)
7. Badui, R. T. (2020), "Extraction of Spin Density Matrix Elements...," Ph.D. dissertation, Indiana University

---

## License

**Proprietary** — © 2025–2026 Bradly Biron Baker Adams / Tigantic Holdings LLC. All rights reserved.

This software and all associated intellectual property are the exclusive property of the owner. Unauthorized access, use, copying, modification, or distribution is strictly prohibited. See [`LICENSE`](LICENSE).

---

## Citation

```bibtex
@software{hypertensor2026,
  title     = {HyperTensor: The Planetary Operating System for Physics Computation},
  author    = {Adams, Bradly Biron Baker},
  year      = {2026},
  version   = {4.0.1},
  url       = {https://github.com/tigantic/HyperTensor-VM},
  note      = {1.51M LOC. 20 industries. 168 physics nodes. Trustless certificates.
               Three-layer verification: Lean 4 + Halo2 ZK + Ed25519.}
}
```

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║      O N E   C O D E B A S E   ·   O N E   P H Y S I C S   E N G I N E                  ║
║                                                                                          ║
║      1 , 5 1 3 , 1 0 8   L I N E S   O F   C O D E                                      ║
║                                                                                          ║
║      2 0   I N D U S T R I E S   ·   1 6 8   T A X O N O M Y   N O D E S                ║
║                                                                                          ║
║      6   P L A T F O R M S   ·   1 9   R U S T   C R A T E S   ·   3 3   G A U N T L E T S ║
║                                                                                          ║
║      1 1   C I   W O R K F L O W S   ·   3 0 +   M A K E   T A R G E T S               ║
║                                                                                          ║
║      L E A N   4   ·   H A L O 2   ·   E D 2 5 5 1 9                                    ║
║                                                                                          ║
║                          T H E   P L A N E T A R Y   O S                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

**HyperTensor** · Release v4.0.1 · © 2025–2026 Tigantic Holdings LLC

*"In God we trust. All others must bring data."* — W. Edwards Deming

</div>
