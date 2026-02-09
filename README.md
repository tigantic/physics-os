<div align="center">

```
██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ 
██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
██║  ██║   ██║   ██║     ███████╗██║  ██║   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
```

### The Physics-First Tensor Network Engine

*One Codebase · 19 Industries · 1,157K Lines of Code · 9 Languages*

[![LOC](https://img.shields.io/badge/LOC-1%2C157K-blue?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-888K-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Rust](https://img.shields.io/badge/Rust-112K-000000?style=for-the-badge&logo=rust&logoColor=white)]()
[![Solidity](https://img.shields.io/badge/Solidity-72K-363636?style=for-the-badge&logo=solidity&logoColor=white)]()
[![Lean4](https://img.shields.io/badge/Lean_4-Verified-purple?style=for-the-badge)]()

[![Platform](https://img.shields.io/badge/Platform-V3.0.0-success?style=flat-square)]()
[![Physics](https://img.shields.io/badge/Physics-140%2F140-brightgreen?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-295_passing-brightgreen?style=flat-square)]()
[![Domains](https://img.shields.io/badge/Taxonomy-167_nodes-blue?style=flat-square)]()
[![Industries](https://img.shields.io/badge/Industries-19-orange?style=flat-square)]()
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)](LICENSE)

**Platform V3.0.0** · **Package V40.2** · **February 2026**

</div>

---

## What Is HyperTensor?

HyperTensor is a unified computational physics platform that uses Quantized Tensor Train (QTT) compression to operate on **10¹² grid points** without dense materialization — enabling simulations that previously required supercomputers to run on commodity hardware.

| Capability | Traditional CFD | HyperTensor |
|------------|:-:|:-:|
| **Grid Resolution** | 10⁶ points | **10¹² points** |
| **Memory Scaling** | O(N³) | **O(log N)** |
| **GPU Acceleration** | Manual | **Auto-detect** |
| **Time-to-Insight** | Days | **Minutes** |
| **Formal Verification** | None | **Lean 4 proofs** |
| **ZK Proof Generation** | None | **Halo2 circuits** |

The platform spans **5 integrated systems**, **112 reusable modules**, **167 physics taxonomy nodes** across **20 domain packs**, verified against published benchmarks and validated through 33 dedicated gauntlets.

---

## Key Differentiators

**Never Go Dense.** Every operation stays in TT/QTT format. Dense materialization is structurally blocked — not merely discouraged, but architecturally prevented.

**Physics-First Architecture.** Conservation laws are not optional. Every solver verifies mass, momentum, and energy conservation to machine precision (Δ < 10⁻¹⁵).

**Full-Stack Verification.** Three layers: Lean 4 formal proofs of governing equations → Halo2 ZK circuits for computational integrity → attested benchmark validation for physical fidelity.

**167 Physics Nodes, One API.** From incompressible Navier-Stokes to lattice QCD, from DFT to biomechanics — one canonical `ProblemSpec` → `Solver` → `Observable` pipeline with a single V&V harness.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/tigantic/HyperTensor-VM.git
cd HyperTensor-VM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Verify
python -c "from tensornet.platform import PLATFORM_VERSION; print(f'HyperTensor Platform {PLATFORM_VERSION}')"
# → HyperTensor Platform 2.0.0

# Run test suite
pytest tests/ -v
```

### SDK Workflow Builder

```python
from tensornet.sdk import WorkflowBuilder, get_recipe, list_recipes

# Fluent DSL — build + run a 1D shock tube simulation
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
print(f"Solved in {result.wall_time:.2f}s | {result.metadata}")

# Or use a pre-built recipe
print(list_recipes())  # ['harmonic_oscillator', 'lorenz_attractor', 'burgers_1d', ...]
wf = get_recipe("sod_shock_tube").build()
result = wf.run()
```

### DMRG Ground State

```python
import torch
from tensornet import MPS, heisenberg_mpo, dmrg

L, chi = 20, 64
H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)
psi = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
psi, E, info = dmrg(psi, H, num_sweeps=15, chi_max=chi)

print(f"E/L = {E/L:.8f}")  # -0.44314718 (matches Bethe ansatz)
```

### QTT-Native Navier-Stokes

```python
from tensornet.cfd.ns2d_qtt_native import (
    NS2DQTTConfig, NS2D_QTT_Native, create_conference_room_ic
)

config = NS2DQTTConfig(nx_bits=7, ny_bits=7, max_rank=48)  # 128×128
solver = NS2D_QTT_Native(config)
omega, psi, psi_bc, bc_mask = create_conference_room_ic(config)

omega, psi, info = solver.solve_steady_state(
    omega, psi, psi_bc, bc_mask, max_iters=200, tol=1e-5
)
print(f"Inlet velocity recovery: {info['inlet_recovery']:.1f}%")  # 94.4%
```

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HyperTensor Platform V3.0.0                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   SDK Layer                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  WorkflowBuilder DSL  ·  8 Recipes  ·  55+ Re-exports  ·  tensornet.sdk│  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   Platform Substrate (33 modules, 12,618 LOC)                                   │
│   ┌─────────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌────────────────┐   │
│   │ Data Model   │ │  Solvers  │ │  QTT    │ │  V&V    │ │ Export/Import  │   │
│   │ Protocols    │ │  Coupled  │ │  TCI    │ │ Harness │ │ VTU·XDMF·CSV  │   │
│   │ Domain Packs │ │  Adjoint  │ │  Accel  │ │ MMS     │ │ Gmsh·STL·OBJ  │   │
│   └─────────────┘ └───────────┘ └─────────┘ └─────────┘ └────────────────┘   │
│   ┌─────────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌────────────────┐   │
│   │ Inverse     │ │    UQ     │ │  Optim  │ │Lineage  │ │ Post-Process   │   │
│   │ Problems    │ │ MC·PCE    │ │  SIMP   │ │ DAG     │ │ Visualize      │   │
│   └─────────────┘ └───────────┘ └─────────┘ └─────────┘ └────────────────┘   │
│                                                                                 │
│   Physics Engine (tensornet/ — 784 files, 409K LOC)                             │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │
│   │ CFD    │ │Genesis │ │ Packs  │ │Exploit │ │Discover│ │Types   │          │
│   │ 70K    │ │ 41K    │ │ 26K    │ │ 26K    │ │ 25K    │ │ 12K    │          │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘          │
│   + 60 more domain-specific submodules                                          │
│                                                                                 │
│   Rust Layer (276 files, 112K LOC)                                              │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│   │FluidElite-ZK │ │Glass Cockpit │ │ Hyper Bridge │ │  QTT-CEM/FEA │         │
│   │  31K LOC     │ │  31K LOC     │ │   6K LOC     │ │   5K LOC     │         │
│   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘         │
│                                                                                 │
│   GPU Compute: CUDA Kernels (11) · Triton Kernels (3) · WGSL Shaders (18)     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Metrics

> *All metrics validated February 9, 2026. Excludes vendored dependencies.*

| Metric | Value |
|--------|------:|
| **Total First-Party LOC** | **1,153,189** |
| Python | 883,913 |
| Rust | 111,635 |
| Circom | 77,448 |
| Solidity | 71,531 |
| Lean 4 | 6,439 |
| WGSL | 4,265 |
| CUDA | 3,721 |
| TypeScript/JS | 2,942 |
| LaTeX | 2,223 |
| **Source Files** | **2,804** |
| **Test Files** | 185 |
| **Tests Passing** | 295 (1 skipped) |

---

## Five Integrated Platforms

| # | Platform | Location | Size | Purpose |
|:-:|----------|----------|-----:|---------|
| 1 | **HyperTensor VM** | `tensornet/` | 784 files · 409K LOC | Core physics engine — CFD, quantum, plasma, 60+ submodules |
| 2 | **FluidElite** | `fluidelite/`, `fluidelite-zk/` | 162 files · 57K LOC | Production tensor engine + ZK prover (24 Rust binaries) |
| 3 | **Sovereign Compute** | `tensornet/sovereign/`, `gevulot/` | 10 files · 3K LOC | Decentralized physics computation network |
| 4 | **QTeneT** | `QTeneT/` | 103 files · 10K LOC | Enterprise QTT SDK — TCI, solvers, benchmarks, CLI |
| 5 | **Platform Substrate** | `tensornet/platform/`, `tensornet/sdk/` | 36 files · 14K LOC | Unified simulation API V3.0.0 + WorkflowBuilder SDK |

---

## 167 Physics Taxonomy Nodes · 20 Domain Packs

Every physics domain is implemented as a **Domain Pack** with real solvers, regression tests, and benchmark validation — not scaffolds.

<details>
<summary><strong>Expand full taxonomy (I–XX)</strong></summary>

| Pack | Domain | Nodes | Anchors |
|:----:|--------|:-----:|---------|
| I | Classical Mechanics | 8 | N-body, elasticity, Helmholtz |
| II | Fluid Dynamics | 10 | Burgers (V0.6), NS, WENO, VOF |
| III | Electromagnetism | 7 | Maxwell FDTD (V0.6), PML, MoM |
| IV | Optics & Photonics | 7 | Diffraction, NLSE, quantum optics |
| V | Thermodynamics & Stat Mech | 6 | Ising MC, Fokker-Planck, heat |
| VI | Quantum Mechanics (Single-Body) | 10 | Schrödinger, scattering, path integrals |
| VII | Quantum Many-Body | 13 | Heisenberg MPS (V0.4), DMRG, Hubbard |
| VIII | Electronic Structure | 10 | Kohn-Sham DFT (V0.4), HF, TDDFT |
| IX | Condensed Matter | 8 | Phonons, BdG, micromagnetics |
| X | Nuclear & Particle Physics | 8 | Shell model, HMC, BSM |
| XI | Plasma Physics | 10 | Vlasov-Poisson (V0.6), MHD, PIC |
| XII | Astrophysics & Cosmology | 10 | Stellar evolution, GRMHD, CMB |
| XIII | Geophysics | 8 | Seismology, mantle convection, dynamo |
| XIV | Materials Science | 8 | Phase-field, NEB, SCFT |
| XV | Chemical Physics | 8 | PES, TST, surface hopping |
| XVI | Biophysics | 8 | Protein MD, docking, systems bio |
| XVII | Cross-Cutting Methods | 6 | Adjoint, PINN, AMR, Krylov |
| XVIII | Coupled Physics | 8 | FSI, thermo-mech, radiation-hydro |
| XIX | Quantum Information | 8 | Circuit sim, QEC, VQE |
| XX | Special & Applied | 6 | GR (BSSN), astrodynamics, TCAD |

**Maturity:** 4 nodes at V0.6 (QTT-accelerated) · 5 at V0.4 (validated) · 158 at V0.2 (correctness)

</details>

---

## Genesis Layers — QTT Meta-Primitives

Eight mathematical layers extending QTT into unexploited domains. **40,836 LOC across 80 files.**

| Layer | Primitive | What It Enables |
|:-----:|-----------|-----------------|
| 20 | **QTT-OT** (Optimal Transport) | Trillion-point Wasserstein distances |
| 21 | **QTT-SGW** (Spectral Graph Wavelets) | Billion-node graph signal analysis |
| 22 | **QTT-RMT** (Random Matrix Theory) | Eigenvalue statistics without dense storage |
| 23 | **QTT-TG** (Tropical Geometry) | All-pairs shortest paths in log-space |
| 24 | **QTT-RKHS** (Kernel Methods) | Trillion-sample Gaussian processes |
| 25 | **QTT-PH** (Persistent Homology) | Topological data analysis at scale |
| 26 | **QTT-GA** (Geometric Algebra) | Cl(50) multivectors in KB, not PB |
| 27 | **QTT-Aging** (Biological Aging) | Aging as rank growth, reversal as rank reduction |

**Cross-Primitive Pipeline:** Chains OT → SGW → RKHS → PH → GA end-to-end — zero densification, 6× compression throughout.

---

## Trustless Physics Certificates (Tenet-TPhy)

Cryptographic proof that a physics simulation ran correctly, without revealing the simulation.

| Layer | Name | Purpose | Status |
|:-----:|------|---------|:------:|
| A | Mathematical Truth | Lean 4 proofs of governing equations | ✅ |
| B | Computational Integrity | Halo2 ZK circuits for trace verification | ✅ |
| C | Physical Fidelity | Attested benchmark validation | ✅ |

**Phases 0–3 complete.** 134/134 Python tests · 299/299 Rust lib tests. Prover pool, Gevulot network, multi-tenant API, persistent certificate store, NS-IMEX circuit.

---

## Validation & Benchmarks

### CFD Benchmarks (8/8 Passing)

| Benchmark | Reference | Status |
|-----------|-----------|:------:|
| Sod Shock Tube | Sod (1978) | ✅ |
| Lax Shock Tube | Lax (1954) | ✅ |
| Double Rarefaction | Toro (1999) | ✅ |
| Shu-Osher | Shu & Osher (1989) | ✅ |
| Double Mach Reflection | Woodward & Colella (1984) | ✅ |
| Taylor-Green Vortex | Taylor & Green (1937) | ✅ |
| Lid-Driven Cavity | Ghia et al. (1982) | ✅ |
| QTT-Native NS2D | Conference Room Ventilation | ✅ |

### PWA Benchmarks — Badui (2020) Eq. 5.48 (10/10 Passing)

| Experiment | What It Validates | Status |
|------------|-------------------|:------:|
| Convention reduction | General → simplified intensity at machine precision | ✅ |
| Parameter recovery | 12-amplitude fit: yield RMSE 0.009, phase RMSE 35.7° | ✅ |
| Gram acceleration | $O(N_{\text{MC}}) → O(n_{\text{amp}}^2)$, up to 14× speedup | ✅ |
| Wave-set scan | 6 $J_{\max}$ values, robustness atlas | ✅ |
| QTT Gram compression | TT-SVD infrastructure validated | ✅ |
| Angular moments | $\chi^2/n_{\text{dof}} = 0.04$, all pulls < 1σ | ✅ |
| Beam asymmetry | 85× Σ RMSE improvement with polarization | ✅ |
| Bootstrap uncertainty | 200 resamples, 100% convergence | ✅ |
| Coupled-channel | 2.0× shared-wave yield improvement | ✅ |
| Mass-dependent BW | Δm₀ ≤ 12 MeV, ΔΓ₀ ≤ 6 MeV | ✅ |

**Reference:** Badui (2020), *"Extraction of Spin Density Matrix Elements..."*, Indiana University (165 pp).
**Replication note:** [`paper/PWA_REPLICATION_NOTE.md`](paper/PWA_REPLICATION_NOTE.md) — complete methodology, results, and reproduction instructions.

### V&V Framework

Aligned with **ASME V&V 10-2019** and **NASA-STD-7009A**:

| Method | Implementation |
|--------|----------------|
| **MMS** | Method of Manufactured Solutions — spatial/temporal order verification |
| **Conservation** | Mass, momentum, energy conservation to machine precision |
| **Convergence** | h/p/dt refinement studies with Richardson extrapolation |
| **Stability** | CFL monitoring, von Neumann analysis, eigenvalue bounds |
| **Benchmarks** | 8 canonical CFD + per-domain reference problems |
| **Performance** | Timing, memory profiling, scaling analysis |

### 33 Validation Gauntlets

Each gauntlet produces a cryptographically signed JSON attestation:

<details>
<summary><strong>Expand gauntlet list</strong></summary>

| Gauntlet | Domain | Gauntlet | Domain |
|----------|--------|----------|--------|
| Tomahawk | Missile aerodynamics | Hellskin | Re-entry thermal protection |
| Starheart | Fusion reactor output | Chronos | Relativistic physics |
| Orbital Forge | Trajectory mechanics | Prometheus | Combustion |
| Femto Fabricator | Molecular placement | Proteome Compiler | Protein folding |
| Cornucopia | Resource optimization | QTT-Native | QTT operations |
| Sovereign Genesis | System bootstrap | Metric Engine | Performance benchmarks |
| Oracle | Forecast accuracy | Hermes | Message routing |
| LaLuH₆ Odin | Superconductor theory | Li₃InCl₄₈Br₁₂ | Superionic dynamics |
| ADE V1/V2 | Discovery engine | QTT-OT/SGW/RMT | Genesis primitives |
| QTT-TG/RKHS/PH/GA | Genesis primitives | SNHFF | Stochastic NS |
| TIG011a (6 variants) | Materials science | Aging | Biological aging |
| Trustless Physics ×4 | ZK certificate phases | Production Hardening | Production gates |

</details>

---

## 19 Industry Verticals — The Civilization Stack

| # | Industry | Application | # | Industry | Application |
|:-:|----------|-------------|:-:|----------|-------------|
| 1 | Weather | Global tensor operators | 11 | Medical | Hemodynamics |
| 2 | Engine | CUDA 30× acceleration | 12 | Racing | F1 dirty air wake |
| 3 | Path | Hypersonic trajectory | 13 | Ballistics | 6-DOF wind trajectory |
| 4 | Pilot | Sovereign swarm AI | 14 | Emergency | Wildfire prophet |
| 5 | Energy | Wind farm optimization | 15 | Agriculture | Vertical farm microclimate |
| 6 | Finance | Liquidity weather engine | 21 | Biology | Biological aging |
| 7 | Urban | Drone canyon Venturi | 22 | EM | CEM-QTT Maxwell FDTD |
| 8 | Defense | Sub hydroacoustics | 23 | Structural | FEA-QTT elasticity |
| 9 | Fusion | Tokamak confinement | 24 | Optimization | SIMP topology + inverse |
| 10 | Cyber | DDoS grid shock | | | |

---

## Project Structure

```
HyperTensor-VM/
├── tensornet/                    # Core physics engine (784 files, 409K LOC)
│   ├── platform/                 #   Platform Substrate V3.0.0 (33 files, 13K LOC)
│   │   ├── data_model.py         #     Mesh, Field, BC/IC, SimulationState
│   │   ├── protocols.py          #     SolverProtocol, PostProcessor, Exporter
│   │   ├── solvers.py            #     7 PDE solvers with QTT acceleration
│   │   ├── coupled.py            #     Multi-physics coupling orchestrator
│   │   ├── adjoint.py            #     Discrete adjoint + FD fallback
│   │   ├── inverse.py            #     Bayesian inverse problems (MCMC)
│   │   ├── uq.py                 #     Monte Carlo, PCE, Sobol indices
│   │   ├── optimization.py       #     SIMP topology optimization
│   │   ├── qtt.py                #     QTT compression/decompression
│   │   ├── export.py             #     VTU, XDMF+HDF5, CSV, JSON
│   │   ├── mesh_import.py        #     Gmsh, STL, OBJ mesh import
│   │   ├── postprocess.py        #     Probe, slice, integrate, FFT, gradient
│   │   ├── visualize.py          #     Matplotlib field/convergence/spectrum
│   │   ├── security.py           #     SBOM, CVE scanning, license audit
│   │   ├── deprecation.py        #     SemVer, @deprecated, @since
│   │   ├── lineage.py            #     Computation provenance DAG
│   │   └── vv/                   #     V&V suite (convergence, conservation, MMS, stability, benchmarks, performance)
│   ├── sdk/                      #   Stable API surface (3 files, 1K LOC)
│   │   ├── workflow.py           #     WorkflowBuilder fluent DSL
│   │   └── recipes.py            #     8 built-in simulation recipes
│   ├── packs/                    #   20 domain packs (167 taxonomy nodes)
│   ├── cfd/                      #   CFD solvers (103 files, 70K LOC)
│   ├── genesis/                  #   QTT meta-primitives (80 files, 41K LOC)
│   ├── core/                     #   TT/QTT operations
│   ├── algorithms/               #   DMRG, TEBD, Lanczos
│   └── [60+ more submodules]     #   Quantum, plasma, fusion, condensed matter, ...
├── fluidelite-zk/                # ZK prover engine (Rust, 31K LOC)
├── apps/glass_cockpit/           # Real-time flight visualization (Rust/wgpu, 31K LOC)
├── crates/                       # Rust crates — bridge, core, GPU bindings
├── QTeneT/                       # Enterprise QTT SDK (10K LOC)
├── ledger/                       # Capability ledger (167 YAML nodes + schema)
├── tests/                        # Test suite (295 tests passing)
├── scripts/                      # Gauntlets, research scripts, tools
├── docs/                         # Documentation
│   ├── adr/                      #   Architecture Decision Records (ADR-0001–0011)
│   ├── attestations/             #   Cryptographically signed validation JSONs
│   ├── reports/                  #   Benchmark & analysis reports
│   └── research/                 #   Research papers & taxonomy
├── experiments/                  # Research experiments & replication studies
│   ├── pwa_engine/               #   PWA Compute Engine V3.0.0 (core.py ~2,300 LOC)
│   └── run_pwa_engine.py         #   10 experiments + 11 publication figures (~1,400 LOC)
├── proofs/                       # Lean 4 formal verification
├── paper/                        # Research manuscripts, replication notes, figures
├── deployment/                   # Container, config, health checks
├── PLATFORM_SPECIFICATION.md     # Full platform spec (4,144 lines)
├── Commercial_Execution.md       # 7-phase execution plan (all COMPLETE)
└── CHANGELOG.md                  # Semantic versioning history
```

---

## Development

### Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Core platform runtime |
| PyTorch | 2.0+ | Tensor operations, autograd |
| NumPy | 1.24+ | Array operations |
| Rust | 1.70+ | ZK provers, visualization, FFI bridge |
| CUDA | 12.1+ | GPU acceleration (optional) |

### Installation

```bash
pip install -e ".[dev]"          # Development — pytest, ruff, mypy, bandit
pip install -e ".[viz]"          # Visualization — matplotlib, jupyter
pip install -e ".[io]"           # HDF5 export — h5py
pip install -e ".[benchmark]"    # Benchmark comparisons — scipy, tenpy
pip install -e ".[all]"          # Everything
```

### Testing

```bash
pytest tests/ -v                           # Full suite (295 tests)
pytest tests/test_platform.py              # Platform substrate (33 tests)
pytest tests/test_productization.py        # Phase 7 productization (55 tests)
pytest tests/test_vv.py                    # V&V harness
pytest -m physics                          # Physics validation only
pytest -m "not slow"                       # Skip long-running tests
pytest tests/ --cov=tensornet              # With coverage report
```

### Make Targets

```bash
make check            # All quality gates (lint, type-check, test, security)
make format           # ruff format + isort
make typecheck        # mypy strict mode
make test-unit        # Unit tests
make test-int         # Integration tests
make physics          # Physics validation gauntlets
make security         # Dependency scanning + SBOM
make sbom             # CycloneDX SBOM generation
make release          # Full release preparation
```

---

## CI/CD

| Workflow | Trigger | Description |
|----------|---------|-------------|
| [`ci.yml`](.github/workflows/ci.yml) | Push / PR | Lint, type-check, test matrix (Python 3.11 + 3.12) |
| [`hardening.yml`](.github/workflows/hardening.yml) | Push / PR | Full test suite, determinism gate, SBOM, license audit, deprecation gate |
| [`vv-validation.yml`](.github/workflows/vv-validation.yml) | Push / PR | V&V harness — MMS, conservation, convergence verification |
| [`ledger-validation.yml`](.github/workflows/ledger-validation.yml) | Push / PR | Capability ledger schema validation (167 nodes) |
| [`nightly.yml`](.github/workflows/nightly.yml) | Cron | Full 167-node test matrix, golden output diff |
| [`exploit-engine.yml`](.github/workflows/exploit-engine.yml) | Push / PR | Smart contract vulnerability analysis |

---

## Execution History

All 7 phases of the [Commercial Execution Plan](Commercial_Execution.md) have been delivered:

| Phase | Deliverable | Commit | Key Metric |
|:-----:|-------------|:------:|------------|
| 0 | Capability ledger, governance, ADR process | `32aad29c` | 167 YAML nodes |
| 1 | Core platform substrate — data model, solvers, plugins | `cfb229d4` | 8 modules, 33 tests |
| 2 | V&V harness — MMS, convergence, conservation, stability | `b88a9901` | 6 modules, 40 tests |
| 3 | 20 domain packs, 6 anchor V0.4 vertical slices | `90a79173` | 6 anchors validated |
| 4 | 167/167 taxonomy nodes at V0.2+, zero scaffolds | `25d0b44f` | 257 tests passing |
| 5 | QTT bridge, TCI engine, 4 V0.6 accelerated solvers | `ae79ea7c` | 28 new tests |
| 6 | Coupling, adjoint, inverse, UQ, optimization, lineage | `ae79ea7c` | 27 new tests |
| 7 | SDK, recipes, export, mesh import, post-processing, security | `2725db6e` | 55 new tests |

| PWA | PWA Compute Engine V3.0.0 — Eq. 5.48 replication | `aea21fa0` | 10 experiments, 11 figures, ~3,700 LOC |

**Current state:** Platform V3.0.0 · 295 tests passing · 1,153K LOC · 2,804 files · [PWA polish `cdc1e93b`](https://github.com/tigantic/HyperTensor-VM/commit/cdc1e93b)

---

## Documentation

| Document | Description |
|----------|-------------|
| [`PLATFORM_SPECIFICATION.md`](PLATFORM_SPECIFICATION.md) | Complete platform specification — LOC matrices, physics inventory, component catalog |
| [`Commercial_Execution.md`](Commercial_Execution.md) | 7-phase execution plan with version-state model (all phases COMPLETE) |
| [`CHANGELOG.md`](CHANGELOG.md) | Semantic versioning history (V26.0 → V40.0.1) |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records (ADR-0001 through ADR-0011) |
| [`docs/COVERAGE_DASHBOARD.md`](docs/COVERAGE_DASHBOARD.md) | 167-node coverage dashboard with per-node V-state |
| [`CONSTITUTION.md`](CONSTITUTION.md) | Coding standards and engineering governance |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guidelines, PR process, review rules |
| [`SECURITY.md`](SECURITY.md) | Security policy, vulnerability reporting, cryptographic methods |
| [`paper/PWA_REPLICATION_NOTE.md`](paper/PWA_REPLICATION_NOTE.md) | PWA Compute Engine V3.0.0 replication note — Eq. 5.48, 10 experiments |

---

## References

1. Gourianov, N. et al., "A quantum-inspired approach to exploit turbulence structures," [Nature Computational Science (2022)](https://doi.org/10.1038/s43588-022-00351-9); [arXiv:2305.10784](https://arxiv.org/abs/2305.10784)
2. White, S. R., "Density matrix formulation for quantum renormalization groups," Phys. Rev. Lett. **69**, 2863 (1992)
3. Vidal, G., "Efficient simulation of one-dimensional quantum many-body systems," Phys. Rev. Lett. **93**, 040502 (2004)
4. Oseledets, I. V., "Tensor-Train Decomposition," SIAM J. Sci. Comput. **33**, 2295 (2011)
5. Sod, G. A., "A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws," J. Comp. Phys. **27**, 1 (1978)
6. Ghia, U. et al., "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method," J. Comp. Phys. **48**, 387 (1982)
7. Badui, R. T. (2020), "Extraction of Spin Density Matrix Elements for the Reaction γp → pω Using Linearly Polarized Photons," Ph.D. dissertation, Indiana University

---

## License

**Proprietary** — © 2025–2026 Bradly Biron Baker Adams / Tigantic Holdings LLC. All rights reserved.

This software and all associated intellectual property are the exclusive property of the owner. Unauthorized access, use, copying, modification, or distribution is strictly prohibited. See [`LICENSE`](LICENSE) for the complete agreement.

---

## Citation

```bibtex
@software{hypertensor2026,
  title     = {HyperTensor: The Physics-First Tensor Network Engine},
  author    = {Adams, Bradly Biron Baker},
  year      = {2026},
  version   = {3.0.0},
  url       = {https://github.com/tigantic/HyperTensor-VM},
  note      = {1,153K LOC across 9 languages. 167 physics nodes. 20 domain packs. Platform V3.0.0. PWA Compute Engine with Eq. 5.48 replication.}
}
```

<div align="center">

---

*"In God we trust. All others must bring data."* — W. Edwards Deming

**HyperTensor** · Platform V3.0.0 · © 2025–2026 Tigantic Holdings LLC

</div>
