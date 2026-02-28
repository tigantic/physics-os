# The Physics OS — Repository Map 🗺️

> Structural navigation tree for The Physics OS-VM codebase
> Generated: 2026-01-16 | 329 core modules | 51 gauntlets

---

## 📂 Root Structure

```
HyperTensor-VM-main/
├── 📁 ontic/           # Core library (329 modules)
├── 📁 demos/               # Demo scripts & evidence
├── 📁 apps/                # Applications (glass_cockpit, global_eye)
├── 📁 crates/              # Rust bridges (CUDA, core)
├── 📁 docs/                # API docs, architecture, tutorials
├── 📁 crates/fluidelite/          # Alternative fluid simulation stack
├── 📁 integrations/        # Unity, Unreal integrations
├── 📁 sdk/                 # Docker, Conda, Enterprise SDK
├── 📁 scripts/             # Setup, profiling, testing
├── 📁 tests/               # Integration tests
├── 📁 notebooks/           # Jupyter notebooks
├── 📁 proofs/              # Mathematical proofs
│
├── 🐍 51 Python files      # Gauntlets & solvers (see below)
├── 📄 *.md                 # Documentation & attestations
├── 📄 *.json               # Attestation records
└── 📄 Cargo.toml           # Rust workspace config
```

---

## 🔬 ontic/ — Core Library (329 modules)

### By Module Count

| Dir | Modules | Description |
|-----|---------|-------------|
| **cfd/** | 70 | CFD, QTT, Navier-Stokes, turbulence |
| **sovereign/** | 10 | Sovereign AI infrastructure |
| **hyperenv/** | 10 | Hypercomputing environment |
| **core/** | 10 | Core tensor operations |
| **fusion/** | 9 | Fusion reactor simulation |
| **ml_surrogates/** | 8 | ML/neural surrogates |
| **quantum/** | 7 | Quantum computing bridges |
| **provenance/** | 7 | Data provenance tracking |
| **intent/** | 7 | Intent recognition |
| **hypersim/** | 7 | Hyperparameter simulation |
| **gpu/** | 7 | GPU kernels, memory mgmt |
| **fieldos/** | 7 | Field operations OS |
| **benchmarks/** | 7 | Performance benchmarks |
| **validation/** | 6 | Validation utilities |
| **substrate/** | 6 | Substrate layer |
| **simulation/** | 6 | General simulation |
| **hypervisual/** | 6 | Visualization pipelines |
| **guidance/** | 6 | Guidance systems |
| **gateway/** | 6 | API gateway |
| **distributed/** | 6 | Distributed computing |
| **digital_twin/** | 6 | Digital twin framework |
| **algorithms/** | 6 | Core algorithms |
| **site/** | 5 | Site management |
| **realtime/** | 5 | Real-time processing |
| **neural/** | 5 | Neural network ops |
| **integration/** | 5 | Integration layer |
| **flight_validation/** | 5 | Flight validation |
| **docs/** | 5 | Internal docs |
| **distributed_tn/** | 5 | Distributed tensor networks |
| **cuda/** | 5 | CUDA kernels |
| **coordination/** | 5 | Multi-agent coordination |
| **autonomy/** | 5 | Autonomous systems |
| **physics/** | 4 | Physics engines |
| **mpo/** | 4 | Matrix Product Operators |
| **financial/** | 4 | Financial modeling |
| **deployment/** | 4 | Deployment tools |
| **defense/** | 4 | Defense applications |
| **adaptive/** | 4 | Adaptive algorithms |
| **urban/** | 3 | Urban simulation |
| **fuel/** | 3 | Fuel cell modeling |
| **energy/** | 3 | Energy systems |
| **data/** | 3 | Data utilities |
| **certification/** | 3 | Certification tools |
| **visualization/** | 2 | Visualization |
| **racing/** | 2 | Racing simulation |
| **numerics/** | 2 | Numerical methods |
| **mps/** | 2 | Matrix Product States |
| **medical/** | 2 | Medical applications |
| **fieldops/** | 2 | Field operations |
| **emergency/** | 2 | Emergency response |
| **cyber/** | 2 | Cybersecurity |
| **agri/** | 2 | Agricultural modeling |

---

### 🌊 ontic/cfd/ — CFD Engine (70 modules)

```
ontic/cfd/
├── __init__.py                     # Exports all
│
├── # QTT Core
├── qtt.py                          # QTT fundamentals
├── qtt_core.py                     # Extended core ops
├── qtt_mpo.py                      # Matrix Product Operators
├── qtt_factory.py                  # QTT creation utilities
│
├── # Physics (Phase 23)
├── qtt_spectral.py                 # WHT, energy spectrum
├── qtt_shift_stable.py             # Rank-preserving shift
├── qtt_hadamard.py                 # Element-wise multiplication
├── qtt_regularity.py               # Vorticity, BKM criterion
│
├── # Navier-Stokes
├── ns_qtt_solver.py                # QTT-based NS solver
├── ns_qtt_gpu.py                   # GPU-accelerated NS
├── ns_regularity_monitor.py        # Singularity monitoring
├── ns_slip_conditions.py           # Boundary conditions
├── ns_spectral_ops.py              # Spectral methods
├── ns_dimensional.py               # Dimensional analysis
├── ns_spectral_derivative.py       # Spectral derivatives
├── ns_spectral_gpu_solver.py       # GPU spectral solver
├── ns_compact_stencil.py           # Compact stencils
│
├── # Analysis & Diagnostics
├── chi_diagnostic.py               # χ-diagnostic for singularities
├── singularity_hunter.py           # Black Swan hunting
├── tensor_slicer.py                # 3D→2D visualization
├── local_enstrophy.py              # Local enstrophy
├── spectral_slope.py               # Energy cascade analysis
├── qtt_diagnostics.py              # General diagnostics
│
├── # Turbulence
├── smagorinsky_les.py              # LES Smagorinsky model
├── sgs_stress.py                   # Sub-grid stress
├── sgs_model.py                    # SGS modeling
├── turbulence_models.py            # Multiple turbulence models
├── energy_cascade.py               # Energy cascade
│
├── # GPU & Rendering
├── qtt_torch_renderer.py           # PyTorch rendering
├── qtt_cupy_ops.py                 # CuPy operations
├── cuda_qtt_contract.py            # CUDA contractions
├── qtt_contract_cuda.py            # CUDA contract alt
│
├── # Compression & Optimization
├── tci_compress.py                 # TCI compression
├── adaptive_rank.py                # Rank adaptation
├── cross_approximation.py          # Cross approximation
│
├── # Field Operations
├── field_3d.py                     # 3D field class
├── field_ops.py                    # Field operations
├── advection.py                    # Advection schemes
├── diffusion.py                    # Diffusion operators
├── projection.py                   # Pressure projection
├── poisson.py                      # Poisson solvers
│
├── # Boundary & Mesh
├── boundary.py                     # Boundary conditions
├── mesh.py                         # Mesh generation
├── amr.py                          # Adaptive mesh refinement
│
├── # Benchmarks
├── benchmark_qtt.py                # QTT benchmarks
├── benchmark_ns.py                 # NS benchmarks
├── kida_benchmark.py               # Kida vortex test
│
└── # Integration
    ├── hvac_cfd.py                 # HVAC simulation
    ├── thermal.py                  # Thermal solver
    ├── industrial_benchmark.py     # Industrial validation
    └── ...
```

---

### 🔮 ontic/quantum/ — Quantum Computing (7 modules)

```
ontic/quantum/
├── __init__.py
├── quantum_bridge.py               # Quantum-classical bridge
├── quantum_gates.py                # Gate implementations
├── quantum_circuits.py             # Circuit construction
├── quantum_optimization.py         # QAOA, VQE
├── quantum_simulation.py           # Quantum simulation
└── quantum_error.py                # Error correction
```

---

### ⚛️ ontic/fusion/ — Fusion Reactor Simulation (9 modules)

```
ontic/fusion/
├── __init__.py
├── tokamak_core.py                 # Tokamak physics
├── plasma_equilibrium.py           # MHD equilibrium
├── transport.py                    # Transport modeling
├── heating.py                      # RF/NBI heating
├── confinement.py                  # Confinement analysis
├── disruption.py                   # Disruption prediction
├── stellarator.py                  # Stellarator support
└── diagnostics.py                  # Diagnostic integration
```

---

### 🧠 ontic/ml_surrogates/ — Machine Learning (8 modules)

```
ontic/ml_surrogates/
├── __init__.py
├── neural_operator.py              # DeepONet, FNO
├── physics_informed.py             # PINNs
├── autoencoder.py                  # Latent compression
├── transformer.py                  # Attention-based
├── gnn_mesh.py                     # Graph neural networks
├── ensemble.py                     # Ensemble methods
└── active_learning.py              # Active sampling
```

---

### 🖥️ ontic/gpu/ — GPU Acceleration (7 modules)

```
ontic/gpu/
├── __init__.py
├── memory_manager.py               # VRAM management
├── kernel_launcher.py              # Kernel dispatch
├── cuda_streams.py                 # Stream management
├── tensor_core_ops.py              # Tensor Core ops
├── multi_gpu.py                    # Multi-GPU support
└── profiler.py                     # GPU profiling
```

---

## 🎯 Root Python Files — Gauntlets & Solvers (51)

### 🏆 Gauntlets (Validation Suites)

| File | Purpose |
|------|---------|
| `chronos_gauntlet.py` | Time-series validation |
| `cornucopia_gauntlet.py` | Materials discovery |
| `femto_fabricator_gauntlet.py` | Nanofabrication |
| `hellskin_gauntlet.py` | Thermal extremes |
| `hermes_gauntlet.py` | Communication systems |
| `laluh6_odin_gauntlet.py` | LaLuH6 superconductor |
| `li3incl48br12_superionic_gauntlet.py` | Superionic conductor |
| `metric_engine_gauntlet.py` | Metric validation |
| `oracle_gauntlet.py` | Prediction accuracy |
| `orbital_forge_gauntlet.py` | Orbital mechanics |
| `prometheus_gauntlet.py` | Fire/thermal dynamics |
| `proteome_compiler_gauntlet.py` | Protein folding |
| `snhff_stochastic_gauntlet.py` | Stochastic methods |
| `sovereign_genesis_gauntlet.py` | Sovereign AI genesis |
| `starheart_gauntlet.py` | Fusion validation |
| `tig011a_dielectric_gauntlet.py` | TIG011A material |
| `tomahawk_cfd_gauntlet.py` | Tomahawk CFD |

### 🔬 Solvers & Engines

| File | Purpose |
|------|---------|
| `hellskin_thermal_solver.py` | Extreme thermal |
| `hypertensor_dynamics.py` | Core dynamics engine |
| `odin_superconductor_solver.py` | Superconductor |
| `starheart_fusion_solver.py` | Fusion reactor |
| `ssb_optimize_resonance.py` | SSB resonance |
| `ssb_superionic_solver.py` | Superionic battery |
| `wilson_plaquette_engine.py` | Yang-Mills lattice |
| `real_yang_mills_engine.py` | Yang-Mills engine |

### 🔥 Black Swan Hunters

| File | Purpose |
|------|---------|
| `navier_stokes_black_swan.py` | Original hunter |
| `ns_qtt_singularity_hunt.py` | QTT-based hunt |
| `ns_unified_black_swan.py` | Unified approach |
| `ns_unified_black_swan_v2.py` | V2 refinement |
| `black_swan_level2.py` | Level 2 diagnostics |
| `navier_stokes_regularity_v2.py` | Regularity proofs |
| `kida_convergence_study.py` | Kida vortex study |
| `kida_high_res_verification.py` | High-res Kida |

### 🧮 Yang-Mills Proofs

| File | Purpose |
|------|---------|
| `elite_yang_mills_proof.py` | Elite proof v1 |
| `elite_yang_mills_proof_v2.py` | Elite proof v2 |
| `yang_mills_proof_pipeline.py` | Proof pipeline |
| `yang_mills_unified_proof.py` | Unified proof |

### 💊 TIG011A Drug Discovery

| File | Purpose |
|------|---------|
| `tig011a_attestation.py` | Attestation |
| `tig011a_docking_qmmm.py` | Docking QM/MM |
| `tig011a_dynamic_validation.py` | MD validation |
| `tig011a_multimechanism.py` | Multi-mechanism |
| `tig011a_tox_screen.py` | Toxicity screen |
| `tig011a_wiggle_tt.py` | Conformational |

### 🔑 Special Files

| File | Purpose |
|------|---------|
| `TURN_THE_KEY.py` | Sovereign activation |
| `ai_mathematician.py` | AI proof assistant |
| `navier_stokes_millennium_pipeline.py` | Millennium Prize |

---

## 🏛️ Supporting Directories

### 📁 apps/ — Applications

```
apps/
├── glass_cockpit/          # Real-time flight dashboard
│   ├── main.py
│   ├── telemetry.py
│   ├── displays.py
│   └── ...
└── global_eye/             # Global monitoring system
    ├── core.py
    ├── satellite.py
    └── ...
```

### 📁 crates/ — Rust Code

```
crates/
├── hyper_bridge/           # Rust ↔ Python bridge
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
└── hyper_core/             # Core Rust tensor ops
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        └── tensor.rs
```

### 📁 crates/fluidelite/ — Alternative Fluid Stack

```
fluidelite/
├── core/                   # Core fluid physics
├── kernels/                # GPU kernels
├── llm/                    # LLM integration
├── optim/                  # Optimization
├── benchmarks/             # Benchmarks
├── scripts/                # Utility scripts
├── tests/                  # Tests
└── utils/                  # Utilities
```

### 📁 integrations/ — Game Engine Integration

```
integrations/
├── unity/                  # Unity C# bindings
│   ├── HyperTensorPlugin.cs
│   └── ...
└── unreal/                 # Unreal C++ bindings
    ├── HyperTensorPlugin.cpp
    └── ...
```

### 📁 sdk/ — Distribution SDKs

```
sdk/
├── conda/                  # Conda package
├── docker/                 # Docker images
├── enterprise/             # Enterprise license
├── qtt-sdk/                # QTT-specific SDK
└── server/                 # Server deployment
```

### 📁 docs/ — Documentation

```
docs/
├── api/                    # API reference
├── architecture/           # Architecture docs
├── attestations/           # Attestation records
├── audits/                 # Security audits
├── legacy/                 # Legacy docs
├── phases/                 # Development phases
├── roadmaps/               # Future plans
├── specifications/         # Specs
├── tutorials/              # User guides
└── workflows/              # Workflow docs
```

### 📁 proofs/ — Mathematical Proofs

```
proofs/
├── navier_stokes/          # NS millennium
├── yang_mills/             # YM mass gap
└── regularity/             # Regularity theory
```

### 📁 demos/ — Demo Scripts

```
demos/
├── evidence/               # Evidence collection
├── demo_*.py               # Various demos
└── showcase_*.py           # Showcase scripts
```

---

## 📜 Key Documentation Files

| File | Purpose |
|------|---------|
| `AGENT_CONTEXT.md` | AI agent context |
| `CONSTITUTION.md` | Project constitution |
| `CONTRIBUTING.md` | Contribution guide |
| `HYPERTENSOR_VV_FRAMEWORK.md` | V&V framework |
| `HyperTensor_CFD_Book.md` | CFD textbook (1.2MB!) |
| `EXTRACTION_MAP.md` | Data extraction |
| `CIVILIZATION_STACK_PHASE_*.md` | Civilization roadmap |
| `TOOLBOX.md` | Functional module catalog |

---

## 🔗 Quick Navigation

### CFD Work
```
ontic/cfd/qtt.py           → QTT fundamentals
ontic/cfd/ns_qtt_solver.py → Navier-Stokes solver
ontic/cfd/qtt_spectral.py  → FFT/WHT operations
ontic/cfd/chi_diagnostic.py → Singularity detection
```

### Black Swan Hunting
```
navier_stokes_black_swan.py        → Main hunter
ontic/cfd/singularity_hunter.py → Detection engine
ontic/cfd/qtt_regularity.py    → BKM criterion
```

### Superconductor Research
```
odin_superconductor_solver.py      → LaLuH6 solver
laluh6_odin_gauntlet.py            → Validation suite
li3incl48br12_superionic_gauntlet.py → Superionic
```

### Fusion Simulation
```
starheart_fusion_solver.py         → Fusion solver
ontic/fusion/*                 → Fusion library
starheart_gauntlet.py              → Validation
```

### Yang-Mills Proofs
```
elite_yang_mills_proof_v2.py       → Latest proof
yang_mills_unified_proof/          → Full proof suite
proofs/yang_mills/lean/                   → Lean formalization
```

---

## 🚀 Getting Started

```bash
# Activate environment
source .venv/bin/activate

# Run a gauntlet
python starheart_gauntlet.py

# Hunt Black Swans
python navier_stokes_black_swan.py

# Explore CFD
python -c "from ontic.cfd import *; print('CFD loaded!')"
```

---

*See [TOOLBOX.md](TOOLBOX.md) for functional descriptions of all 329 modules.*
