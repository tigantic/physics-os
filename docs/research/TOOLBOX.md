# 🧰 Ontic ToolBox Manifest
> **Version**: Phase 27.0 | **Date**: February 6, 2026

The complete catalog of The Physics OS - a physics-first tensor network ecosystem.

> 📖 For detailed specifications, architecture diagrams, and changelog, see [PLATFORM_SPECIFICATION.md](PLATFORM_SPECIFICATION.md)  
> 📊 For machine-readable data, see [component-catalog.json](component-catalog.json)

---

## 📊 Repository Statistics

### Overview
| Category | Count |
|----------|-------|
| **Platforms** | 3 |
| **Modules** | 97 |
| **Applications** | 100 |
| **Tools** | 20 |
| **Total Files** | 24,033 |
| **Total LOC** | **936,723** |
| **Python LOC** | 619,241 |
| **Rust LOC** | 316,929 |
| **Lean 4 LOC** | 553 |

---

### 🐍 Python Distribution (881 files | 399,556 LOC)

| Directory | Files | LOC | % of Python |
|-----------|------:|----:|------------:|
| `ontic/` | 416 | 213,663 | 53.5% |
| `root/*.py` | 57 | 41,830 | 10.5% |
| `tests/` | 60 | 28,232 | 7.1% |
| `crates/fluidelite/` | 82 | 25,604 | 6.4% |
| `demos/` | 45 | 21,910 | 5.5% |
| `yangmills/` | 45 | 18,855 | 4.7% |
| `proofs/` | 34 | 13,424 | 3.4% |
| `tools/scripts/` | 61 | 13,329 | 3.3% |
| `Physics/` | 10 | 7,755 | 1.9% |
| `apps/sdk_legacy/` | 19 | 6,725 | 1.7% |
| `experiments/benchmarks/benchmarks/` | 15 | 3,719 | 0.9% |
| `proofs/proof_engine/` | 7 | 2,759 | 0.7% |
| `tci_llm/` | 10 | 2,261 | 0.6% |
| `ai_scientist/` | 6 | 2,080 | 0.5% |

---

### 🦀 Rust Distribution (141 files | 56,668 LOC)

| Crate | Files | LOC | Purpose |
|-------|------:|----:|---------|
| `apps/glass_cockpit` | 68 | 30,608 | Flight instrumentation |
| `fluidelite-zk` | 51 | 20,703 | ZK prover engine |
| `crates/ontic_bridge` | 8 | 2,135 | Python/Rust FFI |
| `tci_core_rust` | 6 | 1,871 | Tensor Core Interface |
| `apps/global_eye` | 5 | 1,167 | Global monitoring |
| `crates/ontic_core` | 3 | 184 | Core operations |

---

### 📐 Lean Distribution (7 files | 553 LOC)

| File | LOC | Purpose |
|------|----:|---------|
| `YangMills/MassGap.lean` | 178 | Mass gap theorem |
| `YangMillsUnified.lean` | 113 | Unified proof |
| `YangMills/YangMillsMultiEngine.lean` | 94 | Multi-engine |
| `YangMills/YangMillsVerified.lean` | 88 | Verified proofs |
| `YangMills/NavierStokesRegularity.lean` | 78 | NS regularity |
| `YangMills/Basic.lean` | 1 | Definitions |
| `YangMills.lean` | 1 | Entry point |

---

### 🔬 ontic/ Breakdown (416 files | 213,663 LOC)

#### Top 15 Submodules by LOC

| Submodule | Files | LOC | % of tensornet |
|-----------|------:|----:|---------------:|
| `cfd/` | 73 | 45,681 | 21.4% |
| `exploit/` | 38 | 25,975 | 12.2% |
| `oracle/` | 32 | 9,936 | 4.7% |
| `zk/` | 9 | 9,827 | 4.6% |
| `hyperenv/` | 10 | 5,014 | 2.3% |
| `fusion/` | 9 | 4,831 | 2.3% |
| `validation/` | 6 | 4,406 | 2.1% |
| `docs/` | 5 | 4,398 | 2.1% |
| `simulation/` | 6 | 4,360 | 2.0% |
| `ml_surrogates/` | 8 | 3,919 | 1.8% |
| `digital_twin/` | 6 | 3,866 | 1.8% |
| `quantum/` | 7 | 3,831 | 1.8% |
| `intent/` | 7 | 3,784 | 1.8% |
| `guidance/` | 6 | 3,556 | 1.7% |
| `hypersim/` | 7 | 3,462 | 1.6% |

#### All 55 Submodules

| Submodule | Files | LOC | | Submodule | Files | LOC |
|-----------|------:|----:|-|-----------|------:|----:|
| `adaptive/` | 4 | 1,549 | | `mpo/` | 4 | 966 |
| `agri/` | 2 | 397 | | `mps/` | 2 | 432 |
| `algorithms/` | 6 | 2,316 | | `neural/` | 5 | 2,928 |
| `autonomy/` | 5 | 1,871 | | `numerics/` | 2 | 492 |
| `experiments/benchmarks/benchmarks/` | 7 | 2,534 | | `oracle/` | 32 | 9,936 |
| `certification/` | 3 | 1,212 | | `physics/` | 4 | 1,587 |
| `cfd/` | 73 | 45,681 | | `provenance/` | 7 | 3,056 |
| `coordination/` | 5 | 2,167 | | `quantum/` | 7 | 3,831 |
| `core/` | 10 | 3,127 | | `racing/` | 2 | 349 |
| `cuda/` | 6 | 2,891 | | `realtime/` | 5 | 2,746 |
| `cyber/` | 2 | 456 | | `simulation/` | 6 | 4,360 |
| `data/` | 3 | 891 | | `site/` | 5 | 2,645 |
| `defense/` | 4 | 1,634 | | `sovereign/` | 10 | 3,127 |
| `deploy/` | 4 | 1,423 | | `substrate/` | 6 | 2,549 |
| `digital_twin/` | 6 | 3,866 | | `urban/` | 3 | 1,068 |
| `distributed/` | 6 | 2,891 | | `validation/` | 6 | 4,406 |
| `distributed_tn/` | 5 | 2,134 | | `visualization/` | 2 | 705 |
| `docs/` | 5 | 4,398 | | `zk/` | 9 | 9,827 |
| `emergency/` | 2 | 512 | | `hw/` | 3 | 1,689 |
| `energy/` | 3 | 1,245 | | | | |
| `exploit/` | 38 | 25,975 | | | | |
| `fieldops/` | 2 | 634 | | | | |
| `fieldos/` | 7 | 3,245 | | | | |
| `financial/` | 4 | 1,876 | | | | |
| `flight_validation/` | 5 | 2,341 | | | | |
| `fuel/` | 3 | 1,123 | | | | |
| `fusion/` | 9 | 4,831 | | | | |
| `gateway/` | 6 | 2,567 | | | | |
| `gpu/` | 8 | 3,245 | | | | |
| `guidance/` | 6 | 3,556 | | | | |
| `hyperenv/` | 10 | 5,014 | | | | |
| `hypersim/` | 7 | 3,462 | | | | |
| `hypervisual/` | 6 | 2,891 | | | | |
| `integration/` | 5 | 2,134 | | | | |
| `intent/` | 7 | 3,784 | | | | |
| `medical/` | 2 | 431 | | | | |
| `ml_surrogates/` | 8 | 3,919 | | | | |

---

### 📈 Applications Breakdown

| Category | Count | Description |
|----------|------:|-------------|
| Gauntlets | 18 | Validation suites |
| Demos | 45 | Visualizations & examples |
| Rust Binaries | 24 | FluidElite-ZK executables |
| Proof Pipelines | 5 | Millennium problem proofs |
| Solvers | 4 | Specialized physics solvers |
| Engines | 2 | Continuous simulation |
| Rust Apps | 2 | Native applications |
| **Total** | **100** | |

---

### 📚 Documentation & Artifacts

| Type | Count |
|------|------:|
| Markdown docs (root) | 60 |
| Attestation JSONs | 41 |
| Bench Protocols | 9 |
| Notebooks | 7 |

---

# 🏛️ PLATFORMS

> **Definition**: Integrated systems with multiple applications, APIs, and infrastructure. Provides a foundation for others to build on.

---

## 1. The Ontic Engine
**The Physics-First Tensor Network Engine**

A complete simulation platform for computational physics using QTT (Quantized Tensor Train) compression.

| Component | Location | Description |
|-----------|----------|-------------|
| Core Engine | `ontic/` | 55 Python modules, 416 files |
| GPU Backend | `ontic/gpu/`, `ontic/cuda/` | CUDA acceleration |
| Visualization | `ontic/visualization/`, `ontic/quantum/` | Real-time rendering |
| RL Environments | `ontic/hyperenv/`, `ontic/hypersim/` | Gym-compatible physics |

**Capabilities:**
- CFD at 10¹² grid points without dense materialization
- 5D Vlasov-Poisson plasma kinetics
- Hypersonic flight simulation
- Fusion reactor modeling (tokamak, MARRS)
- Yang-Mills gauge theory

---

## 2. FluidElite
**Production Tensor Network Engine**

High-performance tensor operations for LLMs and ZK proofs.

| Component | Location | Description |
|-----------|----------|-------------|
| Python Core | `crates/fluidelite/` | 59 files - MPS/MPO operations |
| Rust Prover | `crates/fluidelite_zk/` | 48 Rust files + 24 binaries |
| Triton Kernels | `crates/fluidelite/core/triton_kernels.py` | GPU-native ops |
| LLM Integration | `crates/fluidelite/llm/` | Tensor-compressed language models |

**Binaries:**
| Binary | Description |
|--------|-------------|
| `cli` | Command-line interface |
| `server` | Prover server |
| `prover_node` | Distributed prover node |
| `gevulot_prover` | Gevulot network integration |
| `fluid_ingest` | Data ingestion |
| `gpu_*` | 12 GPU benchmark variants |

---

## 3. Sovereign Compute
**Decentralized Physics Computation Network**

| Component | Location | Description |
|-----------|----------|-------------|
| Protocol | `ontic/sovereign/protocol.py` | Compute protocol |
| Bridge | `ontic/sovereign/bridge_writer.py` | Cross-chain bridge |
| Streaming | `ontic/sovereign/qtt_bridge_streamer.py` | QTT over network |
| Weather | `ontic/sovereign/weather_stream.py` | Real-time data ingestion |
| Gevulot | `crates/gevulot/` | Prover network integration |

---

# 📦 MODULES

> **Definition**: Reusable libraries/packages that provide functions/classes to be imported. Has `__init__.py` (Python) or `Cargo.toml` (Rust). **Cannot run standalone.**

---

## Python Modules (87)

### ontic/ - Core Physics Engine
**55 submodules | 416 files**

#### 🌊 CFD (Computational Fluid Dynamics)
**`ontic/cfd/`** | 73 files

```python
from ontic.cfd import Euler1D, Euler2D, Euler3D
from ontic.cfd import QTTNavierStokesIMEX, IMEXScheme
from ontic.cfd import qtt_roll_exact, qtt_walsh_hadamard
```

| Module | Purpose |
|--------|---------|
| `euler_1d.py`, `euler_2d.py`, `euler_3d.py` | Compressible Euler solvers |
| `navier_stokes.py`, `ns_2d.py`, `ns_3d.py` | Incompressible NS solvers |
| `fast_euler_2d.py`, `fast_euler_3d.py` | Native optimized Euler |
| `fast_vlasov_5d.py` | 5D Vlasov-Poisson plasma kinetics |
| `qtt_spectral.py` | Walsh-Hadamard FFT without dense |
| `qtt_shift_stable.py` | Rank-preserving shift |
| `qtt_hadamard.py` | Element-wise QTT multiplication |
| `qtt_regularity.py` | Vorticity & BKM criterion |
| `qtt_imex.py` | Implicit-Explicit time stepping |
| `qtt_multiscale.py` | Variable-rank multi-scale |
| `chi_diagnostic.py` | χ-state singularity tracking |
| `singularity_hunter.py` | Black swan detection |
| `koopman_tt.py` | Koopman operator in TT format |
| `turbulence.py` | Turbulence modeling |
| `chemistry.py` | Reactive flow chemistry |
| `comfort_metrics.py` | HVAC comfort indices |

#### 🎯 Exploit Hunting
**`ontic/exploit/`** | 38 files

```python
from ontic.exploit import KoopmanExploitHunter, HypergridController
from ontic.exploit import PrecisionAnalyzer, BountyReporter
```

| Module | Purpose |
|--------|---------|
| `koopman_hunter.py` | Koopman eigenvalue exploit discovery |
| `hypergrid.py` | Parallel distributed hunting |
| `precision_analyzer.py` | Floating-point vulnerability detection |
| `bounty_api.py` | Immunefi/Code4rena API |
| `bounty_reporter.py` | Automated report generation |
| `contract_loader.py` | Solidity contract parsing |
| `pendle_hunter.py` | Pendle yield derivatives |
| `renzo_hunt.py`, `kelp_hunt.py`, `etherfi_hunt.py` | LRT protocol hunts |

#### 🔮 Oracle (Assumption Extraction)
**`ontic/oracle/`** | 32 files

```python
from ontic.oracle import ImplicitExtractor, AssumptionChallenger
```

| Submodule | Purpose |
|-----------|---------|
| `assumptions/` | Implicit assumption extraction |
| `challenger/` | Assumption challenge generation |
| `execution/` | Symbolic execution engine |
| `parsing/` | Solidity/Circom parsers |
| `semantic/` | Semantic analysis |
| `verification/` | Formal verification |

#### ⚙️ Core Operations
**`ontic/core/`** | 10 files

```python
from ontic.core import decompositions, mpo, mps
from ontic.core.determinism import set_global_seed
```

| Module | Purpose |
|--------|---------|
| `decompositions.py` | TT/QTT decomposition algorithms |
| `mpo.py` | Matrix Product Operators |
| `mps.py` | Matrix Product States |
| `gpu.py` | GPU detection and management |
| `determinism.py` | Reproducibility controls |
| `dense_guard.py` | Prevent accidental dense ops |

#### 🔐 ZK (Zero-Knowledge)
**`ontic/zk/`** | 9 files

```python
from ontic.zk import Halo2Analyzer, PILParser, ZKASMEliteAnalyzer
```

| Module | Purpose |
|--------|---------|
| `halo2_analyzer.py` | Halo2 constraint analysis |
| `pil_parser.py` | PIL constraint parsing |
| `zkasm_elite_analyzer.py` | zkASM assembly analysis |
| `fezk_elite.py` | Elite ZK circuit analysis |

#### ☢️ Fusion
**`ontic/fusion/`** | 9 files

```python
from ontic.fusion import MARRSSimulator, TokamakSolver
```

| Module | Purpose |
|--------|---------|
| `tokamak.py` | Tokamak magnetic confinement |
| `marrs_simulator.py` | MARRS solid-state fusion (DARPA) |
| `electron_screening.py` | Electron screening potential |
| `superionic_dynamics.py` | Deuterium superionic dynamics |
| `phonon_trigger.py` | Fokker-Planck phonon trigger |

#### 🔬 Quantum
**`ontic/quantum/`** | 7 files

```python
from ontic.quantum import QTTTorchRenderer, HybridQTTRenderer
```

| Module | Purpose |
|--------|---------|
| `qtt_torch_renderer.py` | PyTorch QTT rendering |
| `hybrid_qtt_renderer.py` | Hybrid CPU/GPU rendering |
| `qtt_glsl_bridge.py` | GLSL shader bridge |
| `error_mitigation.py` | Quantum error mitigation |

#### 🎮 GPU Acceleration
**`ontic/gpu/`** | 8 files

```python
from ontic.gpu import FluidDynamics, KernelAutotuner
```

| Module | Purpose |
|--------|---------|
| `fluid_dynamics.py` | GPU fluid simulation |
| `stable_fluid.py` | Stam stable fluids |
| `advection.py` | Semi-Lagrangian advection |
| `kernel_autotune_cache.py` | Kernel autotuning |

#### 🤖 ML Surrogates
**`ontic/ml_surrogates/`** | 8 files

```python
from ontic.ml_surrogates import FourierOperator, DeepONet, PINN
```

| Module | Purpose |
|--------|---------|
| `fourier_operator.py` | Fourier Neural Operator |
| `deep_onet.py` | Deep Operator Networks |
| `physics_informed.py` | Physics-Informed NNs |
| `uncertainty.py` | Bayesian uncertainty |

#### 🌌 HyperEnv (RL Environments)
**`ontic/hyperenv/`** | 10 files

```python
from ontic.hyperenv import HypersonicEnv, Trainer
```

| Module | Purpose |
|--------|---------|
| `hypersonic_env.py` | Hypersonic flight environment |
| `agent.py` | RL agent implementations |
| `trainer.py` | Training loops |
| `train_pilot.py` | Autonomous pilot training |

#### 🎯 HyperSim (Gym Environments)
**`ontic/hypersim/`** | 7 files

```python
from ontic.hypersim import FluidEnv, QTTSpaces
```

| Module | Purpose |
|--------|---------|
| `env.py` | Fluid environment |
| `spaces.py` | QTT-native observation/action |
| `rewards.py` | Physics-based rewards |
| `curriculum.py` | Progressive difficulty |

#### 🎨 Visualization
**`ontic/visualization/`** | 2 files

```python
from ontic.visualization import TensorSlicer
```

| Module | Purpose |
|--------|---------|
| `tensor_slicer.py` | Decompression-free 2D slices of 10¹²+ tensors |

#### 🔢 Numerics
**`ontic/numerics/`** | 2 files

| Module | Purpose |
|--------|---------|
| `interval.py` | Interval arithmetic for computer-assisted proofs |

---

### 🌍 Domain Verticals (ontic/)

| Domain | Files | Purpose |
|--------|-------|---------|
| `adaptive/` | 4 | Adaptive mesh refinement |
| `agri/` | 2 | Agricultural simulation |
| `algorithms/` | 6 | Core algorithms |
| `autonomy/` | 5 | Autonomous systems |
| `experiments/benchmarks/benchmarks/` | 7 | Performance benchmarks |
| `certification/` | 3 | Safety certification |
| `coordination/` | 5 | Multi-agent coordination |
| `cyber/` | 2 | Cybersecurity |
| `data/` | 3 | Data utilities |
| `defense/` | 4 | Defense applications |
| `deploy/` | 4 | Deployment tooling |
| `digital_twin/` | 6 | Digital twin simulation |
| `distributed/` | 6 | Distributed computing |
| `distributed_tn/` | 5 | Distributed tensor networks |
| `emergency/` | 2 | Emergency response |
| `energy/` | 3 | Energy systems |
| `fieldops/` | 2 | Field operations |
| `fieldos/` | 7 | Field OS platform |
| `financial/` | 4 | Financial modeling |
| `flight_validation/` | 5 | Flight test validation |
| `fuel/` | 3 | Fuel systems |
| `gateway/` | 6 | API gateway |
| `guidance/` | 6 | Guidance systems |
| `integration/` | 5 | System integration |
| `intent/` | 7 | Intent recognition |
| `medical/` | 2 | Medical applications |
| `mpo/` | 4 | MPO operations |
| `mps/` | 2 | MPS operations |
| `neural/` | 5 | Neural network integration |
| `physics/` | 4 | Hypersonic, trajectory |
| `provenance/` | 7 | Data provenance |
| `racing/` | 2 | Racing simulation |
| `realtime/` | 5 | Real-time systems |
| `simulation/` | 6 | General simulation |
| `site/` | 5 | Site management |
| `substrate/` | 6 | Substrate integration |
| `urban/` | 3 | Urban planning |
| `validation/` | 6 | Validation framework |

---

### crates/fluidelite/ - Production Tensor Engine
**12 submodules | 59 files**

| Submodule | Files | Purpose |
|-----------|-------|---------|
| `core/` | 11 | MPS/MPO operations, decompositions |
| `llm/` | 4 | LLM integration |
| `zk/` | 5 | ZK proof integration |
| `fe_tci/` | 3 | Tensor Core Interface |
| `optim/` | 2 | Riemannian optimization |
| `utils/` | 6 | Utilities |

---

### yangmills/ - Gauge Theory
**2 submodules | 28 files**

```python
from yangmills import Hamiltonian, Lattice, SU2, GroundState
```

| Module | Purpose |
|--------|---------|
| `hamiltonian.py` | Yang-Mills Hamiltonian |
| `lattice.py` | Lattice gauge theory |
| `su2.py` | SU(2) gauge group |
| `ground_state.py` | Ground state computation |
| `transfer_matrix_proof.py` | Transfer matrix proofs |
| `yangmills_4d_qtt.py` | 4D Yang-Mills in QTT |

---

### Other Python Modules

| Module | Files | Purpose |
|--------|-------|---------|
| `proofs/` | 34 | Mathematical proof scripts |
| `apps/sdk_legacy/` | 19 | Enterprise SDK, QTT-SDK |
| `tci_llm/` | 10 | Tensor Core Interface for LLMs |
| `ai_scientist/` | 6 | Automated scientific discovery |
| `proofs/proof_engine/` | 7 | Proof orchestration |
| `Physics/` | 10 | Standalone physics benchmarks |

---

## Rust Crates (6)

| Crate | Location | Purpose |
|-------|----------|---------|
| `fluidelite-zk` | `crates/fluidelite_zk/` | High-perf ZK prover (48 files) |
| `ontic_core` | `crates/ontic_core/` | Core tensor operations |
| `ontic_bridge` | `crates/ontic_bridge/` | Python/Rust FFI |
| `glass_cockpit` | `apps/glass_cockpit/` | Flight display (68 files) |
| `global_eye` | `apps/global_eye/` | Global monitoring |
| `tci_core_rust` | `crates/tci_core_rust/` | Tensor Core Interface |

---

## Lean Packages (2)

| Package | Files | Purpose |
|---------|-------|---------|
| `proofs/yang_mills/lean/` | 7 | Yang-Mills formalization, mass gap theorem |
| `yang_mills_unified_proof/` | 1 | Lean 4 unified proof |

---

# 🚀 APPLICATIONS

> **Definition**: Standalone executables that perform a specific task. Has `main()` or `if __name__ == "__main__"`. **You run these directly.**

---

## 🏆 Gauntlets (Validation Suites) - 18

Comprehensive test suites that validate entire subsystems.

| Gauntlet | Domain | What It Validates |
|----------|--------|-------------------|
| `test_aging_gauntlet.py` | Biological aging | Cell state QTT, rank dynamics, Yamanaka reversal |
| `chronos_gauntlet.py` | Time evolution | TDVP accuracy, conservation |
| `cornucopia_gauntlet.py` | Optimization | Resource allocation |
| `femto_fabricator_gauntlet.py` | Molecular | Atomic placement <0.1Å |
| `hellskin_gauntlet.py` | Thermal | Re-entry heat shield |
| `hermes_gauntlet.py` | Messaging | Routing correctness |
| `laluh6_odin_gauntlet.py` | Superconductor | LaLuH₆ at 300K |
| `li3incl48br12_superionic_gauntlet.py` | Battery | Li₃InCl₄₈Br₁₂ superionic |
| `metric_engine_gauntlet.py` | Benchmarks | Performance metrics |
| `oracle_gauntlet.py` | Prediction | Forecast accuracy |
| `orbital_forge_gauntlet.py` | Orbital | Trajectory mechanics |
| `prometheus_gauntlet.py` | Combustion | Fire simulation |
| `proteome_compiler_gauntlet.py` | Biology | Protein folding |
| `snhff_stochastic_gauntlet.py` | Stochastic | NS with noise |
| `sovereign_genesis_gauntlet.py` | Bootstrap | System initialization |
| `starheart_gauntlet.py` | Fusion | Reactor output |
| `tig011a_dielectric_gauntlet.py` | Materials | Dielectric properties |
| `tomahawk_cfd_gauntlet.py` | Aerodynamics | Missile CFD |

**Run:** `python chronos_gauntlet.py`

---

## ⚡ Solvers - 4

Specialized physics solvers for specific problems.

| Solver | Domain | Problem |
|--------|--------|---------|
| `hellskin_thermal_solver.py` | Thermal | Re-entry protection |
| `odin_superconductor_solver.py` | Quantum | Room-temp superconductor |
| `ssb_superionic_solver.py` | Battery | Solid-state battery |
| `starheart_fusion_solver.py` | Fusion | Reactor dynamics |

**Run:** `python starheart_fusion_solver.py`

---

## 🔧 Engines - 2

Continuous simulation engines.

| Engine | Domain | Purpose |
|--------|--------|---------|
| `real_yang_mills_engine.py` | Gauge theory | Yang-Mills dynamics |
| `wilson_plaquette_engine.py` | Lattice | Plaquette computation |

---

## 📈 Proof Pipelines - 5

End-to-end mathematical proof automation.

| Pipeline | Target | Millennium Problem? |
|----------|--------|---------------------|
| `navier_stokes_millennium_pipeline.py` | NS regularity | ✅ Yes |
| `yang_mills_proof_pipeline.py` | Mass gap | ✅ Yes |
| `yang_mills_unified_proof.py` | Unified YM | ✅ Yes |
| `elite_yang_mills_proof.py` | Elite YM | ✅ Yes |
| `integrated_proof_pipeline_v2.py` | Combined | Both |

**Run:** `python navier_stokes_millennium_pipeline.py`

---

## 🖥️ Rust Applications - 2

Native high-performance applications.

### Glass Cockpit
**`apps/glass_cockpit/`** | 68 Rust files

Real-time flight instrumentation display.

```bash
cargo run -p glass_cockpit
```

### Global Eye
**`apps/global_eye/`** | 5 Rust files

Global monitoring and Earth visualization.

```bash
cargo run -p global_eye
```

---

## 🎬 Demos - 45

Demonstrations and visualizations.

### Black Swan (Singularity Hunting)
| Demo | Purpose |
|------|---------|
| `trap_the_swan.py` | Main singularity hunter |
| `black_swan_945_forensic.py` | Forensic analysis at 945³ |
| `black_swan_1024_confirm.py` | 1024³ confirmation |
| `black_swan_reproduce.py` | Reproduction script |

### HVAC/CFD Visualization
| Demo | Purpose |
|------|---------|
| `conference_room_cfd.py` | Room airflow simulation |
| `conference_room_qtt.py` | QTT-native room |
| `conference_room_native.py` | Native solver comparison |

### Physics Demos
| Demo | Purpose |
|------|---------|
| `cfd_shock.py` | Shock tube visualization |
| `blue_marble.py` | Earth rendering |
| `demo_holy_grail.py` | 5D holy grail |
| `ontic_hub.py` | Main dashboard |

**Run:** `python demos/cfd_shock.py`

---

## 🦀 Rust Binaries - 24

FluidElite-ZK prover binaries.

| Binary | Purpose |
|--------|---------|
| `cli` | Command-line interface |
| `server` | Prover server |
| `prover_node` | Distributed prover |
| `gevulot_prover` | Gevulot network |
| `fluid_ingest` | Data ingestion |
| `fluid_block` | Block processing |
| `gpu_test` | GPU validation |
| `gpu_stress_test` | Stress testing |
| `gpu_benchmark` | Benchmarking |
| `vram_stress_test` | VRAM limits |
| + 14 more GPU variants | Various benchmarks |

**Run:** `cargo run --bin server`

---

# 🔧 TOOLS

> **Definition**: Single-purpose utilities that perform a specific task. Often invoked by other systems or used for one-off analysis.

---

## �️ The_Compressor - 5 ⭐ NEW

> **63,321x QTT Compression Engine** — Breakthrough compression technology

| Tool | Purpose | Performance |
|------|---------|-------------|
| `compress.py` | 4D QTT compression with Morton Z-order | 16.95 GB → 258 KB |
| `decompress.py` | Original frame-based decompressor | ~10k queries/sec |
| `universal.py` | Universal N-D decompressor | Any dimensionality |
| `compress_24h.py` | Satellite data variant | 24-hour time series |
| `__init__.py` | Module initialization | Self-contained |

### Core Technology
- **4D Quantics Tensor Train (QTT)** decomposition
- **Morton Z-order bit-interleaving** for space-time locality
- **mmap streaming** — zero RAM during compression
- **Core-by-core GPU SVD** — VRAM <100 MB
- **Eigendecomposition fallback** for wide matrices
- **float16 core storage**

### Usage
```bash
# Compress
python The_Compressor/compress.py --input data_folder --output compressed.npz

# Inspect
python The_Compressor/universal.py info compressed.npz

# Query point (any dimensionality)
python The_Compressor/universal.py query compressed.npz 16,1024,1024

# Reconstruct
python The_Compressor/universal.py reconstruct compressed.npz -o output.npy

# Benchmark
python The_Compressor/universal.py benchmark compressed.npz
```

### Results
| Metric | Value |
|--------|-------|
| Original Size | 16.95 GB (NOAA GOES-18) |
| Compressed Size | 258 KB |
| Compression Ratio | **63,321x** |
| L2 Cache Fit | ✅ (<2.5 MB) |
| Point Query | ~93 µs |
| Queries/sec | 10,000+ |
| Dependencies | numpy, torch (CUDA optional) |

**Release**: `v1.0.0-the-compressor` on GitHub

---

## �🔐 Hardware Security - 3

| Tool | Location | Purpose |
|------|----------|---------|
| `verilog_elite_analyzer.py` | `ontic/hw/` | Pattern-based Verilog security scanner |
| `yosys_netlist_analyzer_v2.py` | `ontic/hw/` | sv2v+Yosys synthesis pipeline |
| `yosys_netlist_analyzer.py` | `ontic/hw/` | JSON netlist analysis |

```python
from ontic.hw import VerilogEliteAnalyzer, YosysNetlistAnalyzer
analyzer = VerilogEliteAnalyzer()
results = analyzer.analyze_file("rtl/module.sv")
```

---

## 🎯 Bounty Hunting - 5

| Tool | Location | Purpose |
|------|----------|---------|
| `hunt_renzo.py` | root | Renzo protocol hunt |
| `temp_debridge_hunt.py` | root | deBridge hunt |
| `advanced_vulnerability_hunt.py` | root | Multi-protocol hunt |
| `GMX_V2_VULNERABILITY_ANALYSIS.py` | root | GMX V2 analysis |
| `ontic/exploit/cairo_circuit_hunter.py` | module | Cairo ZK circuits |

---

## 📐 Proof Tools - 3

| Tool | Location | Purpose |
|------|----------|---------|
| `proofs/run_all_proofs.py` | `proofs/` | Execute all proofs |
| `proofs/proof_master.py` | `proofs/` | Master orchestrator |
| `proofs/proof_summary.py` | `proofs/` | Summarize results |

---

## 🔬 Attestation Validators - 4

| Tool | Purpose |
|------|---------|
| `tig011a_attestation.py` | Validate TIG011A results |
| Various `*_validation.py` | Domain-specific validation |

---

# 📚 DOCUMENTATION

| Category | Count | Location |
|----------|-------|----------|
| Root `.md` files | 60 | `/` |
| Attestation JSONs | 41 | `/` |
| Bench Protocols | 9 | `/` |

### Key Documents
| Document | Purpose |
|----------|---------|
| `CIVILIZATION_STACK_PHASE_*.md` | Roadmap |
| `Ontic_CFD_Book.md` | CFD documentation |
| `ONTIC_ENGINE_VV_FRAMEWORK.md` | V&V framework |
| `ORACLE_ARCHITECTURE.md` | Oracle design |
| `The_Civilization_Stack.md` | Vision |

---

# 🔌 INTEGRATIONS

| Integration | Location | Purpose |
|-------------|----------|---------|
| Unity | `integrations/unity/` | Game engine |
| Unreal | `integrations/unreal/` | Game engine |
| Gevulot | `crates/gevulot/` | Prover network |

---

# 🏛️ ARCHITECTURE PRINCIPLES

1. **Never Go Dense**: Use QTT cores, never materialize full tensors
2. **Rank Control**: Always truncate after operations that grow rank
3. **GPU First**: Auto-detect CUDA, fall back to CPU
4. **Reproducibility**: Seed control via `ontic/core/determinism.py`
5. **Attestation**: Every gauntlet produces signed JSON proof
6. **Physics First**: Numerical methods grounded in physical laws

---

# 📋 QUICK REFERENCE

## Taxonomy Cheat Sheet

| Type | Definition | Example | How to Use |
|------|------------|---------|------------|
| **Platform** | Integrated system | Ontic Engine | Deploy & configure |
| **Module** | Reusable library | `ontic/cfd/` | `import` it |
| **Application** | Standalone executable | `hellskin_gauntlet.py` | `python` run it |
| **Tool** | Single-purpose utility | `verilog_elite_analyzer.py` | Invoke for task |

## Import Patterns

```python
# Modules - you IMPORT them
from ontic.cfd import Euler3D, QTTNavierStokesIMEX
from ontic.exploit import KoopmanExploitHunter
from ontic.fusion import MARRSSimulator
from ontic.hw import VerilogEliteAnalyzer

# Applications - you RUN them
# python hellskin_gauntlet.py
# python demos/cfd_shock.py
# cargo run -p glass_cockpit

# Tools - you INVOKE them
# python proofs/run_all_proofs.py
# python hunt_renzo.py
```

---

# 🍳 COOKBOOK

## Common Recipes

### 1. Run a CFD Simulation
```python
from ontic.cfd import Euler3D
from ontic.core.determinism import set_global_seed

set_global_seed(42)
solver = Euler3D(grid_bits=10, gamma=1.4)  # 2^10 = 1024 points per dim
result = solver.evolve(dt=0.001, steps=100)
print(f"Final energy: {result.total_energy()}")
```

### 2. Hunt Smart Contract Vulnerabilities
```python
from ontic.exploit import KoopmanExploitHunter

hunter = KoopmanExploitHunter()
results = hunter.analyze_contract("path/to/Contract.sol")
for vuln in results.vulnerabilities:
    print(f"{vuln.severity}: {vuln.description}")
```

### 3. Run a Fusion Reactor Simulation
```python
from ontic.fusion import TokamakSolver

tokamak = TokamakSolver(
    major_radius=6.2,  # meters
    minor_radius=2.0,
    plasma_current=15e6  # Amperes
)
confinement = tokamak.compute_confinement_time()
print(f"Energy confinement: {confinement:.3f} s")
```

### 4. Analyze Hardware Verilog
```python
from ontic.hw import VerilogEliteAnalyzer

analyzer = VerilogEliteAnalyzer()
findings = analyzer.scan_directory("rtl/")
for finding in findings:
    print(f"{finding.file}:{finding.line} - {finding.pattern}")
```

### 5. Use Genesis Primitives
```python
from ontic.genesis.ot import QTTSinkhorn, wasserstein_distance
from ontic.genesis.ga import CliffordAlgebra, QTTMultivector

# Optimal Transport
w2 = wasserstein_distance(source_qtt, target_qtt, p=2)

# Geometric Algebra
cl3 = CliffordAlgebra(3, 0, 0)  # Cl(3,0,0) = 3D Euclidean
mv = QTTMultivector(cl3, coefficients_qtt)
result = mv.geometric_product(mv)
```

### 6. Reverse Biological Aging
```python
from ontic.genesis.aging import (
    young_cell, aged_cell, AgingOperator,
    YamanakaOperator, HorvathClock,
    find_optimal_intervention,
)

# Create cells and age them
young = young_cell(seed=42)
operator = AgingOperator(seed=42)
aged, trajectory = operator.evolve(young, target_age=70.0, dt_years=5.0)
print(f"Rank growth: {young.max_rank} → {aged.max_rank}")

# Measure epigenetic age
clock = HorvathClock()
bio_age = clock.predict_age_from_cell(aged)
print(f"Biological age: {bio_age:.1f}")

# Apply Yamanaka reprogramming (rank-4 intervention)
yamanaka = YamanakaOperator(target_rank=4)
result = yamanaka.apply(aged)
print(f"Reversed: rank {result.rank_before} → {result.rank_after}")
print(f"Years reversed: {result.years_reversed:.1f}")

# Find optimal intervention automatically
best, log = find_optimal_intervention(aged, young, max_intervention_rank=8)
print(f"Best intervention: rank {best.rank_after}, fidelity {best.fidelity:.3f}")
```

---

# 🔧 TROUBLESHOOTING

## Common Issues

### Import Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: tensornet` | Package not installed | `pip install -e .` from repo root |
| `ImportError: cannot import 'Euler3D'` | Wrong import path | Use `from ontic.cfd import Euler3D` |
| `CUDA out of memory` | GPU memory exhausted | Reduce `grid_bits` or use CPU fallback |

### Runtime Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Rank explosion detected` | QTT rank grew too large | Call `qtt.round(eps=1e-6)` after operations |
| `Conservation violation` | Numerical instability | Reduce time step `dt` or increase resolution |
| `Attestation failed` | Cryptographic verification error | Ensure deterministic seed is set |

### Performance Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Slow first run | JIT compilation | Subsequent runs will be faster |
| CPU fallback warnings | No CUDA detected | Install CUDA toolkit or ignore for CPU-only |
| High memory usage | Dense materialization | Check for `.to_dense()` calls, use QTT ops |

### Gauntlet Failures

```bash
# Re-run with verbose output
python hellskin_gauntlet.py --verbose

# Check attestation
cat HELLSKIN_GAUNTLET_ATTESTATION.json | jq '.result'

# Validate environment
python -c "from ontic.core import check_environment; check_environment()"
```

---

# 📎 SEE ALSO

| Document | Description |
|----------|-------------|
| [PLATFORM_SPECIFICATION.md](PLATFORM_SPECIFICATION.md) | Full architecture, changelog, Mermaid diagrams |
| [TENSOR_GENESIS.md](TENSOR_GENESIS.md) | Genesis layers 20-27 detailed specification |
| [QTT_AGING_ATTESTATION.json](QTT_AGING_ATTESTATION.json) | Layer 27 gauntlet attestation (127/127) |
| [Ontic_CFD_Book.md](Ontic_CFD_Book.md) | CFD module deep dive |
| [ORACLE_ARCHITECTURE.md](ORACLE_ARCHITECTURE.md) | Oracle system design |
| [component-catalog.json](component-catalog.json) | Machine-readable component catalog |

---

*Generated by Ontic Phase 27 • February 6, 2026*  
*Run `python tools/scripts/update_loc_counts.py` to refresh statistics*
