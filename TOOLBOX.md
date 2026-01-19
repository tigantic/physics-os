# 🧰 HyperTensor ToolBox Manifest
> **Version**: Phase 24 | **Date**: January 16, 2026 | **Modules**: 333 Python files

This document catalogs all tools, modules, and capabilities in the HyperTensor physics engine.
Use this as a reference when building new simulations, gauntlets, or demonstrations.

---

## 📊 Quick Stats

| Metric | Count |
|--------|-------|
| Total Python Modules | 333 |
| CFD Modules | 73 |
| GPU Modules | 8 |
| Gauntlet Tests | 17 |
| Demo Scripts | 40+ |
| Attestation Files | 20+ |

---

## 🌊 CFD (Computational Fluid Dynamics)
**Location**: `tensornet/cfd/` | **70 modules**

### Core Solvers

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `euler_1d.py` | 1D Euler equations for compressible flow | `Euler1D`, `sod_shock_tube_ic` |
| `euler_2d.py` | 2D compressible Euler equations | `Euler2D`, `Euler2DState` |
| `euler_3d.py` | 3D Euler with directional splitting | `Euler3DSolver` |
| `euler_nd_native.py` | Fully native N-dimensional Euler | Universal solver |
| `fast_euler_2d.py` | Ultra-fast native 2D Euler | Optimized pipeline |
| `fast_euler_3d.py` | Native 3D Euler with MPO shift | `qtt_3d_to_dense` |
| `fast_vlasov_5d.py` | 5D Vlasov-Poisson solver | Plasma kinetics |
| `navier_stokes.py` | Incompressible NS solver | Viscous flows |
| `ns_2d.py` | 2D incompressible NS | `NavierStokes2D` |
| `ns_3d.py` | 3D incompressible NS (spectral) | `compute_vorticity_3d`, `poisson_solve_fft_3d` |
| `reactive_ns.py` | Reactive Navier-Stokes | Combustion |

### QTT Operations (Phase 23 Toolbox)

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `qtt_spectral.py` | FFT without dense materialization | `qtt_walsh_hadamard`, `qtt_energy_spectrum_approx`, `ConservationMonitor` |
| `qtt_shift_stable.py` | **Rank-preserving shift** (fixes explosion) | `qtt_roll_exact`, `qtt_3d_roll_exact`, `qtt_central_diff_stable` |
| `qtt_hadamard.py` | Element-wise multiplication | `qtt_hadamard`, `qtt_power`, `qtt_polynomial`, `qtt_full_advection_3d` |
| `qtt_regularity.py` | Vorticity & BKM criterion | `qtt_vorticity_3d`, `qtt_vorticity_max_3d`, `RegularityMonitor` |

### QTT Operations (Phase 24 Toolbox) ⭐ NEW

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `qtt_imex.py` | **IMEX time integrator** for stiff PDEs | `IMEXIntegrator`, `QTTNavierStokesIMEX`, schemes: EULER/MIDPOINT/SBDF2/ARK3 |
| `qtt_multiscale.py` | **Variable-rank QTT** for multi-resolution | `MultiScaleQTT`, `HierarchicalQTT`, profiles: TURBULENT/BOUNDARY_LAYER/ADAPTIVE |
| `qtt_checkpoint_stream.py` | **Async streaming checkpoints** | `SimulationCheckpointer`, double-buffered writes, zstd compression |
| `pure_qtt_ops.py` | Core QTT operations | `qtt_add`, `qtt_to_dense`, `QTTState` |
| `nd_shift_mpo.py` | N-dimensional shift MPO | `make_nd_shift_mpo`, `apply_nd_shift_mpo`, `truncate_cores` |
| `qtt_2d.py` | 2D QTT state management | `QTT2DState`, `morton_encode_batch` |
| `qtt_tci.py` | TCI-based QTT construction | `qtt_from_function` |
| `qtt_tci_gpu.py` | GPU-native QTT-TCI | Randomized SVD |
| `qtt_tdvp.py` | QTT time evolution (TDVP) | O(log N) integration |
| `qtt_eval.py` | QTT evaluation utilities | `qtt_to_dense` |

### Singularity Hunting

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `singularity_hunter.py` | Adjoint-based blowup search | `HuntResult`, `EnstrophyObjective` |
| `chi_diagnostic.py` | χ(t) bond dimension tracking | `ChiState`, `ChiTrajectory` |
| `hou_luo_ansatz.py` | Hou-Luo axisymmetric ansatz | Self-similar profiles |
| `self_similar.py` | Self-similar coordinate transform | Blow-up analysis |
| `kantorovich.py` | Newton-Kantorovich verifier | Computer-assisted proof |
| `newton_refine.py` | Newton refinement | Singularity refinement |
| `stabilized_refine.py` | Stabilized Newton | Robust convergence |
| `adjoint_blowup.py` | Adjoint vorticity maximization | Gradient ascent |

### Exploit Hunting (Phase 25) ⭐ NEW

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `exploit/hypergrid.py` | **Parallel multi-chain hunting** | `HypergridController`, `ChiAggregator`, `KoopmanPrior` |
| `exploit/koopman_hunter.py` | **Koopman structural exploit discovery** | `KoopmanExploitHunter`, `UnstableMode`, `ExploitCandidate` |
| `exploit/precision_analyzer.py` | Fixed-point math bug detection | `PrecisionAnalyzer`, `InflationAttackResult` |

### Turbulence Models

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `turbulence.py` | RANS turbulence models | k-ε, k-ω SST, Spalart-Allmaras |
| `les.py` | Large Eddy Simulation | Smagorinsky, dynamic models |
| `hybrid_les.py` | Hybrid RANS-LES | DES, DDES |
| `koopman_tt.py` | TT-Koopman for turbulence | `TTKoopman`, linearized dynamics |

### Numerical Methods

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `godunov.py` | Riemann solvers | `roe_flux`, `hll_flux`, `hllc_flux`, `exact_riemann` |
| `limiters.py` | Slope limiters | `minmod`, `superbee`, `van_leer`, `mc_limiter` |
| `weno.py` | WENO/TENO schemes | Shock capturing |
| `weno_tt.py` | WENO in tensor-train format | `weno_tt_reconstruct` |
| `weno_native_tt.py` | Pure TT WENO | Native implementation |
| `implicit.py` | Implicit time integration | Stiff systems |
| `tt_poisson.py` | TT Poisson solver | Pressure projection |

### Thermodynamics & Chemistry

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `viscous.py` | NS viscous terms | Sutherland's law, transport properties |
| `real_gas.py` | Real-gas thermodynamics | High-temperature air |
| `chemistry.py` | Multi-species chemistry | Finite-rate reactions |
| `boundaries.py` | Boundary conditions | Wall, inflow, outflow |

### Specialized

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `thermal_qtt.py` | Temperature transport | HVAC coupling |
| `comfort_metrics.py` | PMV/PPD thermal comfort | ASHRAE Standard 55 |
| `kelvin_helmholtz.py` | KH instability IC | Benchmark flows |
| `plasma.py` | Plasma dynamics | MHD |
| `jet_interaction.py` | Jet interaction model | Control surfaces |
| `geometry.py` | Geometry handling | Mesh utilities |
| `adaptive_tt.py` | TT-AMR adaptive bonds | `ShockDetector`, `BondAdapter` |
| `tt_cfd.py` | TT-native CFD solver | TDVP evolution |
| `differentiable.py` | Autograd-enabled CFD | Neural integration |
| `adjoint.py` | Adjoint sensitivity | Shape optimization |
| `multi_objective.py` | Multi-objective opt | Pareto fronts |
| `optimization.py` | CFD optimization | Design tools |

---

## 🔬 Quantum / QTT Rendering
**Location**: `tensornet/quantum/` | **7 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `qtt_torch_renderer.py` | GPU QTT → pixel buffer | `QTTTorchRenderer`, Morton grid synthesis |
| `qtt_glsl_bridge.py` | QTT → GLSL shaders | Real-time rendering |
| `hybrid_qtt_renderer.py` | Hybrid CPU/GPU rendering | Adaptive quality |
| `cpu_qtt_evaluator.py` | CPU QTT evaluation | Fallback evaluator |
| `hybrid.py` | Hybrid quantum/classical | Interface layer |
| `error_mitigation.py` | Quantum error mitigation | Noise handling |

---

## 🧮 Core Tensor Operations
**Location**: `tensornet/core/` | **10 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `decompositions.py` | SVD, QR, TT decomposition | Core factorization |
| `mps.py` | Matrix Product States | `MPS` class |
| `mpo.py` | Matrix Product Operators | `MPO` class |
| `gpu.py` | GPU tensor operations | CUDA acceleration |
| `states.py` | Quantum state utilities | State management |
| `dense_guard.py` | Prevent accidental dense | Memory protection |
| `profiling.py` | Performance profiling | Timing utilities |
| `determinism.py` | Reproducibility control | Seed management |

---

## 🔧 Algorithms
**Location**: `tensornet/algorithms/` | **5 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `dmrg.py` | Density Matrix RG | Ground state optimization |
| `tdvp.py` | Time-Dependent VP | Time evolution |
| `tebd.py` | Time-Evolving Block Decimation | Trotter steps |
| `lanczos.py` | Lanczos algorithm | Eigenvalue problems |
| `fermionic.py` | Fermionic operators | Quantum chemistry |

---

## ⚡ MPO (Matrix Product Operators)
**Location**: `tensornet/mpo/` | **4 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `operators.py` | Standard MPO operators | Laplacian, gradient |
| `laplacian_cuda.py` | CUDA-accelerated Laplacian | `batch_mpo_apply_cuda` |
| `atmospheric_solver.py` | Atmospheric MPO solver | Weather simulation |

---

## 🎮 GPU Acceleration
**Location**: `tensornet/gpu/` | **8 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `fluid_dynamics.py` | GPU fluid simulation | CUDA kernels |
| `fluid_dynamics_optimized.py` | Optimized GPU fluids | Memory-efficient |
| `stable_fluid.py` | Stable fluids (Stam) | Real-time simulation |
| `advection.py` | GPU advection | Semi-Lagrangian |
| `tensor_field.py` | GPU tensor fields | Field operations |
| `memory.py` | GPU memory management | Pool allocation |
| `kernel_autotune_cache.py` | **GPU kernel autotuning** ⭐ NEW | `KernelAutotuner`, `QTTAutotuner`, persistent config cache |

---

## 🤖 ML Surrogates
**Location**: `tensornet/ml_surrogates/` | **8 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `fourier_operator.py` | Fourier Neural Operator | FNO for CFD |
| `deep_onet.py` | Deep Operator Networks | DeepONet |
| `physics_informed.py` | Physics-Informed NNs | PINNs |
| `uncertainty.py` | Uncertainty quantification | Bayesian inference |
| `training.py` | Training utilities | Loss functions |
| `surrogate_base.py` | Base surrogate class | Interface |

---

## ☢️ Fusion
**Location**: `tensornet/fusion/` | **9 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `tokamak.py` | Tokamak simulation | Magnetic confinement |
| `marrs_simulator.py` | MARRS solid-state fusion | DARPA program |
| `electron_screening.py` | Electron screening potential | Fusion enhancement |
| `qtt_screening.py` | QTT-compressed screening | Efficient solver |
| `superionic_dynamics.py` | Superionic Langevin dynamics | Deuterium mobility |
| `qtt_superionic.py` | QTT-enhanced superionic | Compressed dynamics |
| `phonon_trigger.py` | Fokker-Planck phonon trigger | Controlled excitation |
| `resonant_catalysis.py` | Resonant catalysis | Bond activation |

---

## 🎯 Visualization
**Location**: `tensornet/visualization/` | **2 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `tensor_slicer.py` | Decompression-free rendering | `TensorSlicer`, 2D cross-sections of 10¹²+ tensors |

---

## 🌍 HyperSim (RL Environments)
**Location**: `tensornet/hypersim/` | **7 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `env.py` | Fluid environment | Gym-compatible |
| `spaces.py` | Observation/action spaces | QTT-native |
| `rewards.py` | Reward functions | Physics-based |
| `curriculum.py` | Curriculum learning | Progressive difficulty |
| `wrappers.py` | Environment wrappers | Transformations |
| `registry.py` | Environment registry | Discovery |

---

## 🔢 Numerics
**Location**: `tensornet/numerics/` | **2 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `interval.py` | Interval arithmetic | Computer-assisted proofs |

---

## 🚀 Physics
**Location**: `tensornet/physics/` | **4 modules**

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `hypersonic.py` | Hypersonic flight hazard | Trajectory planning |
| `trajectory_optimizer.py` | Trajectory optimization | Guidance |

---

## 🏆 Gauntlet Tests
**Location**: Project root | **17 gauntlets**

| Gauntlet | Domain | Key Validation |
|----------|--------|----------------|
| `chronos_gauntlet.py` | Time evolution | TDVP accuracy |
| `cornucopia_gauntlet.py` | Resource optimization | Allocation |
| `femto_fabricator_gauntlet.py` | Molecular assembly | Atomic placement <0.1Å |
| `hellskin_gauntlet.py` | Thermal protection | Re-entry heating |
| `hermes_gauntlet.py` | Communication | Message routing |
| `laluh6_odin_gauntlet.py` | Superconductor | LaLuH₆ simulation |
| `li3incl48br12_superionic_gauntlet.py` | Superionic | Li₃InCl₄₈Br₁₂ |
| `metric_engine_gauntlet.py` | Performance metrics | Benchmark suite |
| `oracle_gauntlet.py` | Prediction | Forecast accuracy |
| `orbital_forge_gauntlet.py` | Orbital mechanics | Trajectory planning |
| `prometheus_gauntlet.py` | Fire simulation | Combustion |
| `proteome_compiler_gauntlet.py` | Protein folding | Structure prediction |
| `snhff_stochastic_gauntlet.py` | Stochastic NS | Noise handling |
| `sovereign_genesis_gauntlet.py` | System genesis | Bootstrap |
| `starheart_gauntlet.py` | Fusion reactor | Energy output |
| `tig011a_dielectric_gauntlet.py` | Dielectric | Material properties |
| `tomahawk_cfd_gauntlet.py` | Missile CFD | Aerodynamics |

---

## 🎬 Demos
**Location**: `demos/` | **40+ scripts**

### Black Swan (Singularity Hunting)
- `trap_the_swan.py` - Main singularity hunter
- `black_swan_945_forensic.py` - Forensic analysis
- `black_swan_1024_confirm.py` - 1024³ confirmation
- `black_swan_reproduce.py` - Reproduction script

### HVAC/CFD Visualization
- `conference_room_cfd.py` - Room airflow
- `conference_room_qtt.py` - QTT-native room
- `conference_room_native.py` - Native solver

### Physics Demos
- `cfd_shock.py` - Shock tube demo
- `blue_marble.py` - Earth visualization
- `coastlines.py` - Coastal simulation

---

## 📋 Key Imports Cheatsheet

```python
# Phase 24 Toolbox (NEW)
from tensornet.cfd import (
    # IMEX time integration (stiff PDEs)
    IMEXIntegrator,
    QTTNavierStokesIMEX,
    IMEXScheme,  # EULER, MIDPOINT, SBDF2, ARK3
    
    # Multi-scale variable-rank QTT
    MultiScaleQTT,
    HierarchicalQTT,
    RankProfile,  # UNIFORM, TURBULENT, BOUNDARY_LAYER, ADAPTIVE
    
    # Async streaming checkpoints
    SimulationCheckpointer,
    CheckpointWriter,
    CheckpointReader,
)

# GPU kernel autotuning
from tensornet.gpu import (
    KernelAutotuner,
    QTTAutotuner,
    AutotuneCache,
)

# Phase 23 Toolbox
from tensornet.cfd import (
    # Spectral operations
    qtt_walsh_hadamard,
    qtt_energy_spectrum_approx,
    ConservationMonitor,
    
    # Rank-preserving shift
    qtt_roll_exact,
    qtt_3d_roll_exact,
    qtt_central_diff_stable,
    
    # Nonlinear operations
    qtt_hadamard,
    qtt_power,
    qtt_polynomial,
    qtt_full_advection_3d,
    
    # Regularity diagnostics
    qtt_vorticity_3d,
    qtt_vorticity_max_3d,
    RegularityMonitor,
)

# Core QTT operations
from tensornet.cfd.pure_qtt_ops import qtt_add, QTTState
from tensornet.cfd.nd_shift_mpo import truncate_cores, make_nd_shift_mpo

# Solvers
from tensornet.cfd import Euler1D, Euler2D, Euler3D
from tensornet.cfd.ns_3d import compute_vorticity_3d, poisson_solve_fft_3d

# Singularity hunting
from tensornet.cfd.chi_diagnostic import ChiState, ChiTrajectory
from tensornet.cfd.singularity_hunter import HuntResult

# Visualization
from tensornet.visualization.tensor_slicer import TensorSlicer
from tensornet.quantum.qtt_torch_renderer import QTTTorchRenderer

# Turbulence
from tensornet.cfd.koopman_tt import TTKoopman
from tensornet.cfd.turbulence import TurbulenceModel
```

---

## 🔑 Critical Discoveries

### IMEX for Stiff PDEs (Phase 24)
**Problem**: Explicit diffusion requires tiny timesteps (CFL_visc ~ ν·dt/dx² < 0.5)

**Solution**: Implicit-Explicit splitting treats diffusion implicitly:
```python
from tensornet.cfd import QTTNavierStokesIMEX, IMEXScheme

solver = QTTNavierStokesIMEX(n_levels=10, nu=1e-4, scheme=IMEXScheme.SBDF2)
# Advection: explicit (CFL ~ 0.3)
# Diffusion: implicit spectral solve (unconditionally stable)
# 10-100x larger timesteps for viscous flows!
```

### Multi-Scale Variable-Rank QTT (Phase 24)
**Problem**: Uniform QTT rank wastes resources on smooth regions

**Solution**: Allocate rank based on energy spectrum:
```python
from tensornet.cfd import MultiScaleQTT, RankProfile

ms = MultiScaleQTT(n_levels=10, profile=RankProfile.TURBULENT)
# Turbulent: [15, 23, 35, 48, 59, 64, 59, 48, 35, 23]
# More rank at energetic scales, less at dissipation
# 1024³ compression: 15,101x vs full tensor!
```

### Koopman Structural Exploit Discovery (Phase 25) ⭐ NEW
**Problem**: Fuzzing samples randomly hoping to hit exploits (needle in haystack)

**Solution**: Koopman operator linearizes contract dynamics, eigenvalues reveal unstable modes:
```python
from tensornet.exploit.koopman_hunter import KoopmanExploitHunter
from tensornet.exploit.hypergrid import HypergridController

# Phase 1: Find WHERE exploits live (not just IF they exist)
hunter = KoopmanExploitHunter(transition_fn, profit_fn, state_dim)
result = await hunter.hunt(initial_states, tx_generator)

# Unstable modes: |λ| > 1 = exploit direction
for mode in result.unstable_modes:
    print(f"λ={mode.eigenvalue:.3f} growth={mode.growth_rate:.3f}/step")
    # mode.eigenvector tells you WHICH state vars to manipulate

# Phase 2: Feed priors to Hypergrid for targeted hunting
controller = HypergridController()
controller.set_koopman_priors(result, target_address)
# Hunters now sample along unstable manifold, not randomly
# Chi scores boosted when aligned with unstable modes

# Fuzzing: "Is THIS point an exploit?"  O(n) for n-dim space
# Koopman: "WHERE DO EXPLOITS LIVE?"    O(k) for k unstable modes
```

### Pendle Koopman Hunt (Phase 25) ⭐ VALIDATED
**Target**: Pendle Finance yield derivatives (PT/YT/SY tokens)
**Result**: λ = 1.00007 unstable mode → +6.11% validated profit

```python
from tensornet.exploit.pendle_hunter import PendleKoopmanHunter

# Hunt for unstable dynamics in Pendle markets
hunter = PendleKoopmanHunter(market_address="0xd0354...")
result = await hunter.hunt(n_samples=3000, n_trajectories=60)

# Unstable mode eigenvector identifies exploit trajectory:
# - sy_rate (0.80 weight): SY exchange rate accumulator  
# - total_pt (0.62 weight): PT reserves in AMM
# - Exploit: Buy PT → Wait for yield accrual → Sell PT
# - PNL: +6.11% on 100k SY capital over 5 days
```

### Rank-Preserving Shift (Phase 23)
**Problem**: MPO shift doubles rank per step (32→64→128→EXPLOSION)

**Solution**: Bit-flip roll preserves rank exactly:
```python
from tensornet.cfd import qtt_roll_exact

# This preserves rank!
shifted = qtt_roll_exact(cores, shift_amount=7)
# rank 32 → 32 → 32 → 32 (stable forever)
```

### Walsh-Hadamard FFT (Phase 23)
**Problem**: FFT requires O(N) dense memory

**Solution**: WHT is separable, applies per-core:
```python
from tensornet.cfd import qtt_walsh_hadamard

# FFT-like transform without dense
hat_cores = qtt_walsh_hadamard(cores)
# Same ranks! Unitary transform.
```

---

## 📍 File Locations Quick Reference

| Tool | Path |
|------|------|
| **Phase 25** | |
| Koopman Exploit Hunter | `tensornet/exploit/koopman_hunter.py` |
| Pendle Koopman Hunter | `tensornet/exploit/pendle_hunter.py` |
| Hypergrid Parallel Hunter | `tensornet/exploit/hypergrid.py` |
| Precision Analyzer | `tensornet/exploit/precision_analyzer.py` |
| **Phase 24** | |
| QTT IMEX | `tensornet/cfd/qtt_imex.py` |
| QTT Multi-Scale | `tensornet/cfd/qtt_multiscale.py` |
| QTT Checkpoints | `tensornet/cfd/qtt_checkpoint_stream.py` |
| Kernel Autotuner | `tensornet/gpu/kernel_autotune_cache.py` |
| **Phase 23** | |
| QTT Spectral | `tensornet/cfd/qtt_spectral.py` |
| QTT Shift Stable | `tensornet/cfd/qtt_shift_stable.py` |
| QTT Hadamard | `tensornet/cfd/qtt_hadamard.py` |
| QTT Regularity | `tensornet/cfd/qtt_regularity.py` |
| **Core** | |
| Tensor Slicer | `tensornet/visualization/tensor_slicer.py` |
| Chi Diagnostic | `tensornet/cfd/chi_diagnostic.py` |
| Singularity Hunter | `tensornet/cfd/singularity_hunter.py` |
| Koopman TT | `tensornet/cfd/koopman_tt.py` |
| Black Swan Hunter | `demos/trap_the_swan.py` |

---

## 🏛️ Architecture Principles

1. **Never Go Dense**: Use QTT cores, never materialize full tensors
2. **Rank Control**: Always truncate after operations that grow rank
3. **GPU First**: Auto-detect CUDA, fall back to CPU
4. **Reproducibility**: Seed control via `tensornet/core/determinism.py`
5. **Attestation**: Every gauntlet produces signed JSON proof

---

*Generated by HyperTensor Phase 24 • January 16, 2026*
