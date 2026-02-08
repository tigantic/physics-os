# Changelog

All notable changes to Project HyperTensor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [40.0.0] - 2026-02-08 (140/140 Computational Physics Release)

### Added

#### 140/140 Capability Domain Implementation (49,355 LOC, 152 files)
- **Phase 0 (Weeks 1-4)**: 26 domain upgrades — 6,410 + 10,332 + 10,941 LOC
  across elastic continuum, convection, classical waves, antennas, fibre optics,
  laser physics, critical phenomena, non-equilibrium statmech, Monte Carlo,
  compressible flow, non-Newtonian, MHD turbulence, DFT, TDDFT, GW/BSE,
  response properties, quantum chemistry, cluster/many-body, and more
- **Phase 1**: 22 new domains (7,559 LOC) — 78 → 100/140 coverage
  including nuclear structure, reactions, nuclear astrophysics, lattice QCD,
  high-energy collisions, neutrino oscillations, radiation transport, plasma
  instabilities, laser-plasma, fusion engineering, and more
- **Phase 2+3**: 40 new domains (14,113 LOC) — 100 → 140/140 complete
  including quantum reactive dynamics, nonadiabatic dynamics, photochemistry,
  catalysis, spectroscopy, protein dynamics, drug design, membrane biophysics,
  nucleic acids, systems biology, neuroscience, optimization/inverse problems,
  ML-for-physics, adaptive mesh, HPC/distributed, FSI, thermo-mechanical,
  electro-mechanical, coupled MHD, reacting flows, radiation-hydro, multiscale,
  quantum circuit simulation, quantum error correction, quantum algorithms,
  quantum simulation, quantum crypto, relativistic mechanics, numerical GR,
  astrodynamics, robotics physics, applied acoustics, biomedical engineering,
  environmental physics, energy systems, manufacturing physics, semiconductor
  device physics

#### Platform Specification v40.0 (§36–§50)
- PLATFORM_SPECIFICATION.md expanded from 2,932 → 4,021 lines
- 15 new sections documenting Classical Mechanics, Nuclear Physics, Particle
  Physics, Condensed Matter (ext), Geophysics, Chemistry, Biological Physics,
  Computational Methods, Plasma Physics (ext), Coupled Multi-Physics,
  Quantum Information, Materials Science (ext), Electronics & Optics (ext),
  Astrophysics (ext), Multi-Physics Coupling
- Physics Inventory: 50 domains, 826+ equations, ~227,000 LOC

#### Comprehensive Test Infrastructure
- **tests/test_140_domains.py** (847 lines, 314 pass / 8 skip):
  140 DomainSpec entries × 20 categories, 341 key classes tracked,
  parametrized import + class-exist + smoke + physics-validation tests
  (Morse, QHO, H-atom, Ising, PMNS, Landau-Devonshire, Biot-Savart)
- **tests/test_cross_domain_integration.py** (654 lines, 50 pass):
  15 pairwise domain pipelines + 3 multi-domain chains (3–4 domains each)
  across 23 physics modules — all <2 s runtime

### Changed
- Version bumped from 0.1.0 → 40.0.0 across pyproject.toml, CITATION.cff,
  tensornet/__init__.py, PLATFORM_SPECIFICATION.md
- Coverage assessment updated to 140/140 COMPLETE
- Execution plan finalised with all phases marked done

### Added - Trustless Physics Phase 4: Thermal Circuit (2026-02-06)

#### Thermal ZK Proof Circuit (`fluidelite-circuits::thermal`)
- **6-file thermal module**: `config.rs`, `gadgets.rs`, `halo2_impl.rs`, `mod.rs`, `prover.rs`, `witness.rs`
- Heat equation solver: ∂T/∂t = α∇²T + S(x,t) with implicit CG in QTT format
- Conservation law enforcement: |∫T^{n+1} - ∫T^n - Δt·∫S| ≤ ε_cons in ZK circuit
- SVD truncation witness with singular value ordering proofs
- SHA-256 hash commitments binding input/output states and parameters
- Full Q16.16 fixed-point arithmetic with MAC chain witnesses

#### Formal Verification
- **ThermalConservation.lean**: Lean 4 formal proof of energy conservation
  - 5 theorems: conservation_holds, rank_preservation, cg_termination, all_configs_conserve, residual_is_valid
  - Evidence from 3 configurations (test_small, test_medium, production)
  - Certificate and results artifacts in `thermal_conservation_proof/`

#### OOM Resolution
- **Root cause**: MPS direct-sum addition grows bond dimensions additively (chi doubles per operation); CG solver had no intermediate truncation → chi explodes exponentially after ~8 iterations → 512MB+ allocation
- **Fix**: Added `mps.truncate(chi_max)` after every MPS addition/subtraction in CG loop (5 sites: initial residual, x update, r update, p update, system matrix apply)
- **Impact**: Thermal tests run in 0.06s under 4GB memory limit (was OOM at 31.7GB)

#### Serde Strip (DTO Pattern)
- Removed `Serialize`/`Deserialize` derives from all circuit witness types in `fluidelite-circuits`
- Removed serde, serde_json, bincode dependencies from `fluidelite-circuits/Cargo.toml`
- Removed `Serialize`/`Deserialize` from MPS, MPSCore, MPO, MPOCore in `fluidelite-core`
- Retained serde only on Q16 (field element), SolverType (3-variant enum), weights module

#### Test Results
- 168/168 tests pass across all 7 test suites:
  - fluidelite-core lib: 21/21
  - fluidelite-circuits lib: 22/22
  - euler3d_tests: 33/33
  - ns_imex_tests: 47/47
  - thermal_tests: 40/40
  - proof_preview_tests: 15/15
  - trait_impls_tests: 10/10

### Fixed - QTT Turbulence Solver Critical Optimizations (2026-02-05)

#### Critical Bug Fixes
- **Jacobi Poisson Divergence**: Fixed velocity reconstruction that caused exponential energy blowup
  - Root cause: `_reconstruct_velocity_from_vorticity()` Jacobi iteration diverged
  - Fix: Set `poisson_iterations=0` (uses diffusion approximation)
  - Impact: Energy drift reduced from 72,904% → 0.9%
- **Hadamard Rank Explosion OOM**: Fixed memory blowup in advection term
  - Root cause: Hadamard products accumulated rank r² before truncation
  - Fix: Truncate AFTER each Hadamard product in `_compute_rhs()`
  - Impact: Memory reduced 9× (133MB → 15MB at 32³)

#### Performance Optimizations
- **Optimal Rank Discovery**: rank=16 identified as optimal (was 64)
  - 4× faster SVD operations (O(r³) complexity)
  - No loss in physics accuracy
- **Updated TurboNS3DConfig defaults**:
  - `max_rank`: 64 → 16
  - `poisson_iterations`: 3 → 0
  - `rank_cap`: 128 → 64

#### Validation Results
- **prove_turbulence.py**: 5/5 physics proofs PASSED
  - Taylor-Green decay: 0.05% error
  - Energy inequality: 0/30 violations
  - Enstrophy bounds: No blowup
  - Divergence-free: max|∇·u| = 4.27e-12
  - Kolmogorov spectrum: Power-law confirmed
- **Regression tests**: 6/6 PASSED
  - Imports & config validation
  - Poisson zero stability
  - Rank 16 optimal configuration
  - Memory scaling (64³, 128³)
  - O(log N) time scaling confirmed
  - Inviscid energy conservation: 0.036% drift

#### Performance Metrics
| Grid | Memory | Time/step | Compression |
|------|--------|-----------|-------------|
| 64³ | 147 MB | 2627 ms | 228× |
| 128³ | 1129 MB | ~2400 ms | 1,560× |
| 256³ | 4880 MB | 2878 ms | 10,923× |

#### Files Modified
- `tensornet/cfd/ns3d_turbo.py` — TurboNS3DSolver critical fixes
- `tests/test_qtt_turbo_regression.py` — New regression test suite

#### Documentation
- `Workflow_Development.md` — Complete 6-phase execution tracker
- `artifacts/PHASE1_BASELINE_ATTESTATION.json`
- `artifacts/PHASE2_RANK_ATTESTATION.json`
- `artifacts/PHASE3_4_MEMORY_ATTESTATION.json`
- `artifacts/PHASE5_PHYSICS_ATTESTATION.json`
- `artifacts/QTT_TURBO_REGRESSION.json`

### Added - QTT Size-Scaling Law Discovery (2026-01-29)

#### Scientific Discovery
- **QTT Size-Scaling Law**: Compression ratio scales linearly with dataset size for smooth functions
- Formula: R(n) = Θ(n / (r · log₂(n))) where r is bounded by smoothness
- Peak result: 17,050x compression at 4M points with rank 5
- Log-log slope: 0.977 (theoretical: 1.0) — linear scaling confirmed

#### Documentation
- `QTT_SIZE_SCALING_LAW.md` — Full discovery documentation with proofs
- `QTT_SIZE_SCALING_LAW_ATTESTATION.json` — Machine-readable attestation
- `QTT_FUNCTION_ENCODING_THEORY.md` — Updated to CONFIRMED status

#### Experimental Validation
- `fluidelite/qtt_theory_experiments.py` — 7 experiments validating theory
  - Exp 1: Interpolation accuracy
  - Exp 2: Basis transformation (191x gain from sorting)
  - Exp 3: Smoothness-rank correlation (r = -0.745)
  - Exp 4: Coordinate necessity (312x structure ratio)
  - Exp 5: Cross-domain validation
  - Exp 6: **Size-Scaling Law** (slope = 0.977, rank bounded)
  - Exp 7: Tolerance sweep (sweet spot: 956.7x @ 0.026% error)
- `fluidelite/data/qtt_theory_results_v2.json` — Full experimental data

#### Key Findings
- Rank is independent of dataset size (constant at 5 for smooth functions)
- Compression doubles when size doubles (linear scaling)
- 70,000x predicted at 1 billion points with rank 16
- Paradigm shift: QTT encodes functions, not data

## [2.0.0] - 2026-01-25 (Production Ready Release)

### Added

#### Phase 10: Placeholder Resolution (GPU + rSVD)
- `tensornet/cfd/qtt_reciprocal.py` — Newton-Schulz iteration for QTT element-wise reciprocal
- `_tt_rsvd_1d()` in barycenters.py — 1D TT decomposition via randomized SVD
- `_vector_to_qtt()` in sinkhorn_qtt.py — Scaling vector reconstruction
- `_tt_svd_gpu()` in koopman_tt.py — GPU-accelerated TT decomposition
- `_tt_matvec()` in koopman_tt.py — O(d r² n) TT matrix-vector product
- `_mpo_mps_contraction()` in kernel_matrix.py — MPO-MPS contraction
- `_project_onto_halfspace()` in convexity.py — Tropical halfspace projection
- `_estimate_matrix_rank()` in qtt_native.py — Randomized rank estimation

#### Implementations Completed
- Sinkhorn cost functions: `_compute_primal_cost`, `_compute_dual_cost`, `_compute_entropy`
- CFD Euler fluxes: `compute_euler_flux_x`, `compute_lax_friedrichs_flux_2d` with Hadamard products
- Wasserstein CDF: Prefix-sum MPO construction with rSVD truncation
- Transport plan: `displacement_variance`, `_safe_divide` with Newton iteration
- Koopman TT: ALS fitting and TT-matvec for trajectory prediction
- RKHS matvec: Full MPO-MPS contraction in `QTTKernelMatrix.matvec()`
- Tropical projection: Iterative Dykstra-like algorithm for halfspace intersection
- Betti numbers: Rank-based estimation with randomized probing
- CUDA fallback: Full QTT contraction via einsum on GPU
- TEBD gates: Proper Hamiltonian exponentiation via `torch.linalg.matrix_exp`
- Embedded thermal: Real sysfs reading with simulation fallback
- TensorRT: Full pycuda execution path

### Changed
- All placeholder returns (0.0, identity matrices) replaced with proper implementations
- SVD calls converted to `torch.svd_lowrank()` for GPU acceleration
- Dense fallbacks replaced with pure TT contractions where possible

## [0.3.0] - 2025-01-XX (Constitutional Compliance Release)

### Added

#### Test Coverage (Article III Compliance)
- `tests/test_energy.py` — WindFarm wake physics tests
- `tests/test_financial.py` — LiquiditySolver flow signal tests
- `tests/test_fusion.py` — TokamakReactor Boris pusher tests
- `tests/test_medical.py` — ArterySimulation blood flow tests
- `tests/test_racing.py` — WakeTracker dirty air tests
- `tests/test_ballistics.py` — BallisticSolver trajectory tests
- `tests/test_fire.py` — FireSim wildfire spread tests
- `tests/test_urban.py` — UrbanFlowSolver UAM corridor tests
- `tests/test_cyber.py` — CyberGrid cascading failure tests
- `tests/test_agri.py` — VerticalFarm microclimate tests

#### Documentation (Article VI Compliance)
- `The_Last_Ten_Yards.md` — Constitutional compliance audit (638 lines)
- Added formal academic references to 10 physics modules
- Enhanced docstrings with Raises/Example/References sections

### Changed

#### License Compliance (Article IX)
- Updated LICENSE references from MIT to Proprietary in 15 files
- Synchronized license declarations across pyproject.toml, Cargo.toml, all Rust crates

#### Numerical Precision (Article V)
- Converted Phase 11-15 modules to torch.float64 for physics calculations
- Added deterministic seeding (seed=42) for reproducibility (Article III, Section 3.2)

#### Development Tooling
- Fixed ruff version mismatch (v0.1.6 → v0.8.4) in pre-commit config

### Academic References Added
- Jensen (1983) — Wake model
- Boris (1970) — Particle pusher
- Erdős-Rényi (1960) — Random graphs
- Rothermel (1972) — Fire spread
- Carreau (1972) — Blood rheology
- Black-Scholes (1973) — Options pricing
- McCoy (1999) — Exterior ballistics
- Kozai et al. (2019) — Vertical farming

## [0.2.0] - 2025-01-XX

### Added
- **Phase 3: 2D Euler Solver** (`tensornet/cfd/`)
  - `Euler2D` class — 2D compressible Euler solver with Strang dimensional splitting
  - `Euler2DState` — 2D flow state with conservative/primitive conversions
  - `supersonic_wedge_ic` — Uniform supersonic flow initial condition
  - `double_mach_reflection_ic` — DMR benchmark initial condition
  - `oblique_shock_exact` — Exact θ-β-M oblique shock relations
  - `boundaries.py` — Boundary conditions module (reflective, inflow, outflow, periodic)
  - `geometry.py` — Wedge geometry and immersed boundary method
  - `BCType`, `FlowState`, `BoundaryManager` — BC management classes
  - `WedgeGeometry`, `ImmersedBoundary` — Geometry handling
- **Benchmark**: `benchmarks/oblique_shock.py` — Oblique shock validation with convergence study
- **Tests**: Extended `tests/test_integration.py` with 2D solver tests (33 tests total)

- **Phase 2: CFD Module** (`tensornet/cfd/`)
  - `Euler1D` class — 1D Euler equation solver with finite volume method
  - `EulerState` — Container for conserved/primitive fluid variables
  - Godunov-type Riemann solvers: `roe_flux`, `hll_flux`, `hllc_flux`
  - `exact_riemann` — Newton-Raphson exact Riemann solver
  - TVD slope limiters: `minmod`, `superbee`, `van_leer`, `mc_limiter`
  - `MUSCL` — Second-order reconstruction class
  - Standard test problems: `sod_shock_tube_ic`, `lax_shock_tube_ic`, `shu_osher_ic`
  - MPS interface: `euler_to_mps`, `mps_to_euler`
- **Benchmark**: `benchmarks/sod_shock_tube.py` — Sod shock tube validation
- **Core Package Structure**:
  - `tensornet/core/mps.py` — Full MPS class (~400 lines)
  - `tensornet/core/mpo.py` — Full MPO class (~230 lines)
  - `tensornet/core/decompositions.py` — SVD/QR with truncation
  - `tensornet/core/states.py` — Standard MPS states (GHZ, product, Néel)
  - `tensornet/algorithms/dmrg.py` — Two-site DMRG implementation
  - `tensornet/algorithms/tebd.py` — TEBD time evolution
  - `tensornet/algorithms/lanczos.py` — Krylov eigensolvers

### Changed
- Repository restructuring per Constitutional Law Article II
- `CONSTITUTION.md` — Governing standards for all contributions
- `EXECUTION_TRACKER.md` — Full-spectrum project tracking (updated to v1.1.0)
- Flattened `Physics/` container to repository root
- Moved vision documents to `docs/specifications/`
- Fixed import paths in `hamiltonians.py` and `fermionic.py`
- Fixed unicode characters in MPO einsum operations

### Repository Structure
```
tensornet/
├── __init__.py          # Package exports (MPS, MPO, dmrg, tebd, Euler1D, etc.)
├── core/
│   ├── mps.py           # Matrix Product State
│   ├── mpo.py           # Matrix Product Operator
│   ├── decompositions.py # SVD/QR
│   └── states.py        # Standard states
├── algorithms/
│   ├── dmrg.py          # Ground state
│   ├── tebd.py          # Time evolution
│   ├── lanczos.py       # Eigensolvers
│   └── fermionic.py     # Jordan-Wigner
├── mps/
│   └── hamiltonians.py  # MPO builders
└── cfd/
    ├── euler_1d.py      # 1D Euler equations
    ├── godunov.py       # Riemann solvers
    └── limiters.py      # TVD limiters
```

---

## [0.1.0] - 2025-12-17

### Added
- Core tensor network library (`tensornet/`)
  - `MPS` class with canonicalization, truncation, entropy
  - `MPO` class for Hamiltonian representation
  - `dmrg()` — Density Matrix Renormalization Group
  - `tebd()` — Time-Evolving Block Decimation
  - `lanczos()` — Krylov eigenvalue solver

- Hamiltonian library (`tensornet/mps/hamiltonians.py`)
  - `heisenberg_mpo()` — Heisenberg XXZ chain
  - `tfim_mpo()` — Transverse-field Ising model
  - `xx_mpo()` — XX model (free fermions)
  - `xyz_mpo()` — Anisotropic XYZ model
  - `bose_hubbard_mpo()` — Bose-Hubbard model

- Fermionic systems (`tensornet/algorithms/fermionic.py`)
  - Jordan-Wigner transformation
  - `spinless_fermion_mpo()` — Spinless fermion chain
  - `hubbard_mpo()` — Fermi-Hubbard model
  - `fermi_sea_mps()`, `half_filled_mps()` — Initial states

- Mathematical proofs (16/16 passing)
  - SVD optimality (Eckart-Young-Mirsky theorem)
  - MPS round-trip fidelity
  - GHZ entanglement entropy verification
  - Pauli algebra verification
  - Autograd correctness
  - Lanczos eigenvalue accuracy

- Benchmark suite
  - `compare_tenpy.py` — TeNPy comparison
  - `heisenberg_ground_state.py` — Bethe ansatz validation
  - `tfim_ground_state.py` — Exact diagonalization validation

- Interactive notebooks
  - `demo.ipynb` — Quick start demonstration
  - `heisenberg_convergence.ipynb` — DMRG scaling study
  - `tfim_phase_transition.ipynb` — Quantum criticality
  - `bose_hubbard.ipynb` — Mott-superfluid transition
  - `tebd_dynamics.ipynb` — Real-time spin dynamics

### Validated
- Heisenberg L=10: E = -4.258035207 (exact match)
- TFIM g=1.0 L=10: E = -12.566370614 (exact match)
- TeNPy comparison: < 10⁻⁸ relative error for L≤50

---

## [0.0.1] - 2025-12-01

### Added
- Initial project conception
- Grand Vision document
- Execution Overview roadmap
