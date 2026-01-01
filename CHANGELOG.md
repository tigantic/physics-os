# Changelog

All notable changes to Project HyperTensor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
