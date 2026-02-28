# physics-os Domain Pack Audit

**Generated from**: `ontic/packs/pack_i.py` through `pack_xx.py`
**Total packs**: 20 | **Total taxonomy nodes**: 166

---

## Summary Table

| Pack | Domain | Version | Nodes | Anchor (V0.4) Spec | Constructor kwargs? |
|------|--------|---------|-------|--------------------|---------------------|
| I | Classical Mechanics | 0.2.0 | 8 | None | No |
| II | Fluid Dynamics | 0.4.0 | 10 | BurgersSpec | Yes |
| III | Electromagnetism | 0.4.0 | 7 | Maxwell1DSpec | Yes |
| IV | Optics and Photonics | 0.2.0 | 7 | None | No |
| V | Thermo & Stat Mech | 0.4.0 | 6 | AdvectionDiffusionSpec | Yes |
| VI | Condensed Matter | 0.2.0 | 10 | None | No |
| VII | Quantum Many-Body | 0.4.0 | 13 | HeisenbergSpec | Yes |
| VIII | Density Functional Theory | 0.4.0 | 10 | KohnShamSpec | Yes |
| IX | Nuclear Physics | 0.2.0 | 8 | None | No |
| X | Nuclear & Particle Physics | 0.4.0 | 9 | None (PWA added at L1427) | No |
| XI | Plasma Physics | 0.4.0 | 10 | VlasovPoissonSpec | Yes |
| XII | Astrophysics | 0.2.0 | 10 | None | No |
| XIII | Geophysics | 0.2.0 | 8 | None | No |
| XIV | Biophysics | 0.2.0 | 8 | None | No |
| XV | Chemical Physics | 0.2.0 | 8 | None | No |
| XVI | Materials Science | 0.2.0 | 8 | None | No |
| XVII | Acoustics | 0.2.0 | 6 | None | No |
| XVIII | Atmospheric Physics | 0.2.0 | 8 | None | No |
| XIX | Quantum Computing | 0.2.0 | 8 | None | No |
| XX | Nonlinear Dynamics | 0.2.0 | 6 | None | No |

---

## Pack I — Classical Mechanics

**File**: `ontic/packs/pack_i.py` (970 lines)
**Pack class**: `ClassicalMechanicsPack` | **Version**: 0.2.0
**Solver return style**: Types (classes) via `dict(_SOLVERS)`

| Node | Spec Class | ndim | field_names | Constructor kwargs | Solver Class | Base |
|------|-----------|------|-------------|-------------------|-------------|------|
| PHY-I.1 | NBodySpec | 2 | (x, y, vx, vy) | None | NBodySolver | ODEReferenceSolver |
| PHY-I.2 | RigidBodySpec | 0 | (theta, omega) | None | RigidBodySolver | ODEReferenceSolver |
| PHY-I.3 | LagrangianSpec | 0 | (theta, omega) | None | LagrangianSolver | ODEReferenceSolver |
| PHY-I.4 | HamiltonianSpec | 0 | (q, p) | None | HamiltonianSolver | ODEReferenceSolver |
| PHY-I.5 | OrbitalSpec | 2 | (x, y, vx, vy) | None | OrbitalSolver | ODEReferenceSolver |
| PHY-I.6 | VibrationsSpec | 0 | (x1, x2, v1, v2) | None | VibrationsSolver | ODEReferenceSolver |
| PHY-I.7 | ContinuumSpec | 1 | (u, v) | None | ContinuumSolver | PDE1DReferenceSolver |
| PHY-I.8 | ChaosSpec | 0 | (x, y, z) | None | ChaosSolver | ODEReferenceSolver |

**Parameters** (all hardcoded in properties):
- I.1: `G=1.0, m=[1.0,1.0,1.0]`
- I.2: `I=1.0, g=9.81, L=1.0`
- I.3: `m=1.0, g=9.81, L=1.0`
- I.4: `m=1.0, k=1.0`
- I.5: `GM=1.0`
- I.6: `K=[[2,-1],[-1,2]], M=I`
- I.7: `c=1.0, N=128`
- I.8: `sigma=10, rho=28, beta=8/3`

---

## Pack II — Fluid Dynamics

**File**: `ontic/packs/pack_ii.py` (1416 lines)
**Pack class**: `FluidDynamicsPack` | **Version**: 0.4.0
**Solver return style**: Types (classes)
**Has**: Discretizations (`FVM_Burgers_1D`), Observables (`BurgersL2Observable`, `BurgersIntegralObservable`, `BurgersKEObservable`), Benchmarks, Vertical slice (`run_fluids_vertical_slice`)

| Node | Spec Class | ndim | field_names | Constructor kwargs (defaults) | Solver Class | Base |
|------|-----------|------|-------------|-------------------------------|-------------|------|
| PHY-II.1 | **BurgersSpec** | 1 | (u,) | **nu=0.01, L=2π, T_final=0.5** | BurgersSolver | — |
| PHY-II.2 | CompressibleFlowSpec | 1 | (rho, u, p) | gamma=1.4 | CompressibleFlowSolver | — |
| PHY-II.3 | TurbulenceSpec | 1 | (u,) | nu=0.001 | TurbulenceSolver | — |
| PHY-II.4 | MultiphaseSpec | 1 | (alpha, u) | None | MultiphaseSolver | — |
| PHY-II.5 | ReactiveFlowSpec | 1 | (rho, u, Y) | Da=10.0 | ReactiveFlowSolver | — |
| PHY-II.6 | RarefiedGasSpec | 1 | (f,) | None | RarefiedGasSolver | — |
| PHY-II.7 | ShallowWaterSpec | 1 | (h, hu) | g=9.81 | ShallowWaterSolver | — |
| PHY-II.8 | NonNewtonianSpec | 1 | (u,) | tau_y=1.0, mu_p=0.1 | NonNewtonianSolver | — |
| PHY-II.9 | PorousMediaSpec | 1 | (P,) | K=1e-10, mu=1e-3 | PorousMediaSolver | — |
| PHY-II.10 | FreeSurfaceSpec | 1 | (eta, u) | None | FreeSurfaceSolver | — |

**Complexity controls (anchor)**:
- `BurgersSpec.nu` → viscosity (lower = sharper shocks, harder)
- `BurgersSpec.T_final` → integration time
- BurgersSolver reads `nu` from spec

---

## Pack III — Electromagnetism

**File**: `ontic/packs/pack_iii.py` (1338 lines)
**Pack class**: `ElectromagnetismPack` | **Version**: 0.4.0
**Has**: Discretizations (`FDTD_1D`), Observables (`EMEnergyObservable`), Benchmarks, Vertical slice (`run_em_vertical_slice`)

| Node | Spec Class | ndim | field_names | Constructor kwargs (defaults) | Solver Class | Base |
|------|-----------|------|-------------|-------------------------------|-------------|------|
| PHY-III.1 | ElectrostaticsSpec | 1 | (phi, E) | None | ElectrostaticsSolver | — |
| PHY-III.2 | MagnetostaticsSpec | 1 | (A, B) | None | MagnetostaticsSolver | — |
| PHY-III.3 | **Maxwell1DSpec** | 1 | (E, H) | **epsilon=1.0, mu=1.0, L=10.0, T_final=4.0, sigma_pulse=0.3, x0_pulse=5.0** | MaxwellSolver | — |
| PHY-III.4 | FreqDomainEMSpec | 1 | (E_r, E_i) | k=6.283 | FreqDomainEMSolver | — |
| PHY-III.5 | WavePropagationSpec | 1 | (E, H) | c=1.0 | WavePropagationSolver | — |
| PHY-III.6 | PhotonicsSpec | 1 | (E, n_eff) | None | PhotonicsSolver | — |
| PHY-III.7 | AntennasSpec | 1 | (I_pattern,) | None | AntennasSolver | — |

**Complexity controls (anchor)**:
- `Maxwell1DSpec.epsilon`, `.mu` → material properties
- `Maxwell1DSpec.sigma_pulse` → narrower pulse = harder resolution
- MaxwellSolver takes `epsilon`, `mu` in `__init__`

---

## Pack IV — Optics and Photonics

**File**: `ontic/packs/pack_iv.py` (1298 lines)
**Pack class**: `OpticsPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | field_names | Constructor kwargs | Solver Class | Base |
|------|-----------|------|-------------|-------------------|-------------|------|
| PHY-IV.1 | RayTracingSpec | 1 | (x, y, slope_x, slope_y) | None | RayTracingSolver | ODEReferenceSolver |
| PHY-IV.2 | WaveOpticsSpec | 0 | (amplitude, phase) | None | WaveOpticsSolver | ODEReferenceSolver |
| PHY-IV.3 | FiberOpticsSpec | 1 | (field_profile,) | None | FiberOpticsSolver | ODEReferenceSolver |
| PHY-IV.4 | FourierOpticsSpec | 0 | (amplitude, phase) | None | FourierOpticsSolver | — |
| PHY-IV.5 | NonlinearOpticsSpec | 0 | (field, chi2, chi3) | None | NonlinearOpticsSolver | — |
| PHY-IV.6 | QuantumOpticsSpec | 0 | (state_vector,) | None | QuantumOpticsSolver | — |
| PHY-IV.7 | PhotonicCrystalSpec | 1 | (band_structure,) | None | PhotonicCrystalSolver | EigenReferenceSolver |

All parameters hardcoded in properties. No complexity controls exposed.

---

## Pack V — Thermodynamics and Statistical Mechanics

**File**: `ontic/packs/pack_v.py` (709 lines)
**Pack class**: `ThermoStatMechPack` | **Version**: 0.4.0
**Has**: Discretizations (`FVM_AdvDiff_1D`), Observables (`AdvDiffL2Observable`, `AdvDiffIntegralObservable`), Benchmarks, Vertical slice (`run_heat_vertical_slice`)

| Node | Spec Class | ndim | field_names | Constructor kwargs (defaults) | Solver Class | Base |
|------|-----------|------|-------------|-------------------------------|-------------|------|
| PHY-V.1 | IsingEnergySpec | 1 | (spins,) | J=1.0, N_spins=32 | IsingMCSolver | MonteCarloReferenceSolver |
| PHY-V.2 | FokkerPlanckSpec | 1 | (p,) | D=0.1, mu=0.0 | DiffusionODESolver | ODEReferenceSolver |
| PHY-V.3 | LennardJonesSpec | 0 | (r, v) | epsilon=1.0, sigma=1.0 | LennardJonesSolver | — |
| PHY-V.4 | RandomWalkSpec | 1 | (positions,) | D=0.1, N_walkers=1000 | RandomWalkSolver | MonteCarloReferenceSolver |
| PHY-V.5 | **AdvectionDiffusionSpec** | 1 | (u,) | **c=1.0, alpha=0.01, L=1.0, T_final=0.5** | AdvDiffSolver | — |
| PHY-V.6 | IsingPartitionSpec | 0 | (Z, F, S) | J=1.0, h=0.0, N_spins=16 | IsingPartitionSolver | — |

**Complexity controls (anchor)**:
- `AdvectionDiffusionSpec.alpha` → diffusion coefficient (lower = stiffer)
- `AdvectionDiffusionSpec.c` → advection speed
- All V0.2 scaffold specs also have constructor kwargs

---

## Pack VI — Condensed Matter Physics

**File**: `ontic/packs/pack_vi.py` (1718 lines)
**Pack class**: `CondensedMatterPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | field_names | Constructor kwargs | Solver Class | Base |
|------|-----------|------|-------------|-------------------|-------------|------|
| PHY-VI.1 | BandStructureSpec | 1 | (eigenvalues, eigenvectors) | None | BandStructureSolver | EigenReferenceSolver |
| PHY-VI.2 | PhononSpec | 1 | (omega, displacement) | None | PhononSolver | EigenReferenceSolver |
| PHY-VI.3 | SuperconductivitySpec | 0 | (gap, critical_temp) | None | SuperconductivitySolver | — |
| PHY-VI.4 | MagnetismSpec | 0 | (magnetization, susceptibility) | None | MagnetismSolver | — |
| PHY-VI.5 | TopologicalInsulatorSpec | — | — | None | TopologicalInsulatorSolver | — |
| PHY-VI.6 | StronglyCorrelatedSpec | — | — | None | StronglyCorrelatedSolver | — |
| PHY-VI.7 | MesoscopicSpec | — | — | None | MesoscopicSolver | — |
| PHY-VI.8 | SurfacePhysicsSpec | — | — | None | SurfacePhysicsSolver | — |
| PHY-VI.9 | DisorderedSystemsSpec | — | — | None | DisorderedSystemsSolver | — |
| PHY-VI.10 | PhaseTransitionsSpec | — | — | None | PhaseTransitionsSolver | — |

All parameters hardcoded. No constructor kwargs.

---

## Pack VII — Quantum Many-Body Physics

**File**: `ontic/packs/pack_vii.py` (1706 lines)
**Pack class**: `QuantumManyBodyPack` | **Version**: 0.4.0
**Has**: Benchmarks, Vertical slice (`run_quantum_mb_vertical_slice`)

| Node | Spec Class | ndim | field_names | Constructor kwargs (defaults) | Solver Class | Base |
|------|-----------|------|-------------|-------------------------------|-------------|------|
| PHY-VII.1 | TNMethodsSpec | 1 | (mps_tensors,) | None | HeisenbergSolver | — |
| PHY-VII.2 | **HeisenbergSpec** | 1 | (mps_tensors,) | **J=1.0, N_sites=8** | HeisenbergSolver | — |
| PHY-VII.3 | CorrelatedElectronsSpec | — | — | None | CorrelatedElectronsSolver | — |
| PHY-VII.4 | TopologicalPhasesSpec | — | — | None | TopologicalPhasesSolver | — |
| PHY-VII.5 | MBLSpec | — | — | None | MBLocalizationSolver | — |
| PHY-VII.6 | LatticeGaugeSpec | — | — | None | LatticeGaugeSolver | — |
| PHY-VII.7 | OpenQuantumSpec | — | — | None | OpenQuantumSolver | — |
| PHY-VII.8 | NonEqQuantumSpec | — | — | None | NonEqQuantumSolver | — |
| PHY-VII.9 | QuantumImpuritySpec | — | — | None | QuantumImpuritySolver | — |
| PHY-VII.10 | BosonicMBSpec | — | — | None | BosonicMBSolver | — |
| PHY-VII.11 | FermionicSpec | — | — | None | FermionicSolver | — |
| PHY-VII.12 | NuclearMBSpec | — | — | None | NuclearMBSolver | — |
| PHY-VII.13 | UltracoldSpec | — | — | None | UltracoldSolver | — |

**Complexity controls (anchor)**:
- `HeisenbergSpec.N_sites` → chain length (exponential Hilbert space growth)
- `HeisenbergSpec.J` → exchange coupling
- `HeisenbergSolver.__init__(chi=16, tau=0.05, n_steps=200)` → bond dimension, Trotter step, iteration count
- Scaffold specs (VII.3–VII.13) generated via `_make_scaffold_spec` factory (ndim=3)

---

## Pack VIII — Density Functional Theory

**File**: `ontic/packs/pack_viii.py` (641 lines)
**Pack class**: `DensityFunctionalTheoryPack` | **Version**: 0.4.0
**Solver return style**: **Instances** (e.g. `KohnShamSolver()`, `_ScaffoldSolver("...")`)
**Has**: Discretizations (`KS_FD_1D`), Observables (`TotalEnergyObs`), Vertical slice (`run_dft_vertical_slice`)

| Node | Spec Class | ndim | field_names | Constructor kwargs (defaults) | Solver Class | Return |
|------|-----------|------|-------------|-------------------------------|-------------|--------|
| PHY-VIII.1 | **KohnShamSpec** | 1 | (density, orbitals) | **Z=2.0, a=1.0, N_electrons=2, L=20.0** | KohnShamSolver | Instance |
| PHY-VIII.2 | XCFunctionalSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.3 | PseudopotentialSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.4 | PlaneWaveBasisSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.5 | LocalizedBasisSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.6 | HybridFunctionalSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.7 | TDDFTSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.8 | ResponseFunctionSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.9 | BandStructureSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-VIII.10 | AIMDSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |

**Complexity controls (anchor)**:
- `KohnShamSpec.Z` → nuclear charge
- `KohnShamSpec.N_electrons` → electron count
- `KohnShamSolver.__init__(N_grid=400, max_iter=300, mix_alpha=0.3, tol=1e-10)` → grid resolution, SCF convergence
- Scaffold specs (VIII.2–VIII.10) created via `_make_scaffold_spec(ndim=3)`

---

## Pack IX — Nuclear Physics

**File**: `ontic/packs/pack_ix.py` (1359 lines)
**Pack class**: `NuclearPhysicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | field_names | Constructor kwargs | Solver Class | Base |
|------|-----------|------|-------------|-------------------|-------------|------|
| PHY-IX.1 | ShellModelSpec | 3 | (eigenvalues, wavefunctions) | None | ShellModelSolver | EigenReferenceSolver |
| PHY-IX.2 | NuclearReactionsSpec | 0 | (Q_value, cross_section) | None | NuclearReactionsSolver | — |
| PHY-IX.3 | FissionSpec | 0 | (barrier, fragments) | None | FissionSolver | — |
| PHY-IX.4 | FusionSpec | 0 | (cross_section, rate) | None | FusionSolver | — |
| PHY-IX.5 | NuclearStructureSpec | 0 | (binding_energy, radii) | None | NuclearStructureSolver | — |
| PHY-IX.6 | DecaySpec | 0 | (half_life, spectrum) | None | DecaySolver | — |
| PHY-IX.7 | ScatteringSpec | 0 | (cross_section, phase_shift) | None | ScatteringSolver | — |
| PHY-IX.8 | NucleosynthesisSpec | 0 | (abundances,) | None | NucleosynthesisSolver | — |

All parameters hardcoded. ShellModelSolver uses physical constants (`_AMU_TO_MEV`, `_HBAR_C_MEV_FM`, etc.).

---

## Pack X — Nuclear and Particle Physics

**File**: `ontic/packs/pack_x.py` (1469 lines)
**Pack class**: `ParticlePhysicsPack` | **Version**: 0.4.0
**Solver return style**: Types via `dict(_SOLVERS)`
**Note**: PHY-X.9 (PWASpec/PWASolver) appended at line 1427–1428

| Node | Spec Class | ndim | field_names | Constructor kwargs | Solver Class | Base |
|------|-----------|------|-------------|-------------------|-------------|------|
| PHY-X.1 | QCDSpec | 0 | (coupling, beta_function) | None | QCDSolver | — |
| PHY-X.2 | ElectroweakSpec | 0 | (cross_section, asymmetry) | None | ElectroweakSolver | — |
| PHY-X.3 | BSMSpec | 0 | (spectrum, couplings) | None | BSMSolver | EigenReferenceSolver |
| PHY-X.4 | LatticeQCDSpec | 1 | (plaquette, wilson_loop) | None | LatticeQCDSolver | — |
| PHY-X.5 | PartonSpec | — | — | None | PartonSolver | — |
| PHY-X.6 | ColliderSpec | — | — | None | ColliderSolver | — |
| PHY-X.7 | DarkMatterSpec | — | — | None | DarkMatterSolver | — |
| PHY-X.8 | NeutrinoSpec | — | — | None | NeutrinoSolver | — |
| PHY-X.9 | PWASpec | — | — | None | PWASolver | — |

No constructor kwargs on any spec.

---

## Pack XI — Plasma Physics

**File**: `ontic/packs/pack_xi.py` (657 lines)
**Pack class**: `PlasmaPhysicsPack` | **Version**: 0.4.0
**Solver return style**: **Instances** (e.g. `VlasovSolver()`, `_ScaffoldSolver("...")`)
**Has**: Discretizations (`VP_Grid_2D`), Observables (`EFieldEnergyObs`), Vertical slice (`run_plasma_vertical_slice`)

| Node | Spec Class | ndim | field_names | Constructor kwargs (defaults) | Solver Class | Return |
|------|-----------|------|-------------|-------------------------------|-------------|--------|
| PHY-XI.1 | **VlasovPoissonSpec** | 2 | (f, E_field) | **epsilon=0.01, k_mode=0.5, L=4π, v_max=6.0** | VlasovSolver | Instance |
| PHY-XI.2 | MHDSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.3 | GyrokineticSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.4 | PICSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.5 | FokkerPlanckPlasmaSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.6 | DispersionRelSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.7 | PlasmaWaveSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.8 | ReconnectionSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.9 | IonAcousticSpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |
| PHY-XI.10 | PlasmaInstabilitySpec | 3 | — | None (scaffold) | _ScaffoldSolver | Instance |

**Complexity controls (anchor)**:
- `VlasovPoissonSpec.epsilon` → perturbation amplitude
- `VlasovPoissonSpec.k_mode` → wavenumber
- `VlasovSolver.__init__(Nx=64, Nv=128, dt=0.1)` → phase-space resolution
- Strang splitting + semi-Lagrangian advection

---

## Pack XII — Astrophysics

**File**: `ontic/packs/pack_xii.py` (1575 lines)
**Pack class**: `AstrophysicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XII.1 | StellarStructureSpec | 1 | StellarStructureSolver |
| PHY-XII.2 | GalaxyDynamicsSpec | 0 | GalaxyDynamicsSolver |
| PHY-XII.3 | CosmologySpec | 1 | CosmologySolver |
| PHY-XII.4 | GravitationalWaveSpec | — | GravitationalWaveSolver |
| PHY-XII.5 | CompactObjectsSpec | — | CompactObjectsSolver |
| PHY-XII.6 | InterstellarMediumSpec | — | InterstellarMediumSolver |
| PHY-XII.7 | AccretionSpec | — | AccretionSolver |
| PHY-XII.8 | RadiationTransportSpec | — | RadiationTransportSolver |
| PHY-XII.9 | DarkEnergySpec | — | DarkEnergySolver |
| PHY-XII.10 | CMBSpec | — | CMBSolver |

No constructor kwargs. All parameters hardcoded.

---

## Pack XIII — Geophysics

**File**: `ontic/packs/pack_xiii.py` (1433 lines)
**Pack class**: `GeophysicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XIII.1 | SeismicWaveSpec | 1 | SeismicWaveSolver (PDE1DReferenceSolver) |
| PHY-XIII.2 | MantleConvectionSpec | 0 | MantleConvectionSolver |
| PHY-XIII.3 | GeomagnetismSpec | 0 | GeomagnetismSolver |
| PHY-XIII.4 | GlaciologySpec | — | GlaciologySolver |
| PHY-XIII.5 | OceanCirculationSpec | — | OceanCirculationSolver |
| PHY-XIII.6 | TectonicsSpec | — | TectonicsSolver |
| PHY-XIII.7 | VolcanologySpec | — | VolcanologySolver |
| PHY-XIII.8 | GeodesySpec | — | GeodesySolver |

No constructor kwargs.

---

## Pack XIV — Biophysics

**File**: `ontic/packs/pack_xiv.py` (1574 lines)
**Pack class**: `BiophysicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XIV.1 | MolecularDynamicsSpec | 1 | MolecularDynamicsSolver (Verlet) |
| PHY-XIV.2 | ProteinFoldingSpec | 1 | ProteinFoldingSolver |
| PHY-XIV.3 | MembraneMechanicsSpec | 0 | MembraneMechanicsSolver |
| PHY-XIV.4 | NeuralModelSpec | — | NeuralModelSolver |
| PHY-XIV.5 | PopulationDynamicsSpec | — | PopulationDynamicsSolver |
| PHY-XIV.6 | EpidemiologySpec | — | EpidemiologySolver |
| PHY-XIV.7 | BiomechanicsSpec | — | BiomechanicsSolver |
| PHY-XIV.8 | CellSignalingSpec | — | CellSignalingSolver |

No constructor kwargs.

---

## Pack XV — Chemical Physics

**File**: `ontic/packs/pack_xv.py` (1516 lines)
**Pack class**: `ChemicalPhysicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `dict(_SOLVERS)`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XV.1 | ReactionKineticsSpec | 1 | ReactionKineticsSolver (RK4) |
| PHY-XV.2 | MolecularSpectroscopySpec | 0 | MolecularSpectroscopySolver |
| PHY-XV.3 | QuantumChemistrySpec | 0 | QuantumChemistrySolver |
| PHY-XV.4 | SurfaceChemistrySpec | — | SurfaceChemistrySolver |
| PHY-XV.5 | ElectrochemistrySpec | — | ElectrochemistrySolver |
| PHY-XV.6 | PolymerPhysicsSpec | — | PolymerPhysicsSolver |
| PHY-XV.7 | ColloidScienceSpec | — | ColloidScienceSolver |
| PHY-XV.8 | CombustionSpec | — | CombustionSolver |

No constructor kwargs.

---

## Pack XVI — Materials Science

**File**: `ontic/packs/pack_xvi.py` (747 lines)
**Pack class**: `MaterialsSciencePack` | **Version**: 0.2.0
**Solver return style**: Types via `_NODE_MAP` (tuple unpacking)

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XVI.1 | CrystalGrowthSpec | 1 | CrystalGrowthSolver |
| PHY-XVI.2 | FractureMechanicsSpec | 1 | FractureMechanicsSolver |
| PHY-XVI.3 | CorrosionSpec | 1 | CorrosionSolver |
| PHY-XVI.4 | ThinFilmsSpec | 1 | ThinFilmsSolver |
| PHY-XVI.5 | NanostructuresSpec | 3 | NanostructuresSolver (EigenReferenceSolver) |
| PHY-XVI.6 | CompositesSpec | 1 | CompositesSolver |
| PHY-XVI.7 | MetamaterialsSpec | — | MetamaterialsSolver |
| PHY-XVI.8 | PhaseFieldSpec | — | PhaseFieldSolver |

No constructor kwargs.

---

## Pack XVII — Acoustics

**File**: `ontic/packs/pack_xvii.py` (906 lines)
**Pack class**: `AcousticsPack` | **Version**: 0.2.0
**Solver return style**: Types via `_NODE_MAP`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XVII.1 | LinearAcousticsSpec | 1 | LinearAcousticsSolver (PDE1DReferenceSolver) |
| PHY-XVII.2 | NonlinearAcousticsSpec | 1 | NonlinearAcousticsSolver |
| PHY-XVII.3 | AeroacousticsSpec | 1 | AeroacousticsSolver |
| PHY-XVII.4 | UnderwaterAcousticsSpec | — | UnderwaterAcousticsSolver |
| PHY-XVII.5 | UltrasoundSpec | — | UltrasoundSolver |
| PHY-XVII.6 | RoomAcousticsSpec | — | RoomAcousticsSolver |

No constructor kwargs.

---

## Pack XVIII — Atmospheric Physics

**File**: `ontic/packs/pack_xviii.py` (1262 lines)
**Pack class**: `AtmosphericPhysicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `_NODE_MAP`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XVIII.1 | WeatherPredictionSpec | 1 | WeatherPredictionSolver (Lorenz '96) |
| PHY-XVIII.2 | ClimateModelingSpec | 0 | ClimateModelingSolver (EBM) |
| PHY-XVIII.3 | AtmosphericChemistrySpec | 0 | AtmosphericChemistrySolver (Chapman) |
| PHY-XVIII.4 | BoundaryLayerSpec | — | BoundaryLayerSolver |
| PHY-XVIII.5 | CloudPhysicsSpec | — | CloudPhysicsSolver |
| PHY-XVIII.6 | RadiationSpec | — | RadiationSolver |
| PHY-XVIII.7 | TurbulenceSpec | — | TurbulenceSolver |
| PHY-XVIII.8 | DataAssimilationSpec | — | DataAssimilationSolver |

No constructor kwargs.

---

## Pack XIX — Quantum Computing

**File**: `ontic/packs/pack_xix.py` (1439 lines)
**Pack class**: `QuantumComputingPack` | **Version**: 0.2.0
**Solver return style**: Types via `_NODE_MAP`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XIX.1 | QuantumCircuitsSpec | 0 | QuantumCircuitsSolver |
| PHY-XIX.2 | QuantumErrorCorrectionSpec | 0 | QuantumErrorCorrectionSolver |
| PHY-XIX.3 | QuantumAlgorithmsSpec | 0 | QuantumAlgorithmsSolver |
| PHY-XIX.4 | EntanglementSpec | — | EntanglementSolver |
| PHY-XIX.5 | QuantumCommunicationSpec | — | QuantumCommunicationSolver |
| PHY-XIX.6 | QuantumSensingSpec | — | QuantumSensingSolver |
| PHY-XIX.7 | QuantumSimulationSpec | — | QuantumSimulationSolver |
| PHY-XIX.8 | QuantumCryptographySpec | — | QuantumCryptographySolver |

No constructor kwargs.

---

## Pack XX — Nonlinear Dynamics

**File**: `ontic/packs/pack_xx.py` (1171 lines)
**Pack class**: `NonlinearDynamicsPack` | **Version**: 0.2.0
**Solver return style**: Types via `_NODE_MAP`

| Node | Spec Class | ndim | Solver Class |
|------|-----------|------|-------------|
| PHY-XX.1 | SolitonsSpec | 1 | SolitonsSolver (split-step Fourier) |
| PHY-XX.2 | PatternFormationSpec | 1 | PatternFormationSolver |
| PHY-XX.3 | BifurcationSpec | 0 | BifurcationSolver |
| PHY-XX.4 | SynchronizationSpec | — | SynchronizationSolver |
| PHY-XX.5 | ComplexNetworksSpec | — | ComplexNetworksSolver |
| PHY-XX.6 | StochasticDynamicsSpec | — | StochasticDynamicsSolver |

No constructor kwargs.

---

## Cross-Cutting Analysis

### Maturity Tiers

**V0.4 (6 packs)** — Full vertical slice with convergence study, determinism check, benchmarks:
- II (Fluid Dynamics), III (Electromagnetism), V (Thermo/StatMech), VII (Quantum Many-Body), VIII (DFT), XI (Plasma Physics)

**V0.4 (partial, 1 pack)** — PWA engine validated but no full vertical-slice runner:
- X (Nuclear & Particle)

**V0.2 (13 packs)** — All solvers implemented with validation but no convergence/benchmark infrastructure:
- I, IV, VI, IX, XII, XIII, XIV, XV, XVI, XVII, XVIII, XIX, XX

### Specs with Constructor kwargs (Complexity Controls)

| Spec | Pack | kwargs |
|------|------|--------|
| BurgersSpec | II | `nu`, `L`, `T_final` |
| CompressibleFlowSpec | II | `gamma` |
| TurbulenceSpec | II | `nu` |
| ReactiveFlowSpec | II | `Da` |
| ShallowWaterSpec | II | `g` |
| NonNewtonianSpec | II | `tau_y`, `mu_p` |
| PorousMediaSpec | II | `K`, `mu` |
| Maxwell1DSpec | III | `epsilon`, `mu`, `L`, `T_final`, `sigma_pulse`, `x0_pulse` |
| FreqDomainEMSpec | III | `k` |
| WavePropagationSpec | III | `c` |
| AdvectionDiffusionSpec | V | `c`, `alpha`, `L`, `T_final` |
| IsingEnergySpec | V | `J`, `N_spins` |
| FokkerPlanckSpec | V | `D`, `mu` |
| LennardJonesSpec | V | `epsilon`, `sigma` |
| RandomWalkSpec | V | `D`, `N_walkers` |
| IsingPartitionSpec | V | `J`, `h`, `N_spins` |
| HeisenbergSpec | VII | `J`, `N_sites` |
| KohnShamSpec | VIII | `Z`, `a`, `N_electrons`, `L` |
| VlasovPoissonSpec | XI | `epsilon`, `k_mode`, `L`, `v_max` |

### Solver Constructor kwargs (Resolution/Accuracy Controls)

| Solver | Pack | kwargs |
|--------|------|--------|
| HeisenbergSolver | VII | `chi=16`, `tau=0.05`, `n_steps=200` |
| KohnShamSolver | VIII | `N_grid=400`, `max_iter=300`, `mix_alpha=0.3`, `tol=1e-10` |
| VlasovSolver | XI | `Nx=64`, `Nv=128`, `dt=0.1` |

### Solver Return Patterns

- **Types** (most packs): `solvers()` returns `Dict[str, Type[Solver]]` — classes, not instances
- **Instances** (packs VIII, XI): `solvers()` returns pre-constructed objects like `KohnShamSolver()`, `VlasovSolver()`, `_ScaffoldSolver("...")`

### ndim Distribution

| ndim | Count | Description |
|------|-------|-------------|
| 0 | ~60 | ODE / algebraic / 0-D models |
| 1 | ~50 | 1-D PDE / spatial problems |
| 2 | 2 | Phase space (Vlasov, N-body) |
| 3 | ~25 | 3-D scaffold specs (VIII, XI scaffolds, IX.1, XVI.5) |

### Base Class Usage

| Base Class | Used By |
|-----------|---------|
| ODEReferenceSolver | Pack I (most), Pack IV (ray tracing, wave, fiber), Pack V (FokkerPlanck) |
| PDE1DReferenceSolver | Pack I (continuum), Pack XIII (seismic), Pack XVII (linear acoustics) |
| EigenReferenceSolver | Pack IV (photonic crystal), Pack VI (band structure, phonon), Pack IX (shell model), Pack X (BSM), Pack XVI (nanostructures) |
| MonteCarloReferenceSolver | Pack V (Ising, random walk) |
| Standalone (no base) | All V0.4 anchor solvers, most V0.2 scaffold solvers |
