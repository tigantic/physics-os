# Phase 8 — Tier 3 Solver Survey (45 Domains)

> Generated: 2026-02-16  
> Scope: Iterative/Eigenvalue domains 70–114

---

## Existing trace_adapters/ directories

| Directory | Status |
|-|-|
| `tensornet/astro/trace_adapters/` | EXISTS |
| `tensornet/cfd/trace_adapters/` | EXISTS |
| `tensornet/chemistry/trace_adapters/` | EXISTS |
| `tensornet/coupled/trace_adapters/` | EXISTS |
| `tensornet/em/trace_adapters/` | EXISTS |
| `tensornet/fluids/trace_adapters/` | EXISTS |
| `tensornet/geophysics/trace_adapters/` | EXISTS |
| `tensornet/materials/trace_adapters/` | EXISTS |
| `tensornet/mechanics/trace_adapters/` | EXISTS |
| `tensornet/optics/trace_adapters/` | EXISTS |
| `tensornet/plasma/trace_adapters/` | EXISTS |
| `tensornet/statmech/trace_adapters/` | EXISTS |

### New trace_adapters/ directories NEEDED for Phase 8

| Directory | Categories Served |
|-|-|
| `tensornet/qm/trace_adapters/` | VI. Quantum Mechanics (70–74) |
| `tensornet/quantum_mechanics/trace_adapters/` | VI. Quantum Mechanics (70–74, alt location) |
| `tensornet/condensed_matter/trace_adapters/` | VII. Quantum Many-Body (76–87), IX. Solid State (95–102) |
| `tensornet/electronic_structure/trace_adapters/` | VIII. Electronic Structure (88–94) |
| `tensornet/nuclear/trace_adapters/` | X. Nuclear & Particle (103–105) |
| `tensornet/qft/trace_adapters/` | X. Nuclear & Particle (106–108) |
| `tensornet/particle/trace_adapters/` | X. Nuclear & Particle (108) |
| `tensornet/quantum/trace_adapters/` | XIX. Quantum Information (112–114) |
| `tensornet/mps/trace_adapters/` | VII. Tensor Network / DMRG (75) |
| `tensornet/algorithms/trace_adapters/` | VII. Tensor Network / DMRG (75, alt) |

---

## VI. Quantum Mechanics (5 domains)

### 70. TISE — Time-Independent Schrödinger Equation

| Property | Value |
|-|-|
| **File** | `tensornet/quantum_mechanics/stationary.py` |
| **Primary class** | `DVRSolver` |
| **Constructor** | `DVRSolver(x_min: float, x_max: float, n_grid: int, mass: float = 1.0, hbar: float = 1.0)` |
| **Key methods** | `solve(potential, n_states) -> EigenResult`; `solve_2d(potential_2d, n_states)` |
| **Alternate classes** | `ShootingMethodSolver`, `SpectralSolver`, `WKBApproximation`, `HydrogenAtom`, `HarmonicOscillator` |
| **Adapter type** | **Eigenvalue** (diagonalises Hamiltonian matrix) |

### 71. TDSE — Time-Dependent Schrödinger Equation

| Property | Value |
|-|-|
| **File** | `tensornet/quantum_mechanics/propagator.py` |
| **Primary class** | `SplitOperatorPropagator` |
| **Constructor** | `SplitOperatorPropagator(x_min: float, x_max: float, n_grid: int, mass: float = 1.0, hbar: float = 1.0)` |
| **Key methods** | `propagate(psi0, potential, dt, n_steps) -> PropagationResult` |
| **Alternate classes** | `CrankNicolsonPropagator`, `ChebyshevPropagator`, `WavepacketTunneling` |
| **Adapter type** | **Time-step** (split-operator / Crank-Nicolson propagation) |

### 72. Scattering

| Property | Value |
|-|-|
| **File** | `tensornet/qm/scattering.py` |
| **Primary class** | `PartialWaveScattering` |
| **Constructor** | `PartialWaveScattering(k: float = 1.0, l_max: int = 10)` |
| **Key methods** | (partial wave phase shift computation) |
| **Alternate classes** | `BornApproximation(mass, hbar)`, `RMatrixScattering(a_boundary, n_basis)`, `BreitWignerResonance(Er, Gamma, ...)` |
| **Adapter type** | **Eigenvalue** (solves radial Schrödinger + phase shifts) |

### 73. Semiclassical

| Property | Value |
|-|-|
| **File** | `tensornet/qm/semiclassical_wkb.py` |
| **Primary class** | `WKBSolver` |
| **Constructor** | `WKBSolver(V: Callable[[float], float], mass: float = 1.0, hbar: float = 1.0)` |
| **Key methods** | (WKB tunnelling, quantisation) |
| **Alternate classes** | `TullySurfaceHopping(n_states, mass)` — `run_trajectory(x0, p0, ...)`, `HermanKlukPropagator(V, mass)` — `propagate(psi0, x_grid, ...)` |
| **Adapter type** | **Time-step** (trajectory propagation / semiclassical dynamics) |

### 74. Path Integrals

| Property | Value |
|-|-|
| **File** | `tensornet/quantum_mechanics/path_integrals.py` |
| **Primary class** | `PIMC` |
| **Constructor** | `PIMC(n_beads: int, temperature: float, mass: float = 1.0, hbar: float = 1.0)` |
| **Key methods** | `run(potential, n_steps, n_equil) -> dict` |
| **Alternate classes** | `RPMD(n_beads, temperature, mass)` — `run(potential_force, ...)`, `InstantonSolver(mass)`, `ThermodynamicIntegration(n_lambda)` |
| **Adapter type** | **SCF-type** (Monte Carlo convergence / iterative sampling) |

---

## VII. Quantum Many-Body (13 domains)

### 75. Tensor Network / DMRG

| Property | Value |
|-|-|
| **File** | `tensornet/algorithms/dmrg.py` + `tensornet/core/mps.py` + `tensornet/mps/hamiltonians.py` |
| **Primary function** | `dmrg(H: MPO, chi_max: int, num_sweeps: int = 10, tol: float = 1e-10, psi0: MPS | None = None, svd_cutoff: float = 1e-14, verbose: bool = False) -> DMRGResult` |
| **Primary class** | `MPS` (in `core/mps.py`) |
| **MPS Constructor** | `MPS(tensors: list[Tensor])` |
| **MPS Key methods** | `random(L, d, chi, ...)`, `norm()`, `canonicalize_left_()`, `canonicalize_right_()`, `entropy(bond)`, `expectation_local(op, site)`, `truncate_(chi_max)` |
| **DMRGResult fields** | `psi: MPS, energy: float, energies: list[float], entropies: list[float], truncation_errors: list[float], converged: bool, sweeps: int` |
| **Hamiltonians** | `heisenberg_mpo()`, `tfim_mpo()`, `xx_mpo()`, `xyz_mpo()`, `bose_hubbard_mpo()` in `mps/hamiltonians.py` |
| **Pack class** | `HeisenbergSolver(chi=16, tau=0.05, n_steps=200)` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (DMRG variational sweep) |

### 76. Quantum Spin

| Property | Value |
|-|-|
| **File** | `tensornet/mps/hamiltonians.py` (MPO constructors) + `tensornet/packs/pack_vii.py` (solver) |
| **Primary functions** | `heisenberg_mpo(L, J, Jz, h, ...)`, `tfim_mpo(L, J, h, ...)`, `xx_mpo(L, J, h, ...)`, `xyz_mpo(L, Jx, Jy, Jz, ...)` |
| **Helper** | `spin_operators(S)`, `pauli_matrices()` |
| **Pack class** | `SpinSystemsSpec` / solved via `HeisenbergSolver` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (DMRG on spin Hamiltonians) |

### 77. Strongly Correlated

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/strongly_correlated.py` |
| **Primary class** | `DMFTSolver` |
| **Constructor** | `DMFTSolver(U: float, mu: float, D: float = 1.0, beta: float = 10.0, n_iwn: int = 256)` |
| **Key methods** | `solve(max_iter, tol, ...) -> dict` |
| **Alternate classes** | `HirschFyeQMC(n_time_slices, ...)` — `solve_impurity(G0_iwn, ...)`, `tJModelMPO(L, t, J, ...)`, `MottTransition` |
| **Pack class** | `CorrelatedElectronsSolver` in `packs/pack_vii.py` |
| **Adapter type** | **SCF-type** (DMFT self-consistency loop) |

### 78. Topological

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/topological_phases.py` |
| **Primary class** | `ToricCode` |
| **Constructor** | `ToricCode(L: int = 4, Js: float = 1.0, Jp: float = 1.0)` |
| **Key methods** | `ground_state_degeneracy() -> int` |
| **Alternate classes** | `ChernNumberCalculator(nk)`, `TopologicalEntanglementEntropy`, `AnyonicBraiding` |
| **Pack class** | `TopologicalPhasesSolver` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (topological invariant computation) |

### 79. MBL & Disorder

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/mbl_disorder.py` |
| **Primary class** | `RandomFieldXXZ` |
| **Constructor** | `RandomFieldXXZ(L: int = 10, J: float = 1.0, Delta: float = 1.0, W: float = ...)` |
| **Key methods** | (exact diagonalisation of disordered chain) |
| **Alternate classes** | `LevelStatistics(energies)`, `ParticipationRatio(eigenstates)`, `EntanglementDynamics(L, H)` — `evolve_and_measure(psi0, times)` |
| **Pack class** | `MBLocalizationSolver` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (full diagonalisation + level statistics) |

### 80. Lattice Gauge

| Property | Value |
|-|-|
| **File** | `tensornet/qft/lattice_qft.py` (pure gauge + HMC) + `tensornet/qft/lattice_qcd.py` (Wilson action) |
| **Primary class** | `HMCSampler` (in `lattice_qft.py`, dataclass) |
| **Constructor** | `HMCSampler(config: LatticeConfig, n_steps: int = 10, step_size: float = 0.1)` |
| **Key methods** | `trajectory(gf: GaugeField) -> Tuple[GaugeField, bool]` |
| **Supporting classes** | `GaugeField`, `ScalarField`, `WilsonFermionOperator`, `LatticeConfig`, `FieldType` |
| **Pack class** | `LatticeGaugeSolver` in `packs/pack_vii.py` |
| **Adapter type** | **SCF-type** (HMC Monte Carlo sampling with accept/reject) |

### 81. Open Quantum

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/open_quantum.py` |
| **Primary class** | `LindbladSolver` |
| **Constructor** | `LindbladSolver(H: NDArray[np.complex128], L_ops: list[NDArray[np.complex128]], ...)` |
| **Key methods** | `evolve(rho_0, t_final, dt) -> dict` |
| **Alternate classes** | `QuantumTrajectories(H, L_ops, ...)` — `run(psi_0, ...)`, `RedfieldEquation(H_S, ...)` — `evolve(rho_0, ...)`, `SteadyStateSolver` |
| **Pack class** | `OpenQuantumSolver` in `packs/pack_vii.py` |
| **Adapter type** | **Time-step** (Lindblad master equation ODE) |

### 82. Non-Equilibrium QM

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/nonequilibrium_qm.py` |
| **Primary class** | `FloquetSolver` |
| **Constructor** | `FloquetSolver(dim: int)` |
| **Key methods** | (Floquet quasi-energy spectrum computation) |
| **Alternate classes** | `ETHDiagnostics`, `LiebRobinsonBound(H, ...)`, `PrethermalisationAnalyser` |
| **Pack class** | `NonEqQuantumSolver` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (Floquet diagonalisation) / **Time-step** (quench dynamics) |

### 83. Kondo/Impurity

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/kondo_impurity.py` |
| **Primary class** | `WilsonChainNRG` |
| **Constructor** | `WilsonChainNRG(Lambda: float = 2.0, n_sites: int = 30, ...)` |
| **Alternate classes** | `AndersonImpurityModel(eps_d, U, ...)`, `CTQMC_HybridisationExpansion(beta, eps_d, ...)` — `run_sampling(n_mc, n_warmup)`, `KondoTemperatureExtractor` |
| **Pack class** | `QuantumImpuritySolver` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (NRG iterative diagonalisation) / **SCF-type** (CT-QMC sampling) |

### 84. Bosonic

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/bosonic.py` |
| **Primary class** | `GrossPitaevskiiSolver` |
| **Constructor** | `GrossPitaevskiiSolver(N_grid: int, x_max: float, mass: float = 1.0)` |
| **Key methods** | `ground_state(V_ext, g, ...) -> GPEResult`, `propagate(psi0, V_ext, g, dt, n_steps)` |
| **Alternate classes** | `BogoliubovTheory(n0, g, mass)`, `TonksGirardeauGas(N_particles, L)` — `ground_state_energy()`, `BoseHubbardPhase(n_max, z)` |
| **Pack class** | `BosonicMBSolver` in `packs/pack_vii.py` |
| **Adapter type** | **SCF-type** (imaginary-time propagation for ground state) / **Time-step** (real-time GPE) |

### 85. Fermionic

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/fermionic.py` |
| **Primary class** | `BCSSolver` |
| **Constructor** | `BCSSolver(N_k: int = 500, E_cutoff: float = 20.0)` |
| **Key methods** | `solve_swave(epsilon_k, V0) -> BCSResult`, `solve_dwave(kx, ky, V0)` |
| **Alternate classes** | `FFLOSolver(N_k)`, `BravyiKitaevTransform(n_modes)`, `FermiLiquidLandau(k_F, m_bare, dim)` |
| **Pack class** | `FermionicSolver` in `packs/pack_vii.py` |
| **Adapter type** | **SCF-type** (BCS gap equation self-consistency) |

### 86. Nuclear Many-Body

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/nuclear_many_body.py` |
| **Primary class** | `NuclearShellModel` |
| **Constructor** | `NuclearShellModel(n_orbits: int = 4, n_particles: int = 2)` |
| **Key methods** | `ground_state_energy() -> float` |
| **Alternate classes** | `RichardsonGaudinPairing(levels, degeneracies)` — `solve_richardson(n_pairs, ...)`, `ChiralEFTInteraction(Lambda_UV)`, `BetheWeizsacker` |
| **Pack class** | `NuclearMBSolver` in `packs/pack_vii.py` |
| **Adapter type** | **Eigenvalue** (shell model diagonalisation) |

### 87. Ultracold Atoms

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/ultracold_atoms.py` |
| **Primary class** | `BoseHubbardModel` |
| **Constructor** | `BoseHubbardModel(L: int = 8, n_max: int = 3, ...)` |
| **Alternate classes** | `BECBCSCrossover(kF, mass)` — `solve_gap_equation(inv_kFas, nk)`, `FeshbachResonance`, `GrossPitaevskiiSolver(nx, Lx)` |
| **Pack class** | `UltracoldSolver` in `packs/pack_vii.py` |
| **Adapter type** | **SCF-type** (gap equation convergence) / **Eigenvalue** (Bose-Hubbard ED) |

---

## VIII. Electronic Structure (7 domains)

### 88. DFT

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/dft.py` |
| **Primary class** | `KohnShamDFT1D` |
| **Constructor** | `KohnShamDFT1D(ngrid: int = 200, L: float = 20.0, ...)` |
| **Key methods** | `scf(max_iter, tol, ...) -> dict` |
| **Supporting classes** | `LDAExchangeCorrelation`, `PBEExchangeCorrelation`, `AndersonMixer(n_history, beta)`, `NormConservingPseudopotential(Z, r_cutoff)` |
| **Pack class** | `KohnShamSolver` in `packs/pack_viii.py` |
| **Adapter type** | **SCF-type** (Kohn-Sham self-consistent field) |

### 89. Beyond-DFT

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/beyond_dft.py` |
| **Primary class** | `RestrictedHartreeFock` |
| **Constructor** | `RestrictedHartreeFock(n_basis: int = 10, n_electrons: int = 2)` |
| **Key methods** | `scf(max_iter, tol) -> Dict[str, float]` |
| **Alternate classes** | `MP2Correlation(C, eigenvalues, eri, n_occ)`, `CCSDSolver(n_occ, n_virt, fock_diag, eri)` — `solve(max_iter, tol) -> float`, `CASSCFSolver(n_active_el, n_active_orb)` — `solve_ci(h1e, h2e) -> float` |
| **Adapter type** | **SCF-type** (Hartree-Fock SCF + post-HF iteration) |

### 90. Tight Binding

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/tight_binding.py` |
| **Primary class** | `SlaterKosterTB` |
| **Constructor** | `SlaterKosterTB(n_atoms: int = 2, ...)` |
| **Alternate classes** | `SCCDFTB(n_atoms, hubbard_U)` — `scf(positions, n_electrons, ...)`, `ExtendedHuckel()` — `solve(H_diag, S)` |
| **Adapter type** | **Eigenvalue** (band diagonalisation) / **SCF-type** (SCC-DFTB self-consistency) |

### 91. Excited States

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/excited_states.py` |
| **Primary class** | `CasidaTDDFT` |
| **Constructor** | `CasidaTDDFT(eigenvalues: NDArray, n_occ: int, ...)` |
| **Alternate classes** | `RealTimeTDDFT(H0, n_occ, dt)` — `run(n_steps, r_matrix, ...)`, `GWApproximation(eigenvalues, n_occ)`, `BetheSalpeterEquation(eps_qp, n_occ)` |
| **Adapter type** | **Eigenvalue** (Casida equation) / **Time-step** (real-time TDDFT) |

### 92. Response Properties

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/response.py` |
| **Primary class** | `DFPTSolver` |
| **Constructor** | `DFPTSolver(H0: NDArray, psi0: NDArray, eigenvalues: NDArray, ...)` |
| **Key methods** | (density-functional perturbation theory response) |
| **Alternate classes** | `Polarisability(eigenvalues, transition_dipoles)`, `DielectricFunction(n_electrons, volume)`, `BornEffectiveCharge(n_atoms)` — `compute_from_forces(...)` |
| **Adapter type** | **SCF-type** (Sternheimer equation iteration) |

### 93. Relativistic Electronic

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/relativistic.py` |
| **Primary class** | `ZORAHamiltonian` |
| **Constructor** | `ZORAHamiltonian(V: NDArray, grid: NDArray)` |
| **Key methods** | `solve() -> Tuple[NDArray, NDArray]` |
| **Alternate classes** | `SpinOrbitCoupling(V, grid)`, `DouglasKrollHess(c)`, `Dirac4Component(Z)` |
| **Adapter type** | **Eigenvalue** (relativistic Hamiltonian diagonalisation) |

### 94. Quantum Embedding

| Property | Value |
|-|-|
| **File** | `tensornet/electronic_structure/embedding.py` |
| **Primary class** | `QMMMEmbedding` |
| **Constructor** | `QMMMEmbedding(qm_atoms: NDArray, mm_atoms: NDArray, ...)` |
| **Alternate classes** | `ONIOMEmbedding()`, `DFTPlusDMFT(n_correlated, ...)`, `ProjectionEmbedding(n_basis_A, n_basis_B)` — `solve_embedded(H_A, V_emb, ...)` |
| **Adapter type** | **SCF-type** (embedding self-consistency / DFT+DMFT loop) |

---

## IX. Solid State / Condensed Matter (8 domains)

### 95. Phonons

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/phonons.py` |
| **Primary class** | `DynamicalMatrix` |
| **Constructor** | `DynamicalMatrix(masses: NDArray[np.float64], ...)` |
| **Key methods** | (computes phonon bands via dynamical matrix diagonalisation) |
| **Alternate classes** | `PhononDOS`, `AnharmonicPhonon(gruneisen, debye_T)`, `PhononBTE(volume)` |
| **Pack class** | `PhononSolver` in `packs/pack_vi.py` |
| **Adapter type** | **Eigenvalue** (dynamical matrix diagonalisation) |

### 96. Band Structure

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/band_structure.py` |
| **Primary class** | `TightBindingBands` |
| **Constructor** | `TightBindingBands(dim: int = 1, n_orbitals: int = 1)` |
| **Alternate classes** | `KdotPMethod(Eg, Ep)`, `DensityOfStates`, `WannierProjection(n_bands, n_k)` |
| **Pack class** | `BandStructureSolver` in `packs/pack_vi.py` |
| **Adapter type** | **Eigenvalue** (Bloch Hamiltonian diagonalisation at k-points) |

### 97. Classical Magnetism

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/classical_magnetism.py` |
| **Primary class** | `LandauLifshitzGilbert` |
| **Constructor** | `LandauLifshitzGilbert(alpha: float = 0.01, Ms: float = 8e5, ...)` |
| **Alternate classes** | `StonerWohlfarth(K_u, Ms, ...)`, `DomainWall(A, K_u, ...)`, `HeisenbergModel2D(L, J, D, ...)` |
| **Pack class** | `MagnetismSolver` in `packs/pack_vi.py` |
| **Adapter type** | **Time-step** (LLG ODE integration) |

### 98. Superconductivity

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/fermionic.py` (BCSSolver, FFLOSolver) |
| **Primary class** | `BCSSolver` |
| **Constructor** | `BCSSolver(N_k: int = 500, E_cutoff: float = 20.0)` |
| **Key methods** | `solve_swave(epsilon_k, V0) -> BCSResult`, `solve_dwave(kx, ky, V0)` |
| **Pack class** | `SuperconductivitySolver` in `packs/pack_vi.py` |
| **Note** | No dedicated `superconductivity.py` — uses fermionic.py `BCSSolver` + `FFLOSolver` |
| **Adapter type** | **SCF-type** (BCS gap equation self-consistency) |

### 99. Disordered Systems

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/disordered.py` |
| **Primary class** | `AndersonModel` |
| **Constructor** | `AndersonModel(L: int, dim: int = 1, t: float = 1.0, W: float = ...)` |
| **Alternate classes** | `KPMSpectral(H, n_chebyshev)`, `EdwardsAndersonSpinGlass(L, dim, J_std, ...)`, `LocalisationMetrics` |
| **Pack class** | `DisorderedSystemsSolver` in `packs/pack_vi.py` |
| **Adapter type** | **Eigenvalue** (Anderson localisation via ED) / **SCF-type** (KPM Chebyshev expansion) |

### 100. Surfaces & Interfaces

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/surfaces_interfaces.py` |
| **Primary class** | `SurfaceEnergy` |
| **Constructor** | `SurfaceEnergy()` |
| **Alternate classes** | `AdsorptionIsotherms`, `SchottkyBarrier(phi_m, chi_s, ...)`, `HeterostructureBandAlignment(material1, material2)` |
| **Pack class** | `SurfacePhysicsSolver` in `packs/pack_vi.py` |
| **Adapter type** | **SCF-type** (surface energy minimisation / Poisson-Boltzmann) |

### 101. Defects

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/defects.py` |
| **Primary class** | `PointDefectCalculator` |
| **Constructor** | `PointDefectCalculator(positions: NDArray[np.float64], ...)` |
| **Alternate classes** | `PeierlsNabarroModel(b, d, mu, nu)`, `GrainBoundaryEnergy(b, mu, nu, ...)` |
| **Adapter type** | **SCF-type** (defect formation energy convergence) |

### 102. Ferroelectrics

| Property | Value |
|-|-|
| **File** | `tensornet/condensed_matter/ferroelectrics.py` |
| **Primary class** | `LandauDevonshire` |
| **Constructor** | `LandauDevonshire(alpha_0: float = 3.8e5, Tc: float = 393.0, ...)` |
| **Alternate classes** | `PiezoelectricCoupling(d33, eps_33, ...)`, `DomainSwitching(Ps, t0, ...)`, `PyroelectricEffect(p, eps_r, ...)` |
| **Adapter type** | **SCF-type** (Landau free energy minimisation) / **Time-step** (domain switching dynamics) |

---

## X. Nuclear & Particle (6 domains)

### 103. Nuclear Structure

| Property | Value |
|-|-|
| **File** | `tensornet/nuclear/structure.py` |
| **Primary class** | `NuclearShellModel` |
| **Constructor** | `NuclearShellModel(A: int = 16, Z: int = 8, ...)` |
| **Alternate classes** | `HartreeFockBogoliubov(n_levels)` — `solve(n_particles, G, ...)`, `NuclearDFT(n_grid, r_max)`, `IMSRG(n_sp, n_occ)` — `solve(...)`, `CoupledClusterSD(n_occ, n_unocc)`, `NCSM(...)` |
| **Pack class** | `NuclearStructureSolver` in `packs/pack_ix.py` |
| **Adapter type** | **Eigenvalue** (shell model diag / IMSRG flow / NCSM) |

### 104. Nuclear Reactions

| Property | Value |
|-|-|
| **File** | `tensornet/nuclear/reactions.py` |
| **Primary class** | `OpticalModelPotential` |
| **Constructor** | `OpticalModelPotential(A_target: int = 40, Z_target: int = 20, ...)` |
| **Alternate classes** | `RMatrixSolver(channel_radius)`, `HauserFeshbach(A_compound, E_excitation)`, `DWBATransfer(mass_projectile, ...)` |
| **Pack class** | `NuclearReactionsSolver` in `packs/pack_ix.py` |
| **Adapter type** | **Eigenvalue** (R-matrix levels) / **SCF-type** (optical model parameter fitting) |

### 105. Nuclear Astrophysics

| Property | Value |
|-|-|
| **File** | `tensornet/nuclear/astrophysics.py` |
| **Primary class** | `ThermonuclearRate` |
| **Constructor** | `ThermonuclearRate(Z1: int = 1, Z2: int = 1, ...)` |
| **Alternate classes** | `NuclearReactionNetwork()`, `RProcess(T9, n_n)`, `SProcess(n_n_vt)` |
| **Pack class** | `NucleosynthesisSolver` in `packs/pack_ix.py` |
| **Adapter type** | **Time-step** (reaction network ODE integration) |

### 106. Lattice QCD

| Property | Value |
|-|-|
| **File** | `tensornet/qft/lattice_qcd.py` |
| **Primary class** | `WilsonGaugeAction` |
| **Constructor** | `WilsonGaugeAction(L: int, dim: int = 4, beta: float = 6.0, seed: Optional[int] = None)` |
| **Key methods** | (plaquette, heatbath, average plaquette) |
| **Alternate classes** | `SU3Group`, `WilsonFermion`, `CreutzRatio`, `HadronCorrelator`, `DynamicalHMC(gauge, kappa, n_steps, step_size)` — `trajectory(use_fermion_force) -> Tuple[bool, float]` |
| **Pack class** | `LatticeQCDSolver` in `packs/pack_x.py` |
| **Adapter type** | **SCF-type** (Monte Carlo / HMC thermalisation convergence) |

### 107. Perturbative QFT

| Property | Value |
|-|-|
| **File** | `tensornet/qft/perturbative.py` |
| **Primary class** | `FeynmanDiagram` |
| **Constructor** | `FeynmanDiagram(...)` |
| **Key methods** | `evaluate_scalar_bubble(p_sq, m1, m2, ...)` |
| **Alternate classes** | `DimensionalRegularisation(d, mu)`, `MSBarRenormalisation(mu)`, `RunningCoupling(theory, n_f, ...)` — `run_qcd_with_thresholds(mu, ...)`, `SplittingFunctions`, `SudakovFormFactor(t_cut)`, `PartonShower(...)` |
| **Pack class** | `PartonSolver` in `packs/pack_x.py` |
| **Adapter type** | **SCF-type** (RG running / iterative renormalisation) |

### 108. Beyond SM

| Property | Value |
|-|-|
| **File** | `tensornet/particle/beyond_sm.py` |
| **Primary class** | `NeutrinoOscillations` |
| **Constructor** | `NeutrinoOscillations(theta12: float = 33.44, theta23: float = 49.2, ...)` |
| **Alternate classes** | `DarkMatterRelic(mass_chi, ...)`, `GUTRunningCouplings(model)` — `running_couplings(mu_range)`, `SMEFTOperators(Lambda)` — `run_coefficients(mu_from, mu_to, ...)` |
| **Pack class** | `BSMSolver` in `packs/pack_x.py` |
| **Adapter type** | **SCF-type** (RG running / relic density integration) |

---

## XV. Chemical Physics (3 remaining domains)

### 109. PES (Potential Energy Surface)

| Property | Value |
|-|-|
| **File** | `tensornet/chemistry/pes.py` |
| **Primary class** | `NudgedElasticBand` |
| **Constructor** | `NudgedElasticBand(energy_func: Callable, grad_func: Callable, ...)` |
| **Key methods** | (NEB path optimisation, saddle point finding) |
| **Alternate classes** | `MorsePotential(D_e, alpha, r_e, ...)`, `LEPSPotential(D_AB, D_BC, ...)`, `IntrinsicReactionCoordinate(energy_func, grad_func, ...)` |
| **Adapter type** | **SCF-type** (NEB image optimisation convergence) |

### 110. Reaction Rate (TST)

| Property | Value |
|-|-|
| **File** | `tensornet/chemistry/reaction_rate.py` |
| **Primary class** | `TransitionStateTheory` |
| **Constructor** | `TransitionStateTheory(E_a: float, frequencies_reactant: List[float], ...)` |
| **Alternate classes** | `VariationalTST(energy_profile, ...)`, `RRKMTheory(E0, freq_reactant, ...)`, `KramersRate(omega_0, omega_b, ...)` |
| **Adapter type** | **SCF-type** (variational dividing surface optimisation) |

### 111. Catalysis

| Property | Value |
|-|-|
| **File** | **NOT FOUND** in `tensornet/chemistry/` |
| **Closest** | `tensornet/fusion/resonant_catalysis.py` — `ResonantCatalysisSolver` (fusion catalysis, domain-specific) |
| **Pack class** | `SurfaceChemistrySolver` in `packs/pack_xv.py` (covers heterogeneous catalysis via surface chemistry) |
| **Note** | No dedicated catalysis solver exists in chemistry — need to create `tensornet/chemistry/catalysis.py` or rely on `SurfaceChemistrySolver` from pack_xv |
| **Adapter type** | **SCF-type** (microkinetic model convergence / steady-state solution) |

---

## XIX. Quantum Information (3 domains)

### 112. Quantum Circuit

| Property | Value |
|-|-|
| **File** | `tensornet/quantum/hybrid.py` |
| **Primary class** | `QuantumCircuit` |
| **Constructor** | `QuantumCircuit(n_qubits: int, chi_max: int = 64)` |
| **Key methods** | `add_gate(gate: QuantumGate)`, (gate simulation via tensor network contraction) |
| **Pack class** | `QuantumCircuitsSolver` in `packs/pack_xix.py` |
| **Adapter type** | **Time-step** (sequential gate application / circuit depth) |

### 113. QEC (Quantum Error Correction)

| Property | Value |
|-|-|
| **File** | `tensornet/quantum/error_mitigation.py` |
| **Primary class** | `QECCode` (ABC) |
| **Concrete classes** | `BitFlipCode`, `PhaseFlipCode`, `ShorCode` |
| **Key methods** | `encode(logical_state) -> Tensor`, `decode(physical_state) -> Tensor`, `syndrome_measure(state) -> Tensor`, `correct_error(state, syndrome) -> Tensor` |
| **Pack class** | `QuantumErrorCorrectionSolver` in `packs/pack_xix.py` |
| **Adapter type** | **Eigenvalue** (syndrome decoding / code distance computation) |

### 114. Quantum Algorithms (VQE)

| Property | Value |
|-|-|
| **File** | `tensornet/quantum/hybrid.py` |
| **Primary class** | `VQE` |
| **Constructor** | `VQE(hamiltonian: Callable, n_qubits: int, config: VQEConfig | None = None)` |
| **Key methods** | `optimize(verbose: bool = True) -> dict` |
| **VQEConfig fields** | ansatz depth, optimizer, tolerance |
| **Alternate class** | `QAOA(...)` — `run_circuit(gammas, betas) -> float`, `optimize() -> dict` |
| **Pack class** | `QuantumAlgorithmsSolver` in `packs/pack_xix.py` |
| **Adapter type** | **SCF-type** (VQE variational parameter optimisation convergence) |

---

## Summary: Adapter Type Distribution

| Adapter Type | Count | Domains |
|-|-|-|
| **SCF-type** (iterative convergence) | **19** | 74, 77, 80, 84, 85, 87, 88, 89, 92, 94, 98, 99, 100, 101, 102, 106, 107, 108, 109, 110, 111, 114 |
| **Eigenvalue** (diag/Lanczos/DMRG) | **16** | 70, 72, 75, 76, 78, 79, 82, 83, 86, 90, 91, 93, 95, 96, 99, 103, 104, 113 |
| **Time-step** (ODE/PDE propagation) | **10** | 71, 73, 81, 82, 84, 91, 97, 102, 105, 112 |

> Note: Several domains are hybrid — they have both Eigenvalue and Time-step or SCF-type sub-solvers. The primary adapter type is listed first.

---

## Missing / Needs Creation

| Domain | ID | Status | Action Needed |
|-|-|-|-|
| Catalysis | 111 | **NO dedicated solver** | Create `tensornet/chemistry/catalysis.py` OR bridge `SurfaceChemistrySolver` from `packs/pack_xv.py` |
| Superconductivity | 98 | No standalone file; uses `fermionic.py` `BCSSolver`/`FFLOSolver` | OK — adapter wraps existing `BCSSolver`. Pack has `SuperconductivitySolver`. |
| Lattice Gauge | 80 | Uses `qft/lattice_qft.py` `HMCSampler` | OK — adapter wraps `HMCSampler` + `GaugeField` |

### New trace_adapters/ directories to create (10)

```
tensornet/qm/trace_adapters/
tensornet/quantum_mechanics/trace_adapters/
tensornet/condensed_matter/trace_adapters/
tensornet/electronic_structure/trace_adapters/
tensornet/nuclear/trace_adapters/
tensornet/qft/trace_adapters/
tensornet/particle/trace_adapters/
tensornet/quantum/trace_adapters/
tensornet/mps/trace_adapters/          (or tensornet/algorithms/trace_adapters/)
tensornet/chemistry/trace_adapters/    (ALREADY EXISTS — skip)
```

**Net new directories: 9** (chemistry already has one).
