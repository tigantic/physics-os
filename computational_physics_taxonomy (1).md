# Computational Physics — Complete Domain Taxonomy
## The Full Capability Map

Every computationally relevant branch of physics, organized hierarchically.
Each entry includes: key equations, standard computational methods, and QTT relevance.

*Purpose: Measure existing Tigantic/HyperTensor coverage against the complete field.*

---

## I. CLASSICAL MECHANICS

### I.1 Newtonian Particle Dynamics
- N-body gravitational: F = -GmM/r² (direct summation, Barnes-Hut, FMM)
- Rigid body: Euler equations, quaternion integration, constraint solvers
- Contact/collision: penalty methods, impulse-based, Signorini conditions
- **Computational methods**: Velocity Verlet, symplectic integrators, Runge-Kutta
- **QTT angle**: Phase space compression for large N

### I.2 Lagrangian / Hamiltonian Mechanics
- Euler-Lagrange equations, Hamilton's equations
- Symplectic integrators (Störmer-Verlet, leapfrog, Ruth, Yoshida)
- Action minimization, variational integrators
- Noether's theorem → conservation law verification
- **Computational methods**: Symplectic maps, generating functions

### I.3 Continuum Mechanics
- Linear elasticity: σ = Cε (Hooke's law, Cauchy stress tensor)
- Nonlinear elasticity: hyperelastic (Neo-Hookean, Mooney-Rivlin, Ogden)
- Viscoelasticity: Maxwell, Kelvin-Voigt, generalized models
- Plasticity: von Mises yield, Drucker-Prager, crystal plasticity
- Fracture mechanics: LEFM (K_I, K_II, K_III), cohesive zone, XFEM, phase-field fracture
- Contact mechanics: Hertz, JKR adhesion, friction (Coulomb, Amontons)
- Large deformation: Updated/Total Lagrangian, ALE
- **Computational methods**: FEM, BEM, meshfree (SPH, RKPM), peridynamics
- **QTT angle**: Stiffness matrices, displacement fields as tensor trains

### I.4 Structural Mechanics
- Beam theory: Euler-Bernoulli, Timoshenko
- Plate/shell theory: Kirchhoff, Mindlin-Reissner, Koiter
- Buckling: eigenvalue stability analysis
- Vibration: modal analysis, frequency response
- Composite laminates: Classical Laminate Theory, failure criteria (Tsai-Wu, Hashin)
- **Computational methods**: FEA (h/p/hp refinement), isogeometric analysis
- **QTT angle**: Mode shapes, stiffness tensors

### I.5 Nonlinear Dynamics & Chaos
- Lyapunov exponents, strange attractors, bifurcation theory
- KAM theory (Kolmogorov-Arnold-Moser)
- Poincaré maps, return maps
- Hamiltonian chaos, Arnold diffusion
- **Computational methods**: Long-time integration, variational equations

### I.6 Acoustics & Vibration
- Wave equation: ∂²p/∂t² = c²∇²p
- Helmholtz equation: ∇²p + k²p = 0 (frequency domain)
- Acoustic scattering, diffraction
- Structural-acoustic coupling (vibroacoustics)
- Room acoustics, ray tracing, image source method
- Underwater acoustics, sonar
- **Computational methods**: BEM, FEM, FDTD, spectral methods
- **QTT angle**: Green's functions, transfer matrices as TT

---

## II. FLUID DYNAMICS

### II.1 Incompressible Navier-Stokes
- ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u, ∇·u = 0
- Stokes flow (Re << 1), creeping flow
- Pressure-velocity coupling: SIMPLE, PISO, fractional step
- **Computational methods**: FVM, FEM, spectral, lattice Boltzmann
- **QTT angle**: Velocity/pressure fields as QTT

### II.2 Compressible Flow
- Euler equations (inviscid): conservation of mass, momentum, energy
- Compressible Navier-Stokes
- Shock capturing: Godunov, Roe, HLLC, WENO schemes
- Detonation/deflagration waves
- **Computational methods**: FVM with Riemann solvers, discontinuous Galerkin

### II.3 Turbulence
- Direct Numerical Simulation (DNS)
- Large Eddy Simulation (LES): Smagorinsky, dynamic, WALE subgrid models
- Reynolds-Averaged (RANS): k-ε, k-ω, SST, Reynolds stress
- Detached Eddy Simulation (DES), hybrid RANS-LES
- Kolmogorov cascade, energy spectrum E(k) ~ k^(-5/3)
- **Computational methods**: Spectral, high-order FEM/DG, adaptive mesh
- **QTT angle**: Your arXiv paper — QTT compression of turbulent fields

### II.4 Multiphase Flow
- Volume of Fluid (VOF), Level Set, Phase Field
- Euler-Euler (two-fluid), Euler-Lagrange (particle tracking)
- Rayleigh-Taylor, Kelvin-Helmholtz instabilities
- Droplet dynamics, coalescence, breakup
- Cavitation
- **Computational methods**: Interface capturing/tracking, coupled solvers

### II.5 Reactive Flow / Combustion
- Species transport: ∂Y_k/∂t + u·∇Y_k = ∇·(D∇Y_k) + ω̇_k
- Chemical kinetics: Arrhenius, detailed mechanisms (GRI-Mech)
- Flame structure: premixed, non-premixed, partially premixed
- Detonation: Chapman-Jouguet, ZND model
- **Computational methods**: Stiff ODE solvers, flamelet models, CMC

### II.6 Rarefied Gas / Kinetic Theory
- Boltzmann equation: ∂f/∂t + v·∇f = Q(f,f) (collision integral)
- BGK approximation
- DSMC (Direct Simulation Monte Carlo)
- Knudsen number regimes
- **Computational methods**: DSMC, discrete velocity methods, moment methods
- **QTT angle**: Distribution function f(x,v,t) is high-dimensional — natural QTT target

### II.7 Shallow Water / Geophysical Fluid Dynamics
- Shallow water equations: ∂h/∂t + ∇·(hu) = 0
- Rotating frames: Coriolis, Rossby waves
- Quasi-geostrophic equations
- Boussinesq approximation
- Ocean/atmosphere coupling
- **Computational methods**: Finite volume, spectral transform

### II.8 Non-Newtonian / Complex Fluids
- Viscoelastic: Oldroyd-B, FENE-P, Giesekus models
- Bingham plastic, Herschel-Bulkley
- Polymer solutions, suspensions
- Blood flow (shear-thinning)
- **Computational methods**: Log-conformation, stabilized FEM

### II.9 Porous Media Flow
- Darcy's law: u = -(k/μ)∇p
- Richards equation (unsaturated)
- Brinkman equation (intermediate porosity)
- Multiphase in porous media: Buckley-Leverett
- **Computational methods**: FVM, mixed FEM, multiscale

### II.10 Free Surface / Interfacial Flows
- Surface tension: Laplace pressure Δp = γ(1/R₁ + 1/R₂)
- Marangoni flow (surface tension gradients)
- Capillary phenomena, wetting, contact angle
- Thin film equations
- **Computational methods**: ALE, level set, phase field

---

## III. ELECTROMAGNETISM

### III.1 Electrostatics
- Poisson equation: ∇²φ = -ρ/ε₀
- Laplace equation (charge-free regions)
- Capacitance computation, charge distributions
- **Computational methods**: FEM, BEM, FDM, multipole expansion

### III.2 Magnetostatics
- Biot-Savart law, vector potential A
- Magnetic circuits, inductance computation
- Permanent magnets, demagnetization
- **Computational methods**: FEM, BEM

### III.3 Full Maxwell (Time-Domain)
- ∂B/∂t = -∇×E, ∂D/∂t = ∇×H - J
- FDTD (Yee lattice), PML absorbing boundaries
- CEM: radar cross section, antenna radiation patterns
- **Computational methods**: FDTD, FETD, DGTD, MoM
- **QTT angle**: Your CEM-QTT module

### III.4 Frequency-Domain EM
- Helmholtz equation: ∇²E + k²E = 0
- Waveguide modes, cavity resonances
- Scattering: Mie theory, T-matrix
- **Computational methods**: FEM, BEM, MoM

### III.5 Electromagnetic Wave Propagation
- Ray optics / geometric optics (eikonal equation)
- Beam propagation method
- Atmospheric propagation, ionospheric effects
- Fiber optics, guided waves
- **Computational methods**: Ray tracing, parabolic equation, split-step Fourier

### III.6 Computational Photonics
- Photonic crystals: band structure, bandgap
- Plasmonics: surface plasmon resonance, LSPR
- Metamaterials: effective medium theory, negative index
- Nonlinear optics: χ², χ³, four-wave mixing, Kerr effect
- Nanophotonics, near-field optics
- **Computational methods**: FDTD, RCWA, plane wave expansion, FDFD

### III.7 Antenna & Microwave Engineering
- Impedance matching, Smith chart
- Array factor, beamforming
- Mutual coupling
- Microstrip, patch, horn, phased array
- **Computational methods**: MoM, FDTD, FEM, hybrid methods

---

## IV. OPTICS & PHOTONICS

### IV.1 Physical Optics
- Diffraction: Fresnel, Fraunhofer, scalar diffraction theory
- Interference, coherence theory
- Polarization: Jones calculus, Mueller matrices, Stokes parameters
- **Computational methods**: Angular spectrum, Fourier optics

### IV.2 Quantum Optics
- Jaynes-Cummings model: H = ℏω_c a†a + ℏω_a σ_z/2 + ℏg(a†σ₋ + aσ₊)
- Cavity QED, photon statistics
- Squeezed states, entangled photons
- Master equation for open quantum optical systems
- Photon blockade, Kerr nonlinearity
- **Computational methods**: QuTiP-style master equation, quantum trajectories
- **QTT angle**: Fock space truncation → MPS

### IV.3 Laser Physics
- Rate equations, gain saturation
- Mode-locking, Q-switching
- Semiconductor laser models
- Beam propagation, Gaussian beam optics
- **Computational methods**: ODE systems, BPM, Fox-Li

### IV.4 Ultrafast Optics
- Pulse propagation: nonlinear Schrödinger equation
- Self-phase modulation, group velocity dispersion
- Attosecond physics, HHG (high-harmonic generation)
- **Computational methods**: Split-step Fourier, TDSE

---

## V. THERMODYNAMICS & STATISTICAL MECHANICS

### V.1 Equilibrium Statistical Mechanics
- Microcanonical, canonical, grand canonical ensembles
- Partition functions: Z = Σ exp(-βE_i)
- Phase transitions: Ising model (exact 2D), Potts, XY
- Critical phenomena: renormalization group, universality, scaling
- Mean-field theory, Landau theory
- **Computational methods**: Monte Carlo (Metropolis, Wolff, Swendsen-Wang), Wang-Landau, exact enumeration

### V.2 Non-Equilibrium Statistical Mechanics
- Master equations, Fokker-Planck
- Fluctuation-dissipation theorem
- Jarzynski equality, Crooks fluctuation theorem
- Linear response theory (Kubo formula)
- Entropy production, irreversibility
- **Computational methods**: Kinetic Monte Carlo, stochastic simulation (Gillespie)

### V.3 Molecular Dynamics
- Force fields: AMBER, CHARMM, OPLS, GROMOS, ReaxFF (reactive)
- Ewald summation / PME for long-range electrostatics
- Thermostats: Nosé-Hoover, Langevin, Berendsen, velocity rescaling
- Barostats: Parrinello-Rahman, Berendsen
- Enhanced sampling: replica exchange (REMD), metadynamics, umbrella sampling, steered MD
- Coarse-graining: MARTINI, systematic (iterative Boltzmann inversion, force matching)
- Ab initio MD: Car-Parrinello, Born-Oppenheimer MD
- **Computational methods**: Verlet, RESPA, neighbor lists, domain decomposition
- **QTT angle**: Free energy surfaces as tensor trains

### V.4 Monte Carlo Methods (General)
- Importance sampling, Markov chain Monte Carlo
- Cluster algorithms (Wolff, Swendsen-Wang)
- Path integral Monte Carlo
- Quantum Monte Carlo: variational (VMC), diffusion (DMC), auxiliary field (AFQMC)
- **QTT angle**: QMC wavefunctions, trial state compression

### V.5 Heat Transfer
- Conduction: Fourier's law, transient/steady-state
- Convection: natural, forced, mixed; Nusselt correlations
- Radiation: view factors, radiosity, participating media (RTE)
- Conjugate heat transfer
- Phase change: Stefan problem, mushy zone, solidification
- **Computational methods**: FEM, FVM, Monte Carlo ray tracing (radiation)
- **QTT angle**: Temperature fields, view factor matrices

### V.6 Lattice Models & Spin Systems
- Ising (1D, 2D, 3D), Potts (q-state), clock model
- XY model, Heisenberg (classical), O(n) models
- Percolation: site, bond, continuum
- Cellular automata: Ising-like, lattice gas
- **Computational methods**: MC, cluster updates, tensor network renormalization (TRG, TNR)

---

## VI. QUANTUM MECHANICS (Single/Few-Body)

### VI.1 Time-Independent Schrödinger Equation
- Hψ = Eψ for bound states
- Hydrogen atom, harmonic oscillator, particle in a box
- WKB approximation
- **Computational methods**: Shooting method, spectral, DVR (discrete variable representation), FEM

### VI.2 Time-Dependent Schrödinger Equation (TDSE)
- iℏ∂ψ/∂t = Hψ
- Wavepacket propagation, tunneling
- Ionization dynamics (strong-field, attosecond)
- **Computational methods**: Crank-Nicolson, split-operator, Chebyshev propagator
- **QTT angle**: Wavefunction on grid → QTT

### VI.3 Scattering Theory
- Partial wave analysis
- Born approximation
- T-matrix, S-matrix
- Resonances (Breit-Wigner, Fano)
- Cross sections (elastic, inelastic, reactive)
- **Computational methods**: R-matrix, close-coupling, coupled channels

### VI.4 Semiclassical / WKB Methods
- Eikonal approximation
- Maslov index, caustics
- Initial value representation (IVR)
- Surface hopping (Tully), Ehrenfest
- **Computational methods**: Trajectory-based, Herman-Kluk propagator

### VI.5 Path Integral Methods
- Feynman path integral: K = ∫ D[x] exp(iS/ℏ)
- Imaginary time → statistical mechanics
- Instanton methods (tunneling rates)
- **Computational methods**: PIMC, ring polymer MD

---

## VII. QUANTUM MANY-BODY PHYSICS

### VII.1 Tensor Network Methods
- MPS / MPO (1D) — DMRG, TEBD, TDVP
- PEPS (2D projected entangled pair states)
- MERA (multi-scale entanglement renormalization ansatz)
- Tree tensor networks (TTN)
- Tensor network renormalization (TRG, TNR, loop-TNR)
- Corner transfer matrix (CTM)
- Infinite-size methods: iDMRG, iTEBD, iPEPS
- **QTT angle**: Core infrastructure — MPS is a tensor train

### VII.2 Quantum Spin Systems
- Heisenberg (XXX, XXZ, XYZ), Ising, XX
- Frustrated: J1-J2, triangular, kagome, pyrochlore
- Spin liquids, valence bond solids
- Haldane phase, SPT (symmetry-protected topological)
- Spin-orbit coupling models (Kitaev honeycomb)
- **Computational methods**: DMRG, QMC, exact diag, tensor networks

### VII.3 Strongly Correlated Electrons
- Hubbard model (single-band, multi-orbital)
- t-J model
- Anderson impurity model → DMFT
- Dynamical Mean-Field Theory (DMFT) + extensions (cluster, DFT+DMFT)
- Kondo physics
- Mott transition
- **Computational methods**: DMRG, DMFT (CT-QMC impurity solver), DCA, CDMFT

### VII.4 Topological Phases of Matter
- Toric code, Kitaev honeycomb model
- Fractional quantum Hall (Laughlin, composite fermion)
- Topological insulators (Z₂ invariant, Chern number)
- Symmetry-protected topological (SPT) phases
- Topological entanglement entropy (TEE)
- Anyons, non-Abelian braiding
- Berry phase, Wannier functions, Wilson loops
- **Computational methods**: Exact diag, DMRG, PEPS, entanglement measures

### VII.5 Many-Body Localization & Disorder
- Anderson localization (single-particle)
- Many-body localization (MBL) transition
- Random-field Heisenberg/XXZ
- Level spacing statistics (Poisson vs. GOE/GUE)
- Participation ratio, fractal dimensions
- Entanglement entropy growth: log(t) vs. linear
- Mobility edge
- **Computational methods**: Exact diag (shift-invert), DMRG, TEBD

### VII.6 Lattice Gauge Theory (Quantum)
- Wilson formulation, Kogut-Susskind Hamiltonian
- SU(2), SU(3) gauge groups
- Confinement, string breaking
- Mass gap, dimensional transmutation
- Schwinger model (QED in 1+1D)
- **Computational methods**: Tensor networks, quantum simulation, exact diag
- **QTT angle**: Your Yang-Mills 13,500 LOC

### VII.7 Open Quantum Systems
- Lindblad master equation: dρ/dt = -i[H,ρ] + Σ(LρL† - ½{L†L,ρ})
- Quantum trajectories / Monte Carlo wavefunction
- Redfield equation, Bloch-Redfield
- Non-Markovian dynamics (Nakajima-Zwanzig, HEOM)
- Quantum thermodynamics, heat engines
- **Computational methods**: MPO density matrix, vectorization, MPDO, HEOM
- **QTT angle**: Density matrix ρ as MPO = tensor train

### VII.8 Non-Equilibrium Quantum Dynamics
- Quantum quenches
- Thermalization, eigenstate thermalization hypothesis (ETH)
- Floquet (periodically driven) systems, time crystals
- Prethermalization
- Light-cone spreading of correlations (Lieb-Robinson)
- **Computational methods**: TEBD, TDVP, Krylov methods

### VII.9 Quantum Impurity & Kondo Physics
- Anderson impurity model
- Kondo effect, Kondo screening cloud
- Numerical renormalization group (NRG)
- Continuous-time QMC (CT-QMC)
- **Computational methods**: NRG, CT-QMC, DMRG

### VII.10 Bosonic Many-Body
- Bose-Hubbard (superfluid-Mott transition)
- Gross-Pitaevskii equation (BEC mean field)
- Bogoliubov theory (excitations)
- Tonks-Girardeau gas (1D hard-core bosons)
- Polariton condensates
- **Computational methods**: DMRG, QMC (worm algorithm), Gutzwiller mean field

### VII.11 Fermionic Systems
- Fermi-Hubbard, t-J model
- Jordan-Wigner, Bravyi-Kitaev transformations
- BCS theory, mean-field superconductivity
- FFLO states, unconventional pairing
- Fermi liquid theory, non-Fermi liquids
- **Computational methods**: DMRG, AFQMC, DMFT, diagrammatic MC

### VII.12 Nuclear Many-Body
- Nuclear shell model (configuration interaction)
- Coupled cluster for nuclei: CCSD, CCSD(T), CR-CC
- Nuclear density functional theory (DFT)
- Richardson-Gaudin pairing models
- Ab initio nuclear structure (chiral EFT interactions)
- Nuclear matter equation of state
- **Computational methods**: Shell model CI, CC, IMSRG, lattice EFT

### VII.13 Ultracold Atoms & Optical Lattices
- Optical lattice potentials: V(x) = V₀ sin²(kx)
- Feshbach resonances (tunable interactions)
- BEC-BCS crossover
- Quantum gas microscopy (single-site resolution)
- Synthetic gauge fields, SOC in cold atoms
- Dipolar gases, Rydberg arrays
- **Computational methods**: DMRG, QMC, exact diag, DMFT, Gross-Pitaevskii

---

## VIII. ELECTRONIC STRUCTURE & QUANTUM CHEMISTRY

### VIII.1 Density Functional Theory (DFT)
- Kohn-Sham equations: [-ℏ²∇²/2m + V_eff]ψ_i = ε_iψ_i
- Exchange-correlation: LDA, GGA (PBE), meta-GGA (SCAN), hybrid (B3LYP, HSE)
- Pseudopotentials: norm-conserving, ultrasoft, PAW
- Plane wave basis, Gaussian basis, real-space grid
- DFT+U (Hubbard correction for correlated systems)
- **Computational methods**: Self-consistent field (SCF), iterative diagonalization
- **QTT angle**: Electron density n(r) and KS orbitals as QTT

### VIII.2 Beyond-DFT: Correlated Methods
- Hartree-Fock (HF)
- MP2, MP3 (Møller-Plesset perturbation theory)
- Coupled Cluster: CCSD, CCSD(T), EOM-CC
- Configuration Interaction: FCI, CISD, MRCI
- CASSCF, CASPT2 (multi-reference)
- DMRG as FCI solver (quantum chemistry DMRG)
- **QTT angle**: Two-electron integrals, CI vectors as TT

### VIII.3 Semi-Empirical & Tight-Binding
- Extended Hückel, DFTB (density functional tight binding)
- Slater-Koster parameterization
- TB+U
- **Computational methods**: O(N) methods, recursive Green's function

### VIII.4 Excited States
- Time-dependent DFT (TDDFT): linear response, real-time
- GW approximation (quasiparticle energies)
- Bethe-Salpeter equation (BSE) for excitons
- ADC, EOM-CC for molecular excited states
- **Computational methods**: Casida equations, real-time propagation, contour deformation

### VIII.5 Response Properties
- Dielectric function ε(q,ω)
- Polarizability, hyperpolarizability
- NMR chemical shifts, EPR g-tensors
- IR/Raman spectra (phonons + dipole derivatives)
- Optical absorption, CD spectra
- **Computational methods**: DFPT (density functional perturbation theory), finite differences

### VIII.6 Relativistic Electronic Structure
- Dirac equation, Dirac-Kohn-Sham
- Scalar relativistic, spin-orbit coupling
- ZORA, Douglas-Kroll-Hess
- **Computational methods**: 4-component, 2-component, perturbative SOC

### VIII.7 Quantum Embedding
- DFT+DMFT
- Projection-based embedding
- ONIOM
- QM/MM (quantum mechanics / molecular mechanics)
- **Computational methods**: Self-consistent embedding loop

---

## IX. SOLID STATE / CONDENSED MATTER (CLASSICAL)

### IX.1 Phonons & Lattice Dynamics
- Dynamical matrix, phonon dispersion
- Density of states, thermodynamic properties
- Anharmonic effects: phonon-phonon scattering
- Thermal conductivity: Boltzmann transport (phonon BTE)
- **Computational methods**: DFPT, frozen phonon, molecular dynamics
- **QTT angle**: Phonon spectral functions, BTE distribution

### IX.2 Band Structure & Electronic Transport
- Bloch theorem, Brillouin zone
- Wannier functions, tight-binding interpolation
- Boltzmann transport equation (electronic)
- Landauer formalism, quantum conductance
- Hall effect, magnetoresistance
- **Computational methods**: DFT, Wannier90, BoltzTraP, NEGF

### IX.3 Magnetism (Classical/Mean-Field)
- Heisenberg exchange from DFT (J_ij mapping)
- Magnetic anisotropy (MAE)
- Spin waves, magnon dispersion
- Micromagnetics: Landau-Lifshitz-Gilbert equation
- Domain walls, skyrmions, magnetic textures
- **Computational methods**: Atomistic spin dynamics, micromagnetics (mumax, OOMMF)

### IX.4 Superconductivity (Computational)
- Eliashberg theory: α²F(ω) spectral function
- Allen-Dynes, McMillan T_c formula
- Ginzburg-Landau: free energy functional
- BdG (Bogoliubov-de Gennes) equations
- Vortex dynamics, Abrikosov lattice
- Unconventional: d-wave, p-wave, topological SC
- **Computational methods**: DFT+Eliashberg, BdG on lattice, Eilenberger

### IX.5 Disordered Systems
- Anderson model (tight-binding + random potential)
- Percolation theory
- Spin glasses: Edwards-Anderson, Sherrington-Kirkpatrick
- Random matrix theory
- **Computational methods**: Transfer matrix, kernel polynomial (KPM), exact diag

### IX.6 Surfaces & Interfaces
- Surface reconstruction, relaxation
- Adsorption energetics (physisorption, chemisorption)
- Surface states, Tamm/Shockley states
- Work function
- **Computational methods**: Slab DFT, surface Green's function

### IX.7 Defects in Solids
- Point defects: vacancies, interstitials, substitutional
- Formation energies, migration barriers (NEB)
- Color centers, deep levels
- Dislocations: Peierls-Nabarro, line tension
- Grain boundaries
- **Computational methods**: Supercell DFT, NEB, dislocation dynamics

### IX.8 Ferroelectrics & Multiferroics
- Polarization (Berry phase theory, modern theory of polarization)
- Landau-Devonshire theory
- Domain switching, piezoelectric response
- Magnetoelectric coupling
- **Computational methods**: DFT Berry phase, effective Hamiltonians, MD

---

## X. NUCLEAR & PARTICLE PHYSICS

### X.1 Nuclear Structure
- Shell model (configuration interaction)
- Nuclear DFT (Skyrme, Gogny, relativistic mean field)
- Ab initio: coupled cluster, IMSRG, no-core shell model
- Collective models: rotor, vibrator, interacting boson model (IBM)
- **Computational methods**: Large-scale CI, CC, IMSRG, Monte Carlo shell model

### X.2 Nuclear Reactions
- Optical model (complex potential scattering)
- Hauser-Feshbach (statistical, compound nucleus)
- R-matrix theory
- Direct reactions (DWBA, CDCC)
- **Computational methods**: Coupled channels, TALYS, EMPIRE

### X.3 Nuclear Astrophysics
- r-process, s-process nucleosynthesis
- Nuclear equation of state (neutron stars)
- Neutron star mergers, kilonovae
- Type Ia/II supernovae nucleosynthesis
- **Computational methods**: Reaction network solvers, hydro + nucleosynthesis

### X.4 Lattice QCD
- Discretized QCD on spacetime lattice
- Wilson/Kogut-Susskind/domain wall/overlap fermions
- Hadron spectrum, form factors
- Quark masses, strong coupling constant α_s
- Finite temperature QCD, deconfinement
- **Computational methods**: HMC (Hybrid Monte Carlo), multigrid solvers
- **QTT angle**: Lattice gauge theory in TN form (your Yang-Mills connects here)

### X.5 Perturbative QFT
- Feynman diagram evaluation
- Loop integrals (dimensional regularization, IBP reduction)
- Renormalization (MS-bar, on-shell)
- Cross sections: QED, QCD, electroweak
- **Computational methods**: Symbolic algebra (FORM), sector decomposition, numerical integration

### X.6 Beyond Standard Model
- Dark matter: WIMP cross sections, relic abundance
- Neutrino oscillations: PMNS matrix, matter effects (MSW)
- Baryon asymmetry: leptogenesis, electroweak baryogenesis
- **Computational methods**: Boltzmann equations, parameter scans

---

## XI. PLASMA PHYSICS

### XI.1 Ideal MHD
- Continuity, momentum (ρdv/dt = -∇p + J×B), induction (∂B/∂t = ∇×(v×B))
- Alfvén waves, magnetosonic waves
- MHD equilibria: Grad-Shafranov equation
- MHD stability: energy principle, ballooning, kink, tearing
- **Computational methods**: FEM/spectral (equilibrium), initial value (stability)
- **QTT angle**: TOMAHAWK operates here

### XI.2 Resistive / Extended MHD
- Resistive MHD: η∇²B terms (reconnection)
- Hall MHD: J×B/ne terms
- Two-fluid MHD
- Gyroviscosity
- **Computational methods**: Implicit schemes, AMR

### XI.3 Kinetic Theory (Plasma)
- Vlasov equation: ∂f/∂t + v·∇f + (q/m)(E+v×B)·∇_v f = 0
- Vlasov-Poisson, Vlasov-Maxwell
- Landau damping, filamentation
- **Computational methods**: PIC (particle-in-cell), Vlasov solvers, continuum kinetic
- **QTT angle**: Your Vlasov 5D/6D QTT solvers

### XI.4 Gyrokinetics
- Reduced kinetic theory (average over gyration)
- 5D phase space: (R, v_∥, μ)
- Turbulent transport in tokamaks
- ITG, TEM, ETG instabilities
- **Computational methods**: PIC (GTC, ORB5), continuum (GENE, GS2)
- **QTT angle**: 5D distribution function → QTT compression

### XI.5 Magnetic Reconnection
- Sweet-Parker, Petschek models
- Collisionless reconnection (kinetic)
- Guide field reconnection
- Plasmoid instability, fractal reconnection
- **Computational methods**: PIC, hybrid PIC-MHD, resistive MHD

### XI.6 Laser-Plasma Interaction
- Parametric instabilities: SRS, SBS, TPD
- Laser wakefield acceleration
- Relativistic self-focusing
- Target normal sheath acceleration (TNSA)
- Inertial confinement fusion (ICF) implosion physics
- **Computational methods**: PIC (OSIRIS, EPOCH, WarpX), radiation-hydro

### XI.7 Dusty / Complex Plasmas
- Charging of dust grains
- Yukawa (screened Coulomb) systems
- Dust-acoustic waves, dust crystals
- **Computational methods**: MD, PIC with grain dynamics

### XI.8 Space & Astrophysical Plasma
- Solar wind, magnetospheric physics
- Cosmic ray transport (Parker equation)
- Astrophysical jets, accretion disks
- Dynamo theory (magnetic field generation)
- **Computational methods**: Global MHD, hybrid, PIC

---

## XII. ASTROPHYSICS & COSMOLOGY

### XII.1 Stellar Structure & Evolution
- Hydrostatic equilibrium: dP/dr = -Gρm/r²
- Nuclear burning networks: pp chain, CNO, triple-α
- Stellar opacity (OPAL, OP)
- Convection: mixing length theory, 3D convection simulations
- **Computational methods**: 1D stellar evolution codes (MESA), 3D hydro

### XII.2 Compact Objects
- White dwarf structure (Chandrasekhar limit)
- Neutron star: TOV equation, EOS
- Black holes: Kerr metric, ISCO, photon sphere
- Accretion: thin disk (Shakura-Sunyaev), ADAF, GRMHD
- **Computational methods**: GR hydro, GRMHD, ray tracing

### XII.3 Gravitational Waves
- Linearized GR: h_μν perturbations
- Inspiral: post-Newtonian, effective one-body (EOB)
- Merger: numerical relativity (BSSN, CCZ4)
- Ringdown: quasinormal modes
- Waveform templates (IMRPhenom, SEOBNRv4)
- **Computational methods**: 3+1 numerical relativity, spectral methods (SpEC)

### XII.4 Cosmological Simulations
- N-body (dark matter): PM, P3M, tree-PM, AMR
- Hydrodynamic cosmology: SPH (Gadget), AMR (Enzo, RAMSES), moving mesh (Arepo)
- Structure formation: halo mass function, merger trees
- Cosmic web: filaments, voids, sheets
- **Computational methods**: N-body, hydro, semi-analytic models

### XII.5 CMB & Early Universe
- Boltzmann hierarchy: photon, baryon, CDM, neutrino perturbations
- Recombination (Peebles, HyRec)
- Reionization
- Inflation: slow-roll, power spectrum, tensor modes
- **Computational methods**: CLASS, CAMB, CosmoMC

### XII.6 Radiative Transfer (Astrophysical)
- Radiation transport equation: dI/ds = -κI + j
- Diffusion approximation, Eddington approximation
- Monte Carlo radiative transfer
- **Computational methods**: Ray tracing, MC, discrete ordinates (S_N), VEF

---

## XIII. GEOPHYSICS & EARTH SCIENCE

### XIII.1 Seismology
- Elastic wave equation in heterogeneous media
- P-waves, S-waves, surface waves (Rayleigh, Love)
- Full waveform inversion (FWI)
- Seismic tomography
- **Computational methods**: Spectral element (SPECFEM3D), FD, FWI
- **QTT angle**: 3D Earth models, seismic wavefields

### XIII.2 Mantle Convection
- Stokes flow with temperature-dependent viscosity
- Rayleigh-Bénard convection (high Ra)
- Plate tectonics, subduction dynamics
- **Computational methods**: FEM (CitcomS, ASPECT), spectral

### XIII.3 Geomagnetism & Dynamo
- Geodynamo: coupled MHD in rotating spherical shell
- Secular variation, geomagnetic reversals
- **Computational methods**: Pseudospectral (spherical harmonics + Chebyshev)

### XIII.4 Atmospheric Physics
- Radiative-convective equilibrium
- Atmospheric chemistry (Chapman cycle, ozone)
- Cloud microphysics
- **Computational methods**: GCM (general circulation model), LES

### XIII.5 Oceanography
- Ocean general circulation
- Thermohaline circulation
- Internal waves, tides
- Turbulent mixing, mesoscale eddies
- **Computational methods**: Primitive equations, isopycnal models

### XIII.6 Glaciology
- Ice sheet dynamics: shallow ice approximation (SIA), full Stokes
- Calving, grounding line dynamics
- Ice rheology: Glen's flow law
- **Computational methods**: FEM, SIA

---

## XIV. MATERIALS SCIENCE (Computational)

### XIV.1 First-Principles Materials Design
- High-throughput DFT screening
- Phase diagrams (CALPHAD, cluster expansion)
- Convex hull (thermodynamic stability)
- Phonon stability, mechanical stability
- **Computational methods**: DFT workflows (AiiDA, FireWorks), cluster expansion

### XIV.2 Mechanical Properties
- Elastic constants (C_ij tensor)
- Ideal strength, theoretical shear strength
- Fracture toughness (DFT, MD, phase-field)
- Fatigue: Paris law, microstructure-sensitive
- Creep: diffusion, power-law, Nabarro-Herring, Coble
- **Computational methods**: DFT, MD, continuum (FEM), crystal plasticity FEM (CPFEM)

### XIV.3 Phase-Field Methods
- Cahn-Hilliard (conserved): ∂c/∂t = ∇·(M∇(δF/δc))
- Allen-Cahn (non-conserved): ∂φ/∂t = -L(δF/δφ)
- Solidification: dendritic growth, microsegregation
- Grain growth, recrystallization
- Martensitic transformation
- Spinodal decomposition
- **Computational methods**: FEM, FDM, spectral (FFT-based)
- **QTT angle**: Order parameter fields as tensor trains

### XIV.4 Microstructure Evolution
- Monte Carlo grain growth (Potts model)
- Cellular automata recrystallization
- Precipitate nucleation, growth, coarsening (KWN model)
- Texture evolution (ODF, crystal plasticity)
- **Computational methods**: KMC, Potts MC, level set, phase field

### XIV.5 Radiation Damage
- Primary knock-on atom (PKA) cascades
- Frenkel pairs, interstitial clusters
- Void swelling, radiation embrittlement
- Dislocation loop evolution
- **Computational methods**: BCA (SRIM/TRIM), MD cascades, object KMC, rate theory

### XIV.6 Polymers & Soft Matter
- Polymer self-consistent field theory (SCFT)
- Flory-Huggins: ΔG_mix = kT[n₁lnφ₁ + n₂lnφ₂ + χn₁φ₂]
- Block copolymer morphology
- Rubber elasticity (affine, phantom network)
- Liquid crystals: Maier-Saupe, Landau-de Gennes
- Colloidal assembly, DLVO theory
- **Computational methods**: SCFT, coarse-grained MD, DPD (dissipative particle dynamics)
- **QTT angle**: SCFT order parameter fields

### XIV.7 Ceramics & High-Temperature Materials
- Sintering models (viscous, solid-state)
- Thermal barrier coatings
- Ultra-high temperature ceramics (UHTC)
- **Computational methods**: Phase field, kinetic Monte Carlo, MD
- **QTT angle**: HELL-SKIN operates here

---

## XV. CHEMICAL PHYSICS & REACTION DYNAMICS

### XV.1 Potential Energy Surfaces
- Born-Oppenheimer approximation
- PES construction: ab initio, fitting, machine learning
- Saddle point search: NEB, dimer, growing string
- IRC (intrinsic reaction coordinate)
- **Computational methods**: DFT, CCSD(T), interpolation, neural network PES

### XV.2 Reaction Rate Theory
- Transition state theory (TST)
- Variational TST
- RRKM theory (unimolecular)
- Kramers theory (diffusive barrier crossing)
- Instanton theory (tunneling)
- **Computational methods**: Harmonic TST, VTST, ring polymer instanton

### XV.3 Quantum Reaction Dynamics
- Reactive scattering: H+H₂ benchmark
- Wavepacket propagation on PES
- Cumulative reaction probability: N(E) = Tr[Ŝ†Ŝ]
- **Computational methods**: DVR, split-operator, MCTDH

### XV.4 Nonadiabatic Dynamics
- Conical intersections
- Surface hopping (Tully, FSSH)
- Ehrenfest dynamics
- Multi-configurational Ehrenfest
- Ab initio multiple spawning (AIMS)
- **Computational methods**: Trajectory surface hopping, MCTDH, variational multiconfigurational Gaussians

### XV.5 Photochemistry
- Excited state dynamics, photodissociation
- Internal conversion, intersystem crossing
- Fluorescence, phosphorescence lifetimes
- Photocatalysis
- **Computational methods**: TDDFT, CASSCF dynamics, surface hopping

### XV.6 Catalysis
- Heterogeneous: adsorption, Sabatier principle, volcano plots
- Homogeneous: organometallic catalytic cycles
- Enzyme catalysis: QM/MM
- Electrochemistry: Butler-Volmer, computational hydrogen electrode
- **Computational methods**: DFT (slab models), microkinetic modeling, KMC

### XV.7 Spectroscopy (Computational)
- UV-Vis: TDDFT, EOM-CC
- IR/Raman: harmonic/anharmonic frequencies
- NMR: GIAO, chemical shift tensors
- X-ray: XAS (XANES, EXAFS), XPS, RIXS
- EPR/ESR: g-tensor, hyperfine coupling
- Mössbauer: isomer shift, quadrupole splitting
- **Computational methods**: DFPT, linear response, real-time propagation, Bethe-Salpeter

---

## XVI. BIOPHYSICS & COMPUTATIONAL BIOLOGY

### XVI.1 Protein Structure & Dynamics
- Molecular dynamics with force fields
- Homology modeling, threading
- Ab initio structure prediction (Rosetta, AlphaFold-class)
- Protein folding: free energy landscapes
- Intrinsically disordered proteins
- **Computational methods**: MD, enhanced sampling, coarse-grained

### XVI.2 Drug Design & Binding
- Molecular docking (AutoDock, Glide)
- Free energy perturbation (FEP), thermodynamic integration (TI)
- Pharmacophore modeling
- QSAR/QSPR
- **Computational methods**: Docking, FEP/TI, machine learning
- **QTT angle**: Your TIG-011a (QTT binding pocket)

### XVI.3 Membrane Biophysics
- Lipid bilayer self-assembly
- Membrane protein insertion, channel gating
- Electroporation
- **Computational methods**: Coarse-grained MD (MARTINI), all-atom MD

### XVI.4 Nucleic Acids
- DNA/RNA structure, base stacking, hydrogen bonding
- DNA mechanics: persistence length, supercoiling
- RNA folding: secondary structure prediction (Zuker), 3D
- **Computational methods**: MD, MC, dynamic programming

### XVI.5 Systems Biology
- Metabolic networks: FBA (flux balance analysis)
- Gene regulatory networks: Boolean, ODE
- Signaling pathways: mass action kinetics
- Population dynamics: Lotka-Volterra, SIR/SEIR
- **Computational methods**: LP (FBA), ODE integration, stochastic simulation (Gillespie)

### XVI.6 Neuroscience (Computational)
- Hodgkin-Huxley: C dV/dt = -g_Na m³h(V-E_Na) - g_K n⁴(V-E_K) - g_L(V-E_L) + I
- Cable equation (dendritic computation)
- Network models: balanced E/I, criticality
- Mean-field neural field equations (Wilson-Cowan)
- Synaptic plasticity: STDP, BCM
- **Computational methods**: NEURON, Brian, nest, reduced models
- **QTT angle**: Your QTT Brain / Connectome

---

## XVII. COMPUTATIONAL METHODS (CROSS-CUTTING)

### XVII.1 Optimization
- Gradient-based: Newton, BFGS, L-BFGS, conjugate gradient
- Topology optimization: SIMP, level-set, homogenization
- Shape optimization, adjoint methods
- Global: genetic algorithms, simulated annealing, particle swarm, CMA-ES
- Multi-objective: Pareto front, NSGA-II
- **QTT angle**: Your OPT-QTT module

### XVII.2 Inverse Problems
- Parameter estimation, data assimilation
- Regularization: Tikhonov, L1 (LASSO), total variation
- Bayesian inference: MCMC, variational inference
- Uncertainty quantification: polynomial chaos, stochastic collocation
- **Computational methods**: Adjoint-based, ensemble methods

### XVII.3 Machine Learning for Physics
- Neural network potentials: SchNet, NequIP, MACE, ANI
- Physics-informed neural networks (PINNs)
- Neural operators: FNO, DeepONet
- Equivariant neural networks (E(3), SE(3))
- Generative models for molecular design (diffusion, flow matching)
- Active learning for materials discovery
- **QTT angle**: TT decomposition of weight tensors, TT-cross for training

### XVII.4 Mesh Generation & Adaptive Methods
- Delaunay, advancing front
- Adaptive mesh refinement (AMR, h/p/hp)
- Immersed boundary / fictitious domain
- Octree/quadtree
- **QTT angle**: Hierarchical grids map to QTT indexing

### XVII.5 Linear Algebra (Large-Scale)
- Krylov solvers: CG, GMRES, BiCGSTAB, MINRES
- Preconditioners: ILU, AMG, domain decomposition
- Eigensolvers: Lanczos, Arnoldi, Davidson, LOBPCG
- Sparse direct: LU, Cholesky (SuperLU, MUMPS, PARDISO)
- Randomized linear algebra
- **QTT angle**: Your CG solvers, Lanczos in DMRG

### XVII.6 High-Performance Computing
- MPI, OpenMP, GPU (CUDA, ROCm)
- Domain decomposition, load balancing
- Communication-avoiding algorithms
- Scalable I/O
- Performance modeling (roofline, Amdahl/Gustafson)

---

## XVIII. CONTINUUM SCALE COUPLED PHYSICS

### XVIII.1 Fluid-Structure Interaction (FSI)
- Partitioned: separate fluid + solid solvers, interface coupling
- Monolithic: single system
- ALE formulation, immersed methods
- Aeroelasticity: flutter, buffeting, vortex-induced vibration
- Hemodynamics: blood flow in compliant vessels

### XVIII.2 Thermo-Mechanical Coupling
- Thermal stress: σ = C(ε - αΔT)
- Coupled conduction + deformation
- Thermal buckling
- Phase-change with mechanics (casting, welding)

### XVIII.3 Electro-Mechanical Coupling
- Piezoelectricity: σ = Cε - eE, D = eε + κE
- Electrostriction
- MEMS/NEMS simulation
- Electrostatic actuators

### XVIII.4 Magneto-Hydrodynamics (Coupled)
- Liquid metal MHD: Hartmann flow, MHD pumps
- Crystal growth in magnetic field
- Electromagnetic braking
- **QTT angle**: Your dynamics engine + TOMAHAWK

### XVIII.5 Chemically Reacting Flows
- Combustion + turbulence coupling (TNF)
- Chemical vapor deposition (CVD)
- Atmospheric chemistry transport
- Reactive transport in porous media

### XVIII.6 Radiation-Hydrodynamics
- Flux-limited diffusion
- Implicit Monte Carlo
- ICF implosion physics
- Stellar interiors
- **Computational methods**: Grey/multigroup, S_N, IMC

### XVIII.7 Multiscale Methods
- Concurrent: FE² (macro-micro), HMM
- Sequential: homogenization, coarse-graining
- QM/MM embedding
- Bridging: atomistic-continuum (quasicontinuum, CaDD)

---

## XIX. QUANTUM INFORMATION & COMPUTATION

### XIX.1 Quantum Circuit Simulation
- Gate-based: universal gate sets
- Circuit depth, T-count optimization
- Tensor network contraction of circuits
- Stabilizer simulation (Gottesman-Knill)
- **QTT angle**: Your ZK-QTT

### XIX.2 Quantum Error Correction
- Stabilizer codes: Steane, Shor, surface code
- Topological codes: toric, color code
- Decoding: MWPM, union-find, neural
- Fault-tolerant thresholds
- **QTT angle**: Your error mitigation module

### XIX.3 Quantum Algorithms
- VQE, QAOA
- Quantum phase estimation
- Grover search
- HHL (linear systems)
- Quantum walks
- **Computational methods**: Statevector, density matrix, MPS simulators

### XIX.4 Quantum Simulation
- Digital: Trotter decomposition of H
- Analog: Hamiltonian engineering
- Variational quantum eigensolver
- Quantum approximate optimization
- **QTT angle**: Your hybrid VQE/QAOA module

### XIX.5 Quantum Cryptography & Communication
- QKD protocols: BB84, E91, decoy state
- Entanglement distillation
- Quantum repeaters
- Post-quantum cryptography (lattice, code-based, hash)
- **QTT angle**: Your PQC-QTT (CRYSTALS-Dilithium, FIPS 204)

---

## XX. SPECIAL / APPLIED DOMAINS

### XX.1 Relativistic Mechanics
- Special relativity: Lorentz transformation, 4-vectors
- Relativistic particle dynamics
- Thomas precession
- **QTT angle**: CHRONOS

### XX.2 General Relativity (Numerical)
- 3+1 decomposition: ADM, BSSN, Z4
- Black hole binaries, gravitational waveforms
- Cosmological perturbation theory
- **QTT angle**: METRIC ENGINE

### XX.3 Astrodynamics
- Orbital mechanics: two-body, restricted three-body
- Perturbations: J2, drag, solar radiation pressure, lunisolar
- Interplanetary trajectories, gravity assists
- Formation flying, rendezvous
- Space debris tracking
- **QTT angle**: ORBITAL FORGE

### XX.4 Robotics Physics
- Rigid-body dynamics: Featherstone, recursive Newton-Euler
- Contact: LCP (linear complementarity), friction cone
- Soft body: FEM-based, position-based dynamics
- Cable/tendon: Cosserat rod theory

### XX.5 Acoustics (Applied)
- Aeroacoustics: Lighthill's analogy, FW-H equation
- Computational aeroacoustics (CAA): LEE, APE
- Structural acoustics: coupled FEM-BEM
- Noise prediction: fan, jet, boundary layer

### XX.6 Biomedical Engineering
- Hemodynamics: blood flow, FSI in arteries
- Cardiac electrophysiology: bidomain, monodomain models
- Drug delivery: convection-diffusion, pharmacokinetics
- Tissue mechanics: hyperelastic, poroelastic
- Medical imaging: CT reconstruction, MRI (Bloch equations)

### XX.7 Environmental Physics
- Climate modeling: coupled atmosphere-ocean-land-ice
- Air pollution dispersion: Gaussian plume, Eulerian transport
- Wildfire: coupled fire-atmosphere (WRF-Fire)
- Hydrology: rainfall-runoff, groundwater flow
- Coastal: storm surge, wave modeling (SWAN, WaveWatch)

### XX.8 Energy Systems
- Solar: device physics (drift-diffusion, Shockley-Queisser)
- Wind: BEM (blade element momentum), actuator disk/line
- Battery: Newman model (porous electrode theory), SEI growth
- Fuel cells: Butler-Volmer, proton transport
- Nuclear reactor: neutron transport (diffusion, S_N, Monte Carlo)
- Fusion: see Section XI
- **QTT angle**: Battery (Li₃InCl₄.₈Br₁.₂), fusion (STAR-HEART)

### XX.9 Manufacturing & Process Simulation
- Casting: solidification, shrinkage, hot tearing
- Welding: heat source models (Goldak), residual stress
- Additive manufacturing: powder bed, melt pool (Marangoni)
- Machining: chip formation, tool wear
- Crystal growth: Czochralski, Bridgman, VGF
- **QTT angle**: FEMTO-FABRICATOR (mechanosynthesis)

### XX.10 Semiconductor Device Physics
- Drift-diffusion: Poisson + continuity (electrons, holes)
- Hydrodynamic model
- Quantum transport: NEGF (non-equilibrium Green's function)
- Band-to-band tunneling, impact ionization
- **Computational methods**: TCAD (Sentaurus, Silvaco), NEGF
- **QTT angle**: Your SnHf-F EUV resist physics

---

## DOMAIN COUNT SUMMARY

| Category | Sub-domains |
|----------|:-----------:|
| I. Classical Mechanics | 6 |
| II. Fluid Dynamics | 10 |
| III. Electromagnetism | 7 |
| IV. Optics & Photonics | 4 |
| V. Thermodynamics & Statistical Mechanics | 6 |
| VI. Quantum Mechanics (Single/Few-Body) | 5 |
| VII. Quantum Many-Body Physics | 13 |
| VIII. Electronic Structure & Quantum Chemistry | 7 |
| IX. Solid State / Condensed Matter (Classical) | 8 |
| X. Nuclear & Particle Physics | 6 |
| XI. Plasma Physics | 8 |
| XII. Astrophysics & Cosmology | 6 |
| XIII. Geophysics & Earth Science | 6 |
| XIV. Materials Science | 7 |
| XV. Chemical Physics & Reaction Dynamics | 7 |
| XVI. Biophysics & Computational Biology | 6 |
| XVII. Computational Methods (Cross-Cutting) | 6 |
| XVIII. Continuum Coupled Physics | 7 |
| XIX. Quantum Information & Computation | 5 |
| XX. Special / Applied Domains | 10 |
| **TOTAL** | **140** |

---

*Compiled for Tigantic Holdings LLC — Full Computational Physics Capability Assessment*
*© 2026 Brad McAllister. All rights reserved.*
