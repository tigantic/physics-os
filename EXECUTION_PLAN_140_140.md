# HyperTensor 140/140 Capability Domain Execution Plan

**Repository**: HyperTensor-VM (`workspace-reorg` branch)
**Date**: February 7, 2026
**Baseline**: 78/140 covered (32 full + 46 partial), 62 uncovered
**Target**: 140/140 full coverage — every sub-domain with production solver, equations, validation
**Owner**: Tigantic Holdings LLC

---

## Master Ledger

| Phase | Domains | New LOC | New Equations | Calendar | Cumulative |
|:-----:|---------|--------:|:------------:|---------:|-----------:|
| 0 | Upgrade 46 partial → full | ~18,000 | ~90 | Weeks 1–4 | 78 → 78 full |
| 1 | Natural Extensions (QTT-native) | ~22,000 | ~110 | Weeks 5–10 | 78 → 96 |
| 2 | High-Value New Domains | ~28,000 | ~140 | Weeks 11–18 | 96 → 118 |
| 3 | Completeness (full taxonomy) | ~32,000 | ~160 | Weeks 19–28 | 118 → 140 |
| **Total** | **140/140** | **~100,000** | **~500** | **28 weeks** | **140 full** |

Post-instantiation totals: ~277,645 LOC, ~1,029 equations, 450+ files, 34 → 48 domains in Platform Spec.

---

## Phase 0 — Upgrade 46 Partial Domains to Full (Weeks 1–4)

Each partial domain already has equations/models but lacks completeness. Upgrades fill the specific gaps identified in the coverage assessment.

### Week 1: Classical & Fluid Upgrades

| # | Domain | Current Gap | Deliverable | Est. LOC |
|:-:|--------|------------|------------|--------:|
| I.2 | Lagrangian/Hamiltonian | No variational integrators | Symplectic Störmer-Verlet, Ruth-4, Yoshida-6; Noether conservation verifier; action minimization | 600 |
| I.3 | Continuum Mechanics | Linear only | Neo-Hookean + Mooney-Rivlin hyperelastic; Drucker-Prager plasticity; cohesive-zone fracture; Updated Lagrangian large-deformation | 1,200 |
| I.4 | Structural Mechanics | No beams/plates/modal | Timoshenko beam, Mindlin-Reissner plate, eigenvalue buckling, modal analysis (Lanczos), composite CLT with Tsai-Wu | 1,000 |
| I.6 | Acoustics | Underwater only | Helmholtz BEM, structural-acoustic coupling, room acoustics (image source), aeroacoustics (Lighthill analogy) | 800 |
| II.6 | Rarefied Gas | Vlasov only | DSMC solver (NTC collision kernel), BGK relaxation operator, Knudsen-regime switch | 700 |
| II.8 | Non-Newtonian | Carreau only | Oldroyd-B viscoelastic (log-conformation), Bingham plastic, FENE-P polymer, Herschel-Bulkley | 600 |

**Week 1 subtotal**: 4,900 LOC, 6 domains upgraded

### Week 2: Physics & Quantum Upgrades

| # | Domain | Current Gap | Deliverable | Est. LOC |
|:-:|--------|------------|------------|--------:|
| III.1 | Electrostatics | Generic Poisson only | Dedicated Poisson-Boltzmann solver, capacitance extraction, multipole expansion, charge distributions | 500 |
| IV.1 | Physical Optics | EUV only | Fresnel/Fraunhofer diffraction via QTT, Jones/Mueller polarization calculus, angular spectrum propagation | 600 |
| V.1 | Equilibrium StatMech | Spin models only | Canonical partition function engine, Metropolis/Wolff cluster MC, Wang-Landau flat-histogram, Landau mean-field | 700 |
| V.2 | Non-Eq StatMech | Fokker-Planck only | Jarzynski equality verifier, Crooks theorem, Kubo linear response, kinetic Monte Carlo engine, Gillespie SSA | 600 |
| V.3 | Molecular Dynamics | Langevin only | Velocity Verlet + Nosé-Hoover thermostat + Parrinello-Rahman barostat; AMBER/OPLS force field params; PME electrostatics; REMD enhanced sampling | 900 |
| V.5 | Heat Transfer | No radiation | View factor computation (MC ray-tracing), radiosity method, participating media RTE, Stefan solidification, conjugate CHT | 700 |

**Week 2 subtotal**: 4,000 LOC, 6 domains upgraded

### Week 3: QM, Many-Body & Condensed Matter Upgrades

| # | Domain | Current Gap | Deliverable | Est. LOC |
|:-:|--------|------------|------------|--------:|
| VI.1 | Time-Indep SE | DMRG only | DVR (discrete variable repr.), shooting method, spectral solver for hydrogen/harmonic oscillator/box; WKB tunneling | 500 |
| VI.2 | Time-Dep SE | TEBD/TDVP only | Split-operator for single particle on grid, Crank-Nicolson, Chebyshev propagator, wavepacket tunneling | 500 |
| VI.5 | Path Integrals | φ⁴ lattice only | Ring-polymer MD (RPMD), path-integral MC (PIMC) for He-4, instanton tunneling rate | 600 |
| VII.3 | Strongly Correlated | Hubbard MPO only | DMFT single-site with Hirsch-Fye impurity solver, t-J model MPO, Mott gap tracking | 800 |
| VII.7 | Open Quantum Sys | Kraus only | Full Lindblad master equation (MPO density matrix), quantum trajectories / MC wavefunction, Redfield equation | 700 |
| VII.8 | Non-Eq QM Dynamics | TEBD quenches only | Floquet Hamiltonian, prethermalization tracking, ETH diagnostics (eigenstate-to-eigenstate variances), Lieb-Robinson cone extraction | 600 |
| VII.10 | Bosonic Many-Body | Bose-Hubbard only | Gross-Pitaevskii BEC solver (imaginary-time + real-time), Bogoliubov excitation spectrum, Tonks-Girardeau limit | 600 |
| VII.11 | Fermionic Systems | JW/Hubbard only | BCS mean-field pairing, FFLO solver, Bravyi-Kitaev transform, Fermi-liquid Landau parameter extraction | 600 |

**Week 3 subtotal**: 4,900 LOC, 8 domains upgraded

### Week 4: Applied & Coupled Upgrades

| # | Domain | Current Gap | Deliverable | Est. LOC |
|:-:|--------|------------|------------|--------:|
| IX.1 | Phonons | Debye only | Full dynamical matrix from force constants, phonon dispersion plotter, anharmonic phonon-phonon (3-phonon), phonon BTE thermal conductivity | 700 |
| IX.5 | Disordered Systems | RMT only | Anderson tight-binding model with random on-site, KPM spectral, Edwards-Anderson spin glass MC, participation ratio / fractal dimension | 600 |
| IX.7 | Defects | NEB only | Point defect formation energy calculator, vacancy/interstitial energetics, dislocation Peierls-Nabarro model, grain boundary energy | 500 |
| X.4 | Lattice QCD | SU(2) only | SU(3) gauge group (Gell-Mann matrices), Wilson fermion determinant (quenched), Creutz ratio confinement, hadron correlator | 900 |
| X.5 | Perturbative QFT | φ⁴ + RG only | 1-loop Feynman diagram evaluator (dimensional reg.), MS-bar renormalization, QED vertex corrections, running coupling | 800 |
| XI.2 | Resistive/Ext MHD | Resistive only | Hall MHD (J×B/ne), two-fluid plasma model, gyroviscosity, implicit time integration for stiff terms | 600 |
| XI.6 | Laser-Plasma | Wakefield only | Stimulated Raman scattering (SRS), stimulated Brillouin (SBS), relativistic self-focusing; parametric instability growth rates | 500 |
| XI.8 | Space/Astro Plasma | Solar wind only | Cosmic ray Parker transport, astrophysical jet launching (Blandford-Znajek), planetary dynamo seed model | 500 |
| XII.2 | Compact Objects | Schwarzschild only | TOV equation (NS structure), Kerr metric + ISCO, Shakura-Sunyaev thin disk accretion | 600 |
| XIII.4 | Atmospheric Phys | Weather only | Chapman ozone cycle, cloud microphysics (Kessler warm-rain), radiative-convective equilibrium | 500 |
| XIII.5 | Oceanography | Munk profile only | Primitive equation ocean model (beta-plane), thermohaline box model, internal wave dispersion, tidal constituents | 600 |
| XIV.2 | Mechanical Props | Elastic only | Full C_ij tensor extraction, ideal strength (Frenkel), fracture toughness (Griffith), Paris fatigue law, power-law creep | 500 |
| XIV.7 | Ceramics/High-T | TPS only | Sintering model (viscous + solid-state), UHTC oxidation kinetics, thermal barrier coating heat flux | 400 |
| XV.1 | PES Construction | Phonon PES only | Born-Oppenheimer PES builder, NEB saddle search, IRC follower, 2D PES contour plotter | 500 |
| XV.2 | Reaction Rate | Arrhenius only | Harmonic TST with tunneling (Wigner), variational TST, RRKM unimolecular, Kramers diffusive barrier | 500 |
| XV.5 | Photochemistry | Dill ABC only | Excited-state relaxation (IC/ISC rates), photodissociation on model PES, fluorescence lifetime calculator | 400 |
| XVI.5 | Systems Biology | SIR/SEIR only | Flux balance analysis (FBA via LP), gene regulatory network (Boolean + ODE), Gillespie stochastic simulation | 600 |
| XVIII.4 | Coupled MHD | Fluid + B only | Hartmann liquid metal flow, crystal growth in B-field, MHD pump model, electromagnetic braking | 500 |
| XVIII.7 | Multiscale | QM/MM implied | FE² concurrent macro-micro, homogenization engine, bridging atomistic-continuum (quasi-continuum) | 700 |
| XX.1 | Relativistic Mech | Corrections only | Full Lorentz dynamics (4-vector), Thomas precession, relativistic rocket equation, velocity addition | 400 |
| XX.2 | Numerical GR | Schwarzschild only | BSSN formalism (constraint propagation), gauge conditions (1+log, Gamma-driver), puncture initial data | 800 |
| XX.5 | Applied Acoustics | Underwater only | Lighthill aeroacoustics, FW-H surface integral, LEE linearized Euler, jet noise model (Tam-Auriault) | 500 |
| XX.6 | Biomedical Eng | Blood flow only | Bidomain cardiac electrophysiology (FitzHugh-Nagumo + monodomain), drug delivery PK model, tissue hyperelastic (Holzapfel-Ogden) | 600 |
| XX.7 | Environmental | Wildfire/agri only | Gaussian plume dispersion, hydrological rainfall-runoff (SCS curve number), coastal storm surge (shallow water), coupled fire-atmosphere | 600 |
| XX.8 | Energy Systems | SSB/wind/fusion only | Drift-diffusion solar cell (Shockley-Queisser, 1D Poisson-continuity), Newman porous electrode battery, Butler-Volmer fuel cell, Boltzmann neutron diffusion | 800 |
| XX.9 | Manufacturing | EUV/etch only | Goldak welding heat source, solidification (Scheil equation), melt pool Marangoni (AM), machining Merchant model | 600 |

**Week 4 subtotal**: 14,200 LOC, 26 domains upgraded

**Phase 0 total: 28,000 LOC → all 46 partial upgraded to full (78/140, all full)**

---

## Phase 1 — Natural QTT Extensions: 18 New Domains (Weeks 5–10)

High QTT synergy — these domains either compress naturally into tensor trains or directly extend existing solvers.

### Sprint 1 (Weeks 5–6): Fluid & Continuum Extensions

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| II.4 | Multiphase Flow | Phase-field Cahn-Hilliard in QTT format + Navier-Stokes coupling | $\partial_t\phi + \mathbf{u}\cdot\nabla\phi = M\nabla^2(\phi^3 - \phi - \varepsilon^2\nabla^2\phi)$; VOF advection; Rayleigh-Taylor benchmark | 1,400 |
| II.9 | Porous Media | Darcy solver on QTT grid + Richards unsaturated + Brinkman intermediate | $\mathbf{u} = -(k/\mu)\nabla p$; Buckley-Leverett multiphase; random permeability fields via TCI | 800 |
| II.10 | Free Surface | Level-set in QTT format + surface tension (CSF), Marangoni, capillary | $\partial_t\phi + \mathbf{u}\cdot\nabla\phi = 0$; $\Delta p = \gamma\kappa$; thin-film equation; contact angle dynamics | 1,000 |
| XIV.3 | Phase-Field | Cahn-Hilliard + Allen-Cahn on QTT grid | $\partial_t c = \nabla\cdot(M\nabla\mu)$, $\mu = f'(c) - \varepsilon^2\nabla^2 c$; dendritic solidification; spinodal decomposition benchmark | 900 |
| XVIII.1 | FSI | Partitioned fluid (NS-QTT) + solid (FEA-QTT), ALE interface | Fluid traction $\to$ solid load $\to$ mesh update; aeroelastic flutter; vortex-induced vibration; hemodynamic FSI | 1,200 |
| XVIII.2 | Thermo-Mechanical | Coupled thermal_qtt + fea-qtt | $\sigma = C(\varepsilon - \alpha\Delta T)$; thermal buckling eigenvalue; casting solidification stress; welding residual stress model | 700 |
| XVIII.3 | Electro-Mechanical | Coupled cem-qtt + fea-qtt | Piezoelectric: $\sigma = C\varepsilon - eE$, $D = e\varepsilon + \kappa E$; MEMS cantilever pull-in; electrostatic actuator | 700 |
| XVIII.6 | Radiation-Hydro | Flux-limited diffusion + Euler/NS coupling | $\partial_t E_r + \nabla\cdot\mathbf{F}_r = \kappa_a(aT^4 - E_r)$; implicit MC; grey/multigroup; ICF hohlraum benchmark | 1,000 |

**Sprint 1 subtotal**: 7,700 LOC, 8 new domains → 86/140

### Sprint 2 (Weeks 7–8): Quantum & Many-Body Extensions

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| VII.4 | Topological Phases | Toric code + Kitaev honeycomb as MPO; Chern number from Berry phase | $A_s = \prod_{j\in\text{star}(s)} Z_j$, $B_p = \prod_{j\in\partial p} X_j$; TEE extraction; anyonic braiding | 900 |
| VII.5 | MBL & Disorder | Random-field XXZ via DMRG; level statistics (Poisson↔GOE) | $H = J\sum S\cdot S + \sum h_i S^z_i$; participation ratio; entanglement entropy $\sim \log t$ vs $\sim t$; mobility edge | 700 |
| VII.9 | Kondo & Impurity | NRG (numerical RG) for Anderson impurity + CT-QMC | $H_{\text{AIM}} = \varepsilon_d n_d + U n_\uparrow n_\downarrow + \sum_k \varepsilon_k c^\dagger_k c_k + V_k(c^\dagger_k d + \text{h.c.})$; Kondo temperature $T_K$ extraction | 1,000 |
| VII.12 | Nuclear Many-Body | Nuclear shell model CI up to sd-shell; Richardson-Gaudin pairing | $(H - E)c = 0$ in Slater-determinant basis; nuclear CC singles+doubles; chiral EFT 2N+3N | 1,200 |
| VII.13 | Ultracold Atoms | Optical lattice Hamiltonian + Gross-Pitaevskii + Feshbach | $V(x) = V_0\sin^2(kx)$; BEC-BCS crossover; quantum gas microscope observables; synthetic gauge fields | 800 |
| VI.3 | Scattering | Partial-wave T-matrix; Born approx; R-matrix for reactive | $f(\theta) = \sum_l (2l+1)f_l P_l(\cos\theta)$; Breit-Wigner resonance; cross sections (elastic + inelastic) | 700 |
| VI.4 | Semiclassical/WKB | Eikonal solver, Tully surface hopping, Herman-Kluk | $\phi = \int p\cdot dx / \hbar$; Maslov index; FSSH hopping probability $g_{i\to j} = -2\text{Re}(a_j^*a_i d_{ji}\cdot\dot{R})/|a_i|^2$ | 600 |

**Sprint 2 subtotal**: 5,900 LOC, 7 new domains → 93/140

### Sprint 3 (Weeks 9–10): Applied & Cross-Cutting Extensions

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| XVII.3 | ML for Physics | PINNs on QTT, neural operator (FNO kernel), TT-decomposed weights | PINN loss: $\mathcal{L} = \lambda_{\text{PDE}}\|N[u_\theta]\|^2 + \lambda_{\text{data}}\|u_\theta - u_{\text{obs}}\|^2$; FNO: $v_{l+1} = \sigma(Wv_l + \mathcal{F}^{-1}(R\cdot\mathcal{F}(v_l)))$; SchNet-class NNP | 1,200 |
| XVII.4 | Mesh & AMR | Octree/quadtree with QTT indexing, Delaunay 2D/3D, h-adaptivity | Refinement criterion: $\|\nabla u\| > \theta_{\text{ref}}$; octree ↔ Morton Z-curve natural mapping; immersed boundary for complex geometries | 900 |
| XVI.3 | Membrane Bio | Coarse-grained lipid bilayer (MARTINI-class), electroporation | $V_{\text{memb}} = \sum V_{\text{bond}} + V_{\text{angle}} + V_{\text{LJ}}$; pore nucleation: $\Delta G(r) = 2\pi\gamma r - \pi r^2 \sigma_e$; channel gating | 600 |
| XX.4 | Robotics Physics | Recursive Newton-Euler, Featherstone ABA, LCP contact | $\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$; Coulomb friction cone; Cosserat rod (tendon/cable) | 800 |
| XI.4 | Gyrokinetics | 5D gyrokinetic Vlasov on QTT (extends existing 5D/6D Vlasov) | $\partial_t f + \dot{R}\cdot\nabla f + \dot{v}_\|\partial_{v_\|} f = C[f]$; ITG/TEM/ETG growth rates; tokamak turbulent transport | 1,200 |
| XI.5 | Magnetic Reconnection | Sweet-Parker + Petschek + collisionless kinetic | $v_{\text{in}}/v_A = S^{-1/2}$ (Sweet-Parker); plasmoid instability: $N \sim S^{3/8}$; guide field reconnection; X-line geometry | 800 |
| XI.7 | Dusty Plasmas | Yukawa OCP + dust-acoustic waves + grain charging | $V(r) = (Q^2/4\pi\varepsilon_0 r)e^{-r/\lambda_D}$; dust-acoustic $\omega^2 = \omega_{pd}^2 k^2/(k^2 + \lambda_D^{-2})$; dust crystal lattice | 500 |

**Sprint 3 subtotal**: 6,000 LOC, 7 new domains → 100/140

**Phase 1 total: ~19,600 LOC, 22 new domains → 100/140 (71.4%)**

---

## Phase 2 — High-Value New Domains: 22 More (Weeks 11–18)

Domains requiring new physics infrastructure but with high scientific/commercial value.

### Sprint 4 (Weeks 11–12): Electronic Structure & Quantum Chemistry

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| VIII.1 | DFT | Kohn-Sham SCF on real-space QTT grid | $[-\tfrac{\hbar^2}{2m}\nabla^2 + V_{\text{eff}}]\psi_i = \varepsilon_i\psi_i$; LDA ($\varepsilon_{xc}^{\text{LDA}}$), PBE ($\varepsilon_{xc}^{\text{GGA}}$); norm-conserving pseudopot; SCF mixer (Anderson/Pulay) | 2,000 |
| VIII.2 | Beyond-DFT | Hartree-Fock + MP2 + CCSD via MPS (quantum-chemistry DMRG) | $E_{\text{corr}}^{(2)} = -\sum_{ijab}\frac{|\langle ij\|ab\rangle|^2}{\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}$; CCSD $T = T_1 + T_2$; CASSCF active space | 1,800 |
| VIII.3 | Semi-Empirical/TB | DFTB (Slater-Koster), extended Hückel | $H_{\mu\nu} = \langle\mu|H_0 + V_{\text{rep}}|\nu\rangle$; SCC-DFTB charge self-consistency; TB+U for correlated | 800 |
| VIII.4 | Excited States | TDDFT (Casida + real-time), GW approximation | $\Omega F = \omega^2 F$ (Casida); $\Sigma = iGW$; BSE kernel $K = 2v - W$; optical spectrum | 1,200 |
| VIII.5 | Response Properties | DFPT for phonons/dielectric, polarizability | $\chi(\mathbf{q},\omega) = \chi_0 / (1 - v_c\chi_0)$ (RPA); NMR GIAO; IR dipole derivatives; Raman tensors | 800 |
| VIII.6 | Relativistic Elec | Scalar relativistic + SOC (ZORA) | $H_{\text{ZORA}} = V + \mathbf{p}\cdot\frac{c^2}{2c^2 - V}\mathbf{p}$; SOC: $H_{\text{SOC}} = \xi\mathbf{L}\cdot\mathbf{S}$; Douglas-Kroll-Hess 2nd order | 700 |
| VIII.7 | Quantum Embedding | QM/MM framework + DFT+DMFT interface | $E_{\text{tot}} = E_{\text{QM}} + E_{\text{MM}} + E_{\text{QM/MM}}$; projection-based embedding; ONIOM extrapolation | 700 |

**Sprint 4 subtotal**: 8,000 LOC, 7 new domains → 107/140

### Sprint 5 (Weeks 13–14): Nuclear, Particle & Astrophysics

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| X.1 | Nuclear Structure | m-scheme shell model CI + nuclear DFT (Skyrme) | $H = T + \sum_{i<j} V_{ij}^{NN}$; Skyrme: $E[\rho] = \int\mathcal{H}(\rho, \tau, \mathbf{J})d^3r$; collective rotor $E = \hbar^2 J(J+1)/2\mathcal{I}$ | 1,200 |
| X.2 | Nuclear Reactions | Optical model + Hauser-Feshbach compound nucleus | $U(r) = V_{\text{real}}(r) + iW(r)$ (Woods-Saxon); $\sigma_{a\to b} = \pi\lambda^2\sum_J (2J+1) T_a^J T_b^J / \sum_c T_c^J$; R-matrix | 1,000 |
| X.3 | Nuclear Astrophys | r-process network + NS EOS (poly/tabulated) | $dY_i/dt = \sum_j \lambda_j Y_j + \sum_{jk} \langle jk\rangle Y_j Y_k$; TOV + nuclear EOS; beta-decay rates | 1,000 |
| X.6 | Beyond SM | WIMP relic abundance + neutrino oscillation + leptogenesis | $\Omega h^2 \approx 3\times10^{-27}/\langle\sigma v\rangle$; $P(\nu_\alpha\to\nu_\beta) = |\sum_i U_{\alpha i}^* U_{\beta i} e^{-im_i^2 L/2E}|^2$; Sakharov conditions | 800 |
| XII.1 | Stellar Structure | Lane-Emden + nuclear network + opacity | $dP/dr = -G\rho m/r^2$; $dL/dr = 4\pi r^2\rho\varepsilon_{\text{nuc}}$; pp-chain, CNO, triple-α; mixing length convection | 1,200 |
| XII.3 | Gravitational Waves | EOB inspiral + BSSN merger + QNM ringdown | $h_{+,\times}(t)$ waveform; $\partial_t\tilde{\gamma}_{ij}$, $\partial_t\tilde{A}_{ij}$, $\partial_t K$ (BSSN); $\omega_{\text{QNM}} = \omega_R + i\omega_I$ | 1,400 |
| XII.4 | Cosmological Sims | Tree-PM N-body + Gadget-class SPH | $\mathbf{a}_i = -G\sum_j m_j(\mathbf{r}_i - \mathbf{r}_j)/|\mathbf{r}_i - \mathbf{r}_j|^3 + \mathbf{a}_{\text{PM}}$; halo finder (FOF + spherical overdensity); merger tree | 1,200 |
| XII.5 | CMB & Early Universe | Boltzmann hierarchy (photon, CDM, baryon) | $\dot{\Theta}_l + k/(2l+1)[(l+1)\Theta_{l+1} - l\Theta_{l-1}] = -\dot{\tau}[\Theta_l - \delta_{l0}\Theta_0]$; slow-roll inflation $\varepsilon = -\dot{H}/H^2$; recombination | 1,000 |
| XII.6 | Radiative Transfer | RT equation on QTT + MC radiative transfer | $dI/ds = -\kappa I + j$; Eddington approximation $F = -c/(3\kappa)\nabla E$; S_N discrete ordinates | 800 |

**Sprint 5 subtotal**: 9,600 LOC, 9 new domains → 116/140

### Sprint 6 (Weeks 15–16): Geophysics & Materials Science

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| XIII.1 | Seismology | 3D elastic wave on QTT grid + FWI adjoint | $\rho\ddot{\mathbf{u}} = \nabla\cdot\boldsymbol{\sigma} + \mathbf{f}$; P-wave: $c_p = \sqrt{(\lambda + 2\mu)/\rho}$; S-wave: $c_s = \sqrt{\mu/\rho}$; FWI misfit gradient via adjoint | 1,200 |
| XIII.2 | Mantle Convection | Boussinesq Stokes + T-dep viscosity on QTT | $\nabla\cdot\boldsymbol{\sigma} + \text{Ra}\,T\hat{z} = 0$; $\partial_t T + \mathbf{u}\cdot\nabla T = \nabla^2 T$; slab subduction; Rayleigh = $10^6$–$10^8$ | 900 |
| XIII.3 | Geodynamo | Rotating MHD in spherical shell | $\partial_t\mathbf{u} + 2\Omega\times\mathbf{u} = -\nabla p + \text{Ra}\,T\hat{r} + (\nabla\times\mathbf{B})\times\mathbf{B}/\mu_0 + \nu\nabla^2\mathbf{u}$; dipole/quadrupole ratio; reversal frequency | 1,000 |
| XIII.6 | Glaciology | Shallow ice approx + Glen's flow law + calving | $\partial_t H = -\nabla\cdot(\bar{\mathbf{u}}H) + a_s$; Glen: $\dot{\varepsilon} = A\tau^n$ ($n\approx 3$); grounding line flux $q_g$; calving law | 700 |
| XIV.1 | First-Principles Design | High-throughput screening + convex hull | Stability: $\Delta H_f(A_xB_y) < 0$ & on hull; phonon OK; elastic OK; DFT workflow automation; Pareto screening | 800 |
| XIV.4 | Microstructure | Potts MC grain growth + KWN precipitate | $\Delta E = -J\sum_{\langle ij\rangle}\delta_{s_i s_j}$ (Potts); nucleation $J = Z N_v \beta^* \exp(-\Delta G^*/k_BT)$; LSW coarsening $\bar{r}^3 \sim t$ | 700 |
| XIV.5 | Radiation Damage | BCA cascades + void KMC + rate theory | PKA: $T_{\max} = 4m_1m_2/(m_1+m_2)^2 \cdot E$; Kinchin-Pease NRT; dpa; void nucleation: $J \propto \exp(-\Delta G_v^*/k_BT)$ | 700 |
| XIV.6 | Polymers & Soft Matter | SCFT + Flory-Huggins + DPD | $\Delta G_{\text{mix}}/k_BT = n_1\ln\phi_1 + n_2\ln\phi_2 + \chi n_1\phi_2$; SCFT: $\delta F/\delta w = 0$; block copolymer morphology (LAM, HEX, BCC, GYR) | 800 |

**Sprint 6 subtotal**: 6,800 LOC, 8 new domains → 124/140

### Sprint 7 (Weeks 17–18): Chemical, Optics & Remaining Applied

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| XV.3 | Quantum React Dyn | Wavepacket propagation on PES (MCTDH-class) | $\Psi = \sum_J A_J \prod_\kappa \varphi_\kappa^{(J)}$; H + H₂ reactive scattering; cumulative reaction probability $N(E)$ | 900 |
| XV.4 | Nonadiabatic | Conical intersection + FSSH surface hopping | $g_{i\to j} = \max(0, -2\text{Re}(a_j^*a_i d_{ji}\cdot\dot{R})\Delta t / |a_i|^2)$; Berry phase around CI; ethylene photoisomerization benchmark | 700 |
| XV.7 | Spectroscopy | IR/Raman from Hessian + UV-Vis from TDDFT + NMR GIAO | $\alpha(\omega) \propto \text{Im}[\varepsilon(\omega)]$; IR intensity $\propto |d\mu/dQ|^2$; Raman $\propto |d\alpha/dQ|^2$; GIAO shielding tensor | 800 |
| IV.2 | Quantum Optics | Jaynes-Cummings + cavity QED + open-system master eq | $H = \omega_c a^\dagger a + \omega_a\sigma_z/2 + g(a^\dagger\sigma_- + a\sigma_+)$; photon blockade; squeezed states; photon statistics $g^{(2)}(\tau)$ | 800 |
| IV.3 | Laser Physics | Rate equations + gain saturation + BPM | $dN/dt = R_p - N/\tau - g(N)S$, $dS/dt = [\Gamma g(N) - 1/\tau_p]S + \beta N/\tau$; mode-locking (sech² pulse); semiconductor laser (LinTip) | 600 |
| IV.4 | Ultrafast Optics | Nonlinear Schrödinger (split-step Fourier) + HHG | $i\partial_z A + \beta_2/2\,\partial_t^2 A + \gamma|A|^2 A = 0$; SPM, GVD, soliton; three-step HHG cutoff $E_{\max} = I_p + 3.17 U_p$ | 700 |

**Sprint 7 subtotal**: 4,500 LOC, 6 new domains → 130/140

**Phase 2 total: ~28,900 LOC, 30 new domains → 130/140 (92.9%)**

---

## Phase 3 — Full Completeness: Final 10 Domains (Weeks 19–28)

Domains with lower QTT synergy or requiring specialized infrastructure. Two per sprint.

### Sprint 8 (Weeks 19–20)

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| III.2 | Magnetostatics | Biot-Savart integral + vector potential A on QTT | $\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}\times\hat{r}'}{r'^2}dV'$; $\nabla\times\mathbf{A} = \mathbf{B}$; inductance matrix; permanent magnet (BH curve) | 800 |
| III.4 | Frequency-Domain EM | Helmholtz FEM + waveguide eigenmode solver | $\nabla^2\mathbf{E} + k^2\mathbf{E} = 0$; rectangular/circular waveguide TE/TM modes; cavity Q-factor; Mie scattering | 900 |

### Sprint 9 (Weeks 21–22)

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| III.5 | EM Wave Propagation | Parabolic equation + ray tracing + fiber-mode BPM | Eikonal: $|\nabla S|^2 = n^2$; parabolic approx: $2ik\partial_z u + \nabla_\perp^2 u = 0$; fiber LP modes; ionospheric refraction | 800 |
| III.6 | Comp Photonics | PWE band structure + RCWA + metamaterial effective medium | Photonic bandgap: $\omega(k)$ from $\nabla\times[\varepsilon^{-1}\nabla\times\mathbf{H}] = (\omega/c)^2\mathbf{H}$; Drude metal: $\varepsilon(\omega) = 1 - \omega_p^2/(\omega^2 + i\gamma\omega)$; negative index | 1,000 |

### Sprint 10 (Weeks 23–24)

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| III.7 | Antenna/Microwave | MoM integral equation + array factor + Smith chart | $Z_{\text{in}} = Z_0(Z_L + jZ_0\tan\beta l)/(Z_0 + jZ_L\tan\beta l)$; $AF = \sum_n w_n e^{jk\hat{r}\cdot\mathbf{r}_n}$; patch antenna directivity; phased array scan | 900 |
| IX.2 | Band Structure | Bloch solver + Wannier90-class interpolation + BoltzTraP | $H_k u_{nk} = \varepsilon_{nk} u_{nk}$; Wannier: $|w_n(\mathbf{R})\rangle = \frac{V}{(2\pi)^3}\int e^{-ik\cdot R}|\psi_{nk}\rangle dk$; $\sigma_{\alpha\beta} = e^2\int \tau v_\alpha v_\beta (-\partial f_0/\partial\varepsilon) g(\varepsilon)d\varepsilon$ | 1,000 |

### Sprint 11 (Weeks 25–26)

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| IX.3 | Classical Magnetism | LLG micromagnetics + atomistic spin dynamics + skyrmion | $\partial_t\mathbf{M} = -\gamma\mathbf{M}\times\mathbf{H}_{\text{eff}} + \alpha/(M_s)\mathbf{M}\times\partial_t\mathbf{M}$; MAE: $E = K_1\sin^2\theta$; skyrmion number $Q = \frac{1}{4\pi}\int \mathbf{n}\cdot(\partial_x\mathbf{n}\times\partial_y\mathbf{n})dA$ | 1,000 |
| IX.6 | Surfaces & Interfaces | Slab model + adsorption (Langmuir) + surface states | $\theta = Kp/(1+Kp)$ (Langmuir); work function $\Phi = V_{\text{vac}} - E_F$; surface energy $\gamma_s = (E_{\text{slab}} - nE_{\text{bulk}})/2A$; Tamm/Shockley states | 700 |

### Sprint 12 (Weeks 27–28)

| # | Domain | Architecture | Key Equations | Est. LOC |
|:-:|--------|-------------|--------------|--------:|
| IX.8 | Ferroelectrics | Berry phase polarization + Landau-Devonshire + piezo tensor | $P_s = \frac{e}{V}\sum_n \int \langle u_{nk}|i\nabla_k|u_{nk}\rangle dk$ (modern theory); $F = \alpha P^2 + \beta P^4 + \gamma P^6$; $d_{ij}$ piezoelectric tensor | 800 |
| V.4 | Monte Carlo (General) | Variational/Diffusion QMC + AFQMC + Metropolis engine | VMC: $E[\Psi_T] = \langle\Psi_T|H|\Psi_T\rangle / \langle\Psi_T|\Psi_T\rangle$ via MC; DMC: $\phi(\mathbf{R},\tau) = e^{-\tau(H-E_T)}\phi$; sign problem mitigation; Jastrow-Slater trial | 1,200 |

**Phase 3 total: ~9,100 LOC, 10 new domains → 140/140**

---

## Implementation Standards (All Phases)

Every new solver module **must** include:

### Code Structure
```
tensornet/<domain>/<solver>.py    # or crates/<domain>/src/
├── Physics equations (LaTeX in docstrings)
├── QTT-native implementation (MPS/MPO format where applicable)
├── Reference parameters (physical constants, standard benchmarks)
├── Convergence validation (analytical solutions or published benchmarks)
├── Type hints (Python 3.10+ | Rust strong typing)
└── Error handling (no bare except, no silent failures)
```

### Validation Requirements
| Criterion | Threshold |
|-----------|-----------|
| Analytical benchmark agreement | < 1% relative error (where exact solutions exist) |
| Published benchmark reproduction | < 5% deviation from reference |
| Conservation law satisfaction | Machine precision ($< 10^{-12}$ relative drift) |
| QTT compression ratio | > 10× vs dense at $N \geq 2^{10}$ per dimension |
| Bond dimension scaling | Documented $\chi(N)$ or $\chi(\text{Re})$ |
| Test coverage | ≥ 1 unit test per public function |

### Platform Spec Integration
- Each new solver → new subsection under the appropriate §
- LaTeX equations, source file references, LOC, benchmark table
- Summary table row added
- Changelog entry

### Naming Convention
```
tensornet/<category>/          # Python domain modules
crates/<category>-qtt/         # Rust QTT-native crates
FRONTIER/<XX>_<NAME>/          # Frontier research projects
```

---

## Dependency Graph

```
Phase 0 (upgrades) ─→ Phase 1 (natural extensions) ─→ Phase 2 (new domains) ─→ Phase 3 (completeness)
         │                       │                              │
         │                       │                              │
     ┌───▼───────┐         ┌────▼────┐                   ┌────▼────┐
     │ FEA-QTT   │         │ Phase   │                   │ DFT     │
     │ CEM-QTT   │         │ Field   │                   │ engine  │
     │ OPT-QTT   │         │ engine  │                   │         │
     │ existing   │         │ (II.4,  │                   │ (VIII.1)│
     │ solvers    │         │  XIV.3) │                   │         │
     └───────────┘         └────┬────┘                   └────┬────┘
                                │                              │
                           ┌────▼────┐                   ┌────▼────┐
                           │ FSI     │                   │ Beyond  │
                           │(XVIII.1)│                   │ DFT     │
                           │coupling │                   │ (VIII.2-│
                           │ layer   │                   │  VIII.7)│
                           └─────────┘                   └─────────┘
```

### Critical Path
1. **Phase 0 Week 1** (I.3/I.4 continuum + structural) → **Phase 1 Sprint 1** (FSI, thermo-mech, electro-mech)
2. **Phase 0 Week 3** (VII.3 DMFT, VII.7 Lindblad) → **Phase 1 Sprint 2** (topological, MBL, Kondo)
3. **Phase 1 Sprint 4** (VIII.1 DFT) → **Phase 2** (VIII.2-VIII.7 all depend on DFT infrastructure)
4. **Phase 2 Sprint 5** (X.1 nuclear structure) → X.2 reactions, X.3 astrophysics (depend on nuclear shell model)
5. **Phase 2 Sprint 5** (XII.1 stellar) → XII.3 GW (depends on GR infrastructure from XX.2 upgrade)

### Parallelizable Tracks
- **Track A** (Fluids + Coupled): II.4, II.9, II.10, XIV.3, XVIII.1–3, XVIII.6
- **Track B** (Quantum): VII.4, VII.5, VII.9, VII.12, VII.13, VI.3, VI.4
- **Track C** (Electronic Structure): VIII.1–VIII.7 (sequential, DFT first)
- **Track D** (Astro/Nuclear/Geo): X, XII, XIII (partially parallel)
- **Track E** (EM/Optics): III.2, III.4–III.7, IV.2–IV.4
- **Track F** (Applied): XV.3–4, XV.7, IX.2–3, IX.6, IX.8, XIV.4–6, V.4

With 2 parallel developers: Tracks A+B → C+D → E+F compresses to ~20 weeks.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| DFT SCF convergence on QTT grid | Phase 2 blocker | Medium | Prototype adaptive-rank SCF in Phase 0; fall back to Gaussian basis if real-space QTT infeasible |
| Nuclear shell model CI explosion ($>10^{10}$ Slater dets) | X.1 infeasible | Medium | Limit to sd-shell ($^{28}$Si); use importance-truncated CI; leverage DMRG as CI solver |
| BSSN numerical relativity stability | XII.3 crashes | Medium | Use CCZ4 conformal Z4 variant with constraint damping; start with Brill wave test |
| Surface hopping (XV.4) stochastic noise | Low accuracy | Low | Use 10,000+ trajectories; Landau-Zener benchmark first |
| Scope creep from "full" coverage definition | Schedule overrun | High | Define "full" = ≥1 production solver + ≥1 benchmark per sub-domain; not exhaustive method coverage |
| QTT compression poor for certain domains | Performance | Medium | Identify during prototyping; allow dense fallback for inherently non-compressible problems |

---

## Success Criteria

### Per-Domain Acceptance
- [ ] Production solver implemented (not a stub/demo)
- [ ] ≥ 3 key equations documented with LaTeX
- [ ] ≥ 1 validation benchmark passes
- [ ] Source files in correct directory structure
- [ ] Platform Spec section written with equations + source refs
- [ ] Coverage assessment updated: ❌ → ✅

### Global Acceptance (140/140)
- [ ] All 140 sub-domains show ✅ in coverage assessment
- [ ] Platform Spec Summary table: **140 domains, ~1,029 equations, ~277,645 LOC, 450+ files**
- [ ] All 20 taxonomy categories at 100%
- [ ] No category below 100%
- [ ] `computational_physics_coverage_assessment.md` shows 140/140 (100%)
- [ ] All code committed and pushed to `workspace-reorg`
- [ ] CHANGELOG updated with version history

---

## Version Milestones

| Version | Milestone | Domains | Target Date |
|---------|-----------|:-------:|-------------|
| v36.0 | Current baseline | 78/140 (39%) | Feb 7, 2026 |
| v37.0 | Phase 0 complete — all partial → full | 78/140 (100% full) | Week 4 |
| v38.0 | Phase 1 complete — 100/140 | 100/140 (71%) | Week 10 |
| v39.0 | Phase 2 complete — 130/140 | 130/140 (93%) | Week 18 |
| v40.0 | **140/140 — Full Computational Physics Substrate** | **140/140 (100%)** | **Week 28** |

---

*Execution Plan for Tigantic Holdings LLC — 140/140 Computational Physics Capability Domain Substrate*
*© 2026 Brad McAllister. All rights reserved.*
