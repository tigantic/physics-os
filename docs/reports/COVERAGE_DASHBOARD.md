# HyperTensor-VM Coverage Dashboard

**Generated:** Phase 6 — Coupled Physics, Inverse, UQ & Optimization
**Total Nodes:** 167

## Summary

| State | Count | Pct |
|-------|-------|-----|
| V0.6 | 4 | 2.4% |
| V0.4 | 5 | 3.0% |
| V0.2 | 158 | 94.6% |
| V0.1 | 0 | 0.0% |
| V0.0 | 0 | 0.0% |
| **Total** | **167** | **100%** |

## Phase 5+6 Exit Gate

- **100% nodes at >= V0.2**: 167/167 = PASS
- **QTT-accelerated anchors at V0.6**: 4/4 = PASS (II.1, III.3, V.1, XI.1)
- **Tier A at >= V0.4**: 8/18 (anchors + cross-cutting)
- **All Tier A at >= V0.2**: 18/18 = PASS
- **Platform inverse/optimization modules**: PASS (ADR-0010)
- **Coupling orchestrator operational**: PASS (monolithic + partitioned)
- **UQ toolkit (MC, LHS, PCE)**: PASS
- **Lineage DAG**: PASS

## By Tier

| Tier | V0.6 | V0.4 | V0.2 | Total |
|------|------|------|------|-------|
| A | 4 | 4 | 10 | 18 |
| B | 0 | 1 | 145 | 146 |
| C | 0 | 0 | 3 | 3 |

## By Pack

| Pack | Name | Nodes | V0.6 | V0.4 | V0.2 |
|------|------|-------|------|------|------|
| I | Classical Mechanics | 8 | 0 | 0 | 8 |
| II | Fluid Dynamics | 10 | 1 | 0 | 9 |
| III | Electromagnetism | 7 | 1 | 0 | 6 |
| IV | Optics and Photonics | 7 | 0 | 0 | 7 |
| V | Thermodynamics and Statistical Mechanics | 6 | 1 | 0 | 5 |
| VI | Quantum Mechanics (Single/Few-Body) | 10 | 0 | 0 | 10 |
| VII | Quantum Many-Body Physics | 13 | 0 | 2 | 11 |
| VIII | Electronic Structure and Quantum Chemistry | 10 | 0 | 1 | 9 |
| IX | Solid State / Condensed Matter | 8 | 0 | 0 | 8 |
| X | Nuclear and Particle Physics | 8 | 0 | 0 | 8 |
| XI | Plasma Physics | 10 | 1 | 0 | 9 |
| XII | Astrophysics and Cosmology | 10 | 0 | 0 | 10 |
| XIII | Geophysics and Earth Science | 8 | 0 | 0 | 8 |
| XIV | Materials Science | 8 | 0 | 0 | 8 |
| XV | Chemical Physics and Reaction Dynamics | 8 | 0 | 0 | 8 |
| XVI | Biophysics and Computational Biology | 8 | 0 | 0 | 8 |
| XVII | Cross-Cutting Computational Methods | 6 | 0 | 2 | 4 |
| XVIII | Continuum Coupled Physics | 8 | 0 | 0 | 8 |
| XIX | Quantum Information and Computation | 8 | 0 | 0 | 8 |
| XX | Special and Applied Domains | 6 | 0 | 0 | 6 |

## Detailed Node List

| Node | Name | Pack | Tier | State |
|------|------|------|------|-------|
| PHY-I.1 | Newtonian Particle Dynamics | I | B | V0.2 |
| PHY-I.2 | Lagrangian/Hamiltonian Mechanics | I | B | V0.2 |
| PHY-I.3 | Continuum Mechanics | I | B | V0.2 |
| PHY-I.4 | Structural Mechanics | I | B | V0.2 |
| PHY-I.5 | Nonlinear Dynamics and Chaos | I | B | V0.2 |
| PHY-I.6 | Acoustics and Vibration | I | B | V0.2 |
| PHY-I.7 | PHY-I.7_Continuum_mechanics | I | B | V0.2 |
| PHY-I.8 | PHY-I.8_Chaos_theory | I | B | V0.2 |
| PHY-II.1 | Incompressible Navier-Stokes | II | A | V0.6 |
| PHY-II.2 | Compressible Flow | II | A | V0.2 |
| PHY-II.3 | Turbulence | II | A | V0.2 |
| PHY-II.4 | Multiphase Flow | II | A | V0.2 |
| PHY-II.5 | Reactive Flow / Combustion | II | A | V0.2 |
| PHY-II.6 | Rarefied Gas / Kinetic | II | B | V0.2 |
| PHY-II.7 | Shallow Water / Geophysical Fluids | II | B | V0.2 |
| PHY-II.8 | Non-Newtonian / Complex Fluids | II | B | V0.2 |
| PHY-II.9 | Porous Media | II | B | V0.2 |
| PHY-II.10 | Free Surface / Interfacial | II | B | V0.2 |
| PHY-III.1 | Electrostatics | III | A | V0.2 |
| PHY-III.2 | Magnetostatics | III | A | V0.2 |
| PHY-III.3 | Full Maxwell Time-Domain | III | A | V0.6 |
| PHY-III.4 | Frequency-Domain EM | III | A | V0.2 |
| PHY-III.5 | Wave Propagation | III | B | V0.2 |
| PHY-III.6 | Computational Photonics | III | B | V0.2 |
| PHY-III.7 | Antennas and Microwaves | III | B | V0.2 |
| PHY-IV.1 | Physical Optics | IV | B | V0.2 |
| PHY-IV.2 | Quantum Optics | IV | B | V0.2 |
| PHY-IV.3 | Laser Physics | IV | B | V0.2 |
| PHY-IV.4 | Ultrafast Optics | IV | B | V0.2 |
| PHY-IV.5 | PHY-IV.5_Nonlinear_optics | IV | B | V0.2 |
| PHY-IV.6 | PHY-IV.6_Quantum_optics | IV | B | V0.2 |
| PHY-IV.7 | PHY-IV.7_Photonic_crystal | IV | B | V0.2 |
| PHY-V.1 | Equilibrium Statistical Mechanics | V | B | V0.6 |
| PHY-V.2 | Non-Equilibrium Statistical Mechanics | V | B | V0.2 |
| PHY-V.3 | Molecular Dynamics | V | B | V0.2 |
| PHY-V.4 | Monte Carlo Methods | V | B | V0.2 |
| PHY-V.5 | Heat Transfer | V | A | V0.2 |
| PHY-V.6 | Lattice Models and Spin Systems | V | B | V0.2 |
| PHY-VI.1 | Time-Independent Schrodinger | VI | B | V0.2 |
| PHY-VI.2 | TDSE Propagation | VI | B | V0.2 |
| PHY-VI.3 | Scattering Theory | VI | B | V0.2 |
| PHY-VI.4 | Semiclassical Methods | VI | B | V0.2 |
| PHY-VI.5 | Path Integrals | VI | B | V0.2 |
| PHY-VI.6 | PHY-VI.6_Strongly_correlated | VI | B | V0.2 |
| PHY-VI.7 | PHY-VI.7_Mesoscopic_physics | VI | B | V0.2 |
| PHY-VI.8 | PHY-VI.8_Surface_physics | VI | B | V0.2 |
| PHY-VI.9 | PHY-VI.9_Disordered_systems | VI | B | V0.2 |
| PHY-VI.10 | PHY-VI.10_Phase_transitions | VI | B | V0.2 |
| PHY-VII.1 | Tensor Network Methods | VII | A | V0.4 |
| PHY-VII.2 | Quantum Spin Systems | VII | A | V0.4 |
| PHY-VII.3 | Strongly Correlated Electrons | VII | B | V0.2 |
| PHY-VII.4 | Topological Phases | VII | B | V0.2 |
| PHY-VII.5 | MBL and Disorder | VII | B | V0.2 |
| PHY-VII.6 | Lattice Gauge Theory | VII | B | V0.2 |
| PHY-VII.7 | Open Quantum Systems | VII | B | V0.2 |
| PHY-VII.8 | Non-Equilibrium Quantum Dynamics | VII | B | V0.2 |
| PHY-VII.9 | Quantum Impurity | VII | B | V0.2 |
| PHY-VII.10 | Bosonic Many-Body | VII | B | V0.2 |
| PHY-VII.11 | Fermionic Systems | VII | B | V0.2 |
| PHY-VII.12 | Nuclear Many-Body | VII | B | V0.2 |
| PHY-VII.13 | Ultracold Atoms | VII | B | V0.2 |
| PHY-VIII.1 | DFT | VIII | A | V0.4 |
| PHY-VIII.2 | Beyond-DFT Correlated Methods | VIII | B | V0.2 |
| PHY-VIII.3 | Semi-Empirical and Tight-Binding | VIII | B | V0.2 |
| PHY-VIII.4 | Excited States | VIII | B | V0.2 |
| PHY-VIII.5 | Response Properties | VIII | B | V0.2 |
| PHY-VIII.6 | Relativistic Electronic Structure | VIII | B | V0.2 |
| PHY-VIII.7 | Quantum Embedding | VIII | B | V0.2 |
| PHY-VIII.8 | Response_Functions | VIII | B | V0.2 |
| PHY-VIII.9 | Band_Structure | VIII | B | V0.2 |
| PHY-VIII.10 | Ab_Initio_MD | VIII | B | V0.2 |
| PHY-IX.1 | Phonons and Lattice Dynamics | IX | B | V0.2 |
| PHY-IX.2 | Band Structure and Transport | IX | B | V0.2 |
| PHY-IX.3 | Magnetism | IX | B | V0.2 |
| PHY-IX.4 | Superconductivity | IX | B | V0.2 |
| PHY-IX.5 | Disordered Systems | IX | B | V0.2 |
| PHY-IX.6 | Surfaces and Interfaces | IX | B | V0.2 |
| PHY-IX.7 | Defects in Solids | IX | B | V0.2 |
| PHY-IX.8 | Ferroelectrics and Multiferroics | IX | B | V0.2 |
| PHY-X.1 | Nuclear Structure | X | B | V0.2 |
| PHY-X.2 | Nuclear Reactions | X | B | V0.2 |
| PHY-X.3 | Nuclear Astrophysics | X | B | V0.2 |
| PHY-X.4 | Lattice QCD | X | B | V0.2 |
| PHY-X.5 | Perturbative QFT | X | B | V0.2 |
| PHY-X.6 | Beyond Standard Model | X | B | V0.2 |
| PHY-X.7 | PHY-X.7_Dark_matter | X | B | V0.2 |
| PHY-X.8 | PHY-X.8_Neutrino_physics | X | B | V0.2 |
| PHY-XI.1 | Ideal MHD | XI | A | V0.6 |
| PHY-XI.2 | Resistive/Extended MHD | XI | A | V0.2 |
| PHY-XI.3 | Kinetic Theory Plasma | XI | A | V0.2 |
| PHY-XI.4 | Gyrokinetics | XI | B | V0.2 |
| PHY-XI.5 | Magnetic Reconnection | XI | B | V0.2 |
| PHY-XI.6 | Laser-Plasma Interaction | XI | B | V0.2 |
| PHY-XI.7 | Dusty Plasmas | XI | B | V0.2 |
| PHY-XI.8 | Space and Astrophysical Plasma | XI | B | V0.2 |
| PHY-XI.9 | Ion_Acoustic_Waves | XI | B | V0.2 |
| PHY-XI.10 | Plasma_Instabilities | XI | B | V0.2 |
| PHY-XII.1 | Stellar Structure and Evolution | XII | B | V0.2 |
| PHY-XII.2 | Compact Objects | XII | B | V0.2 |
| PHY-XII.3 | Gravitational Waves | XII | B | V0.2 |
| PHY-XII.4 | Cosmological Simulations | XII | B | V0.2 |
| PHY-XII.5 | CMB and Early Universe | XII | C | V0.2 |
| PHY-XII.6 | Radiative Transfer Astrophysical | XII | B | V0.2 |
| PHY-XII.7 | PHY-XII.7_Accretion | XII | B | V0.2 |
| PHY-XII.8 | PHY-XII.8_Radiation_transport | XII | B | V0.2 |
| PHY-XII.9 | PHY-XII.9_Dark_energy | XII | B | V0.2 |
| PHY-XII.10 | PHY-XII.10_CMB | XII | B | V0.2 |
| PHY-XIII.1 | Seismology | XIII | B | V0.2 |
| PHY-XIII.2 | Mantle Convection | XIII | B | V0.2 |
| PHY-XIII.3 | Geomagnetism and Dynamo | XIII | B | V0.2 |
| PHY-XIII.4 | Atmospheric Physics | XIII | B | V0.2 |
| PHY-XIII.5 | Oceanography | XIII | B | V0.2 |
| PHY-XIII.6 | Glaciology | XIII | B | V0.2 |
| PHY-XIII.7 | PHY-XIII.7_Volcanology | XIII | B | V0.2 |
| PHY-XIII.8 | PHY-XIII.8_Geodesy | XIII | B | V0.2 |
| PHY-XIV.1 | First-Principles Materials Design | XIV | C | V0.2 |
| PHY-XIV.2 | Mechanical Properties | XIV | B | V0.2 |
| PHY-XIV.3 | Phase-Field Methods | XIV | B | V0.2 |
| PHY-XIV.4 | Microstructure Evolution | XIV | B | V0.2 |
| PHY-XIV.5 | Radiation Damage | XIV | B | V0.2 |
| PHY-XIV.6 | Polymers and Soft Matter | XIV | B | V0.2 |
| PHY-XIV.7 | Ceramics and High-Temperature Materials | XIV | B | V0.2 |
| PHY-XIV.8 | PHY-XIV.8_Cell_signaling | XIV | B | V0.2 |
| PHY-XV.1 | Potential Energy Surfaces | XV | B | V0.2 |
| PHY-XV.2 | Reaction Rate Theory | XV | B | V0.2 |
| PHY-XV.3 | Quantum Reaction Dynamics | XV | B | V0.2 |
| PHY-XV.4 | Nonadiabatic Dynamics | XV | B | V0.2 |
| PHY-XV.5 | Photochemistry | XV | B | V0.2 |
| PHY-XV.6 | Catalysis | XV | B | V0.2 |
| PHY-XV.7 | Spectroscopy | XV | B | V0.2 |
| PHY-XV.8 | PHY-XV.8_Combustion | XV | B | V0.2 |
| PHY-XVI.1 | Protein Structure and Dynamics | XVI | B | V0.2 |
| PHY-XVI.2 | Drug Design and Binding | XVI | C | V0.2 |
| PHY-XVI.3 | Membrane Biophysics | XVI | B | V0.2 |
| PHY-XVI.4 | Nucleic Acids | XVI | B | V0.2 |
| PHY-XVI.5 | Systems Biology | XVI | B | V0.2 |
| PHY-XVI.6 | Neuroscience Computational | XVI | B | V0.2 |
| PHY-XVI.7 | PHY-XVI.7_Metamaterials | XVI | B | V0.2 |
| PHY-XVI.8 | PHY-XVI.8_Phase-field_modeling | XVI | B | V0.2 |
| PHY-XVII.1 | Optimization | XVII | A | V0.4 |
| PHY-XVII.2 | Inverse Problems | XVII | A | V0.4 |
| PHY-XVII.3 | ML for Physics | XVII | B | V0.2 |
| PHY-XVII.4 | Mesh Generation and Adaptivity | XVII | B | V0.2 |
| PHY-XVII.5 | Linear Algebra Large-Scale | XVII | B | V0.2 |
| PHY-XVII.6 | High-Performance Computing | XVII | B | V0.2 |
| PHY-XVIII.1 | Fluid-Structure Interaction | XVIII | B | V0.2 |
| PHY-XVIII.2 | Thermo-Mechanical Coupling | XVIII | B | V0.2 |
| PHY-XVIII.3 | Electro-Mechanical Coupling | XVIII | B | V0.2 |
| PHY-XVIII.4 | Magnetohydrodynamics Coupled | XVIII | B | V0.2 |
| PHY-XVIII.5 | Chemically Reacting Flows | XVIII | B | V0.2 |
| PHY-XVIII.6 | Radiation-Hydrodynamics | XVIII | B | V0.2 |
| PHY-XVIII.7 | Multiscale Methods | XVIII | B | V0.2 |
| PHY-XVIII.8 | PHY-XVIII.8_Data_assimilation | XVIII | B | V0.2 |
| PHY-XIX.1 | Quantum Circuit Simulation | XIX | B | V0.2 |
| PHY-XIX.2 | Quantum Error Correction | XIX | B | V0.2 |
| PHY-XIX.3 | Quantum Algorithms | XIX | B | V0.2 |
| PHY-XIX.4 | Quantum Simulation | XIX | B | V0.2 |
| PHY-XIX.5 | Quantum Cryptography and Communication | XIX | B | V0.2 |
| PHY-XIX.6 | PHY-XIX.6_Quantum_sensing | XIX | B | V0.2 |
| PHY-XIX.7 | PHY-XIX.7_Quantum_simulation | XIX | B | V0.2 |
| PHY-XIX.8 | PHY-XIX.8_Quantum_cryptography | XIX | B | V0.2 |
| PHY-XX.1 | Relativistic Mechanics | XX | B | V0.2 |
| PHY-XX.2 | General Relativity Numerical | XX | B | V0.2 |
| PHY-XX.3 | Astrodynamics | XX | B | V0.2 |
| PHY-XX.4 | Robotics Physics | XX | B | V0.2 |
| PHY-XX.5 | Acoustics Applied | XX | B | V0.2 |
| PHY-XX.6 | Biomedical Engineering | XX | B | V0.2 |