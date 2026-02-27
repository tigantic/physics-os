# Repository Inventory

> **Generated**: 2026-02-27 | **Tracked files**: ~10,364 | **Languages**: 20+ | **First-party LOC**: ~1.84M
>
> Comprehensive index of every application, library, experiment, proof, product,
> tool, and infrastructure component in the HyperTensor-VM monorepo.

---

## Table of Contents

| § | Section | Scope |
|---|---------|-------|
| 1 | [The Civilization Stack](#1-the-civilization-stack) | 20 grand R&D projects + validation gauntlets |
| 2 | [TensorNet Core Library](#2-tensornet-core-library) | 16 physics modules, ~310K LOC, ~765 files |
| 3 | [Applications](#3-applications) | 15 deployed apps in `apps/` |
| 4 | [Products](#4-products) | Shippable products in `products/` |
| 5 | [Experiments](#5-experiments) | Frontier physics, demos, research |
| 6 | [Validation Gauntlets](#6-validation-gauntlets) | 38 gauntlets, 37K LOC |
| 7 | [Tools & Scripts](#7-tools--scripts) | 208 files across `tools/scripts/` |
| 8 | [Proofs & Certificates](#8-proofs--certificates) | Trustless Physics Certificates, ZK proofs |
| 9 | [Rust Crates](#9-rust-crates) | 15 crates, ~204K LOC |
| 10 | [Infrastructure & Deployment](#10-infrastructure--deployment) | Deploy, contracts, integrations |
| 11 | [HyperTensor Platform Service](#11-hypertensor-platform-service) | API, billing, CLI, MCP, SDK |
| 12 | [Test Suite](#12-test-suite) | 104 test files, ~52K LOC |
| 13 | [Documentation & Governance](#13-documentation--governance) | docs/, ADRs, governance, specs |
| 14 | [Backward-Compatibility Shim Map](#14-backward-compatibility-shim-map) | Re-export shims under `tensornet/` |
| 15 | [Data & Archive](#15-data--archive) | Datasets, cached models, archived artifacts |

---

## Grand Summary

| Area | Files | LOC | Primary Language |
|------|-------|-----|-----------------|
| `tensornet/` | 1,192 | ~492K | Python |
| `apps/` | 207 | ~84K | Python + Rust |
| `products/` | 154 | ~78K | Python |
| `experiments/` | 2,623 | ~442K | Python |
| `tools/` | 200 | ~110K | Python |
| `tests/` | 115 | ~52K | Python |
| `proofs/` | 7,473 | ~371K | Python + JSON |
| `crates/` | 716 | ~204K | Rust |
| `contracts/` | 3 | ~675 | Solidity |
| `hypertensor/` | 31 | ~3.9K | Python |
| `integrations/` | 1 | ~518 | C# / C++ |
| **Total** | **~12,715** | **~1.84M** | |

---

## 1. The Civilization Stack

Twenty interlocking grand projects — from tokamak plasma control through
self-replicating civilizational seeds — each backed by a dedicated validation
gauntlet in `tools/scripts/gauntlets/`.

| # | Project | Domain | Key Metric | Confidence | Gauntlet |
|---|---------|--------|------------|------------|----------|
| 1 | **TOMAHAWK** | Aerospace / Tokamak | 49,091× CFD compression | Solid Physics | `tomahawk_cfd_gauntlet.py` (823 LOC) |
| 2 | **TIG-011a** | Oncology Drug Design | ΔG = −13.7 kcal/mol | Validated | 6 gauntlets (5,751 LOC) |
| 3 | **SnHf-F** | EUV Lithography | 0.42 nm blur | Plausible | `snhff_stochastic_gauntlet.py` (766 LOC) |
| 4 | **Li₃InCl₄.₈Br₁.₂** | Superionic Electrolyte | 112 S/cm conductivity | Lottery Ticket | `li3incl48br12_superionic_gauntlet.py` (932 LOC) |
| 5 | **LaLuH₆ ODIN** | Room-Temp Superconductor | Tc = 306 K | Lottery Ticket | `laluh6_odin_gauntlet.py` (1,083 LOC) |
| 6 | **HELL-SKIN** | Hypersonic TPS | MP = 4,005 °C | Solid Physics | `hellskin_gauntlet.py` (1,075 LOC) |
| 7 | **STAR-HEART** | Compact Fusion | Q = 14.1 | Lottery Ticket | `starheart_gauntlet.py` (1,177 LOC) |
| 8 | **Dynamics Engine** | Physics Core | Langevin / MHD stability | Solid Physics | *(core engine)* |
| 9 | **QTT Brain** | Neuromorphic | 490 T synapses → 13,660 params | Plausible | *(core engine)* |
| 10 | **Neuromorphic Chip** | Compute | 70 B neurons @ 0.06 W | Plausible | *(HAL backend)* |
| 11 | **Femto-Fabricator** | Molecular Mfg | 0.016 Å placement | Plausible | `femto_fabricator_gauntlet.py` (1,202 LOC) |
| 12 | **Proteome Compiler** | Synthetic Biology | 712 params → 20 K proteins | Plausible | `proteome_compiler_gauntlet.py` (1,111 LOC) |
| 13 | **Metric Engine** | Propulsion | Non-propulsive drive | Lottery Ticket | `metric_engine_gauntlet.py` (945 LOC) |
| 14 | **PROMETHEUS** | Consciousness / IIT | EI = 2.54 bits | Plausible | `prometheus_gauntlet.py` (1,707 LOC) |
| 15 | **ORACLE** | Quantum Computing | 255× thermal advantage | Lottery Ticket | `oracle_gauntlet.py` (791 LOC) |
| 16 | **ORBITAL FORGE** | Space Infrastructure | 500 km station, 50 crew | Solid Physics | `orbital_forge_gauntlet.py` (1,192 LOC) |
| 17 | **HERMES** | Interstellar Comms | 1 M ly beacon | Solid Physics | `hermes_gauntlet.py` (1,276 LOC) |
| 18 | **CORNUCOPIA** | Post-Scarcity Econ | $0.008 / kWh | Solid Physics | `cornucopia_gauntlet.py` (1,180 LOC) |
| 19 | **CHRONOS** | Temporal Physics | GPS 38.5 μs/day | Solid Physics | `chronos_gauntlet.py` (1,246 LOC) |
| 20 | **SOVEREIGN GENESIS** | Civilizational Autarchy | Self-replicating seed | Sum of All | `sovereign_genesis_gauntlet.py` (1,022 LOC) |

### TIG-011a Gauntlet Detail

The oncology drug candidate has the deepest validation suite (6 dedicated files, 5,751 LOC):

| Gauntlet | LOC | Focus |
|----------|-----|-------|
| `tig011a_multimechanism.py` | 2,377 | Coulombic, LJ, hydrophobic, π-π, covalent warhead |
| `tig011a_dynamic_validation.py` | 1,093 | 500 ns MD, H-bond persistence, FEP |
| `tig011a_docking_qmmm.py` | 935 | Multi-pose sampling, QM/MM scoring |
| `tig011a_dielectric_gauntlet.py` | 605 | KRAS G12D binding across ε = 4–80 |
| `tig011a_wiggle_tt.py` | 403 | N-methyl piperazine variant energy well |
| `tig011a_tox_screen.py` | 398 | PAINS, Lipinski, hERG, CYP450, Ames |
| `tig011a_attestation.py` | 340 | Cryptographic hash attestation |

---

## 2. TensorNet Core Library

`tensornet/` — **1,192 files, ~492K LOC** — the physics engine of HyperTensor-VM.

Organized into 16 major modules. Each module may contain STARK trace adapters
for Trustless Physics Certificate generation.

### 2.1 `tensornet/cfd/` — Computational Fluid Dynamics

**115 files, ~78,200 LOC** — The flagship CFD engine. QTT-native solvers from
1D Euler through 6D Vlasov-Poisson, all operating at O(log N) complexity.

| Sub-area | Key Files | Description |
|----------|-----------|-------------|
| **Euler solvers** | `euler_1d.py`, `euler_2d.py`, `euler_3d.py`, `euler2d_native.py`, `euler_nd_native.py`, `fast_euler_2d.py`, `fast_euler_3d.py` | 1D through N-D compressible Euler |
| **Navier-Stokes** | `ns_2d.py`, `ns_3d.py`, `ns2d_qtt_native.py`, `ns3d_native.py`, `ns3d_qtt_native.py`, `ns3d_realtime.py`, `ns3d_turbo.py`, `navier_stokes.py`, `viscous.py` | Incompressible / compressible NS, real-time capable |
| **QTT core ops** | `pure_qtt_ops.py` (1,071), `qtt_native_ops.py` (2,098), `qtt_turbo.py` (1,465), `nd_shift_mpo.py` (859), `qtt.py`, `qtt_cfd.py`, `qtt_hadamard.py`, `qtt_reciprocal.py`, `qtt_shift_stable.py` | Native QTT arithmetic — no dense fallback |
| **QTT-TCI** | `qtt_tci.py` (1,270), `tci_true.py` (841), `tci_flux.py` (871), `qtt_tci_gpu.py`, `flux_2d_tci.py`, `flux_batch.py` | Tensor Cross Interpolation construction + flux |
| **QTT spectral** | `qtt_fft.py` (897), `qtt_spectral.py`, `poisson_spectral.py` (1,047), `turbo_spectral.py` | Spectral methods in QTT format |
| **QTT time integration** | `qtt_tdvp.py` (786), `qtt_imex.py` (662), `implicit.py` | TDVP, IMEX, implicit schemes |
| **Triton GPU kernels** | `qtt_triton_kernels.py` (1,441), `qtt_triton_kernels_v2.py` (2,260), `qtt_triton_native.py` (1,472), `qtt_triton_ops.py` (1,597), `qtt_triton.py`, `triton_qtt3d.py`, `triton_qtt_kernels.py` | Production Triton kernel suite |
| **Turbulence** | `turbulence.py`, `les.py` (1,010), `hybrid_les.py`, `kolmogorov_spectrum.py`, `turbulence_forcing.py`, `turbulence_simulation.py`, `turbulence_validation.py`, `turbulence_qtt_benchmark.py` | RANS, LES, DNS validation |
| **Singularity / blow-up** | `singularity_hunter.py`, `hou_luo_ansatz.py`, `self_similar.py`, `newton_refine.py`, `stabilized_refine.py`, `kantorovich.py`, `adjoint_blowup.py` | NS millennium-problem numerical exploration |
| **Kinetic / Vlasov** | `fast_vlasov_5d.py` | 5D Vlasov-Poisson via ND shift MPO |
| **Koopman** | `koopman_tt.py` (1,645) | TT Koopman operator for turbulence |
| **Numerics** | `weno.py`, `weno_native_tt.py`, `weno_tt.py`, `dg.py`, `sem.py`, `space_time_dg.py`, `limiters.py`, `godunov.py` | WENO/TENO, DG, SEM, Riemann solvers |
| **Speciality** | `dsmc.py` (1,056), `lbm.py`, `sph.py`, `non_newtonian.py`, `reactive_ns.py`, `combustion_dns.py`, `real_gas.py`, `plasma.py`, `chemistry.py` | Rarefied gas, LBM, SPH, reacting flow, real gas |
| **HVAC bridge** | `comfort_metrics.py`, `thermal_qtt.py` | PMV/PPD comfort, thermal transport for HVAC |
| **Infrastructure** | `adaptive_tt.py`, `analytical_qtt.py`, `boundaries.py`, `chi_diagnostic.py`, `differentiable.py`, `geometry.py`, `morton_3d.py`, `multi_objective.py`, `optimization.py`, `qtt_2d.py`, `qtt_2d_shift.py`, `qtt_2d_shift_native.py`, `qtt_3d_state.py`, `qtt_batched_ops.py`, `qtt_checkpoint_stream.py`, `qtt_eval.py`, `qtt_multiscale.py`, `qtt_regularity.py`, `qtt_streaming.py`, `tci_benchmark_suite.py` | AMR, Morton curves, checkpointing, diagnostics |
| **Trace adapters** | 5 files | STARK adapters for Euler3D, NS-IMEX, Heat, Vlasov |

### 2.2 `tensornet/genesis/` — QTT Meta-Primitive Expansion Protocol

**76 files, ~38,000 LOC** — Seven QTT-native mathematical primitives plus
aging simulator and benchmarks.

| Sub-module | Files | LOC | Description |
|------------|-------|-----|-------------|
| `ot/` — Optimal Transport | 8 | ~5,400 | Sinkhorn in QTT, Wasserstein barycenters, transport plans |
| `topology/` — Persistent Homology | 7 | ~2,900 | Boundary operators, persistence, simplicial complexes |
| `sgw/` — Spectral Graph Wavelets | 7 | ~2,800 | Chebyshev approx, filter banks, graph Laplacian |
| `rkhs/` — Kernel Methods | 7 | ~3,400 | Gaussian processes, kernel ridge, MMD |
| `rmt/` — Random Matrix Theory | 7 | ~2,700 | GOE/GUE/GSE ensembles, free probability, universality |
| `ga/` — Geometric Algebra | 8 | ~3,400 | CGA, multivectors, rotors, QTT-compressed GA |
| `tropical/` — Tropical Geometry | 8 | ~4,000 | Semiring algebra, shortest paths, QTT-native tropical ops |
| `aging/` — Biological Aging | 8 | ~5,300 | Cell state dynamics, epigenetic clock, interventions, topology |
| `core/` | 8 | ~3,300 | Exceptions, logging, profiling, rSVD, Triton ops, validation |
| `fusion/` | 3 | ~2,400 | Genesis fusion demos, geometric types pipeline |
| `demos/` | 7 | ~2,900 | GPU QTT, hierarchical compression, NOAA petabyte, 1 TB stream |
| `benchmarks/` | 2 | ~1,000 | "GENESIS vs The World" massacre benchmark |

### 2.3 `tensornet/ml/discovery/` — Autonomous Discovery Engine

**42 files, ~23,500 LOC** — Chains Genesis primitives into autonomous
scientific discovery pipelines with API, connectors, and production hardening.

| Sub-module | Files | Description |
|------------|-------|-------------|
| Core | 8 | Engine orchestrator (v1 + v2), config, findings, pipelines, protocol |
| `api/` | 5 | FastAPI server, distributed API, GPU API, Pydantic models |
| `connectors/` | 7 | Coinbase L2, Ethereum, fusion, historical, molecular PDB, streaming |
| `hypothesis/` | 2 | Automated hypothesis generation |
| `ingest/` | 5 | DeFi, markets, molecular, plasma data ingestion |
| `pipelines/` | 5 | DeFi, markets, molecular, plasma discovery pipelines |
| `primitives/` | 7 | GA, kernel, OT, RMT, SGW, topology discovery primitives |
| `production/` | 5 | Observability, performance, resilience, security (1,016 LOC) |

### 2.4 `tensornet/aerospace/` — Aerospace, Defense & Exploit Engine

**56 files, ~26,700 LOC**

| Sub-module | Files | LOC | Description |
|------------|-------|-----|-------------|
| `guidance/` | 6 | ~3,600 | 6-DOF trajectory, bank-to-turn, proportional nav, kill-vehicle divert, TRN, comms blackout |
| `autonomy/` | 5 | ~2,600 | Mission planning, decision making, A*/RRT path planning, obstacle avoidance |
| `defense/` | 4 | ~1,200 | Hydroacoustic warfare, 6-DOF ballistics, Munk sound speed, FDTD sonar |
| `racing/` | 2 | ~350 | F1 dirty-air wake tracker |
| `exploit/` | 38 | ~19,000 | QTT-based DeFi vulnerability hunting — Compound V3, EigenLayer, Ethena, Euler V2, Lido, Morpho Blue, Pendle, Renzo, Usual, Cairo ZK circuits, Koopman structural exploits, historical validator, Immunefi bounty integration |

### 2.5 `tensornet/quantum/` — Quantum Physics

**99 files, ~22,500 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| `condensed_matter/` | 20 solvers + 20 adapters | Band structure, bosonic/fermionic, Kondo, MBL, NEGF, open quantum, phonons, topological phases, ultracold atoms |
| `electronic_structure/` | 7 solvers + 7 adapters | DFT, beyond-DFT (GW, DMFT), embedding, excited states, relativistic, tight-binding |
| `qft/` | 3 solvers + 3 adapters | Lattice QCD, lattice QFT, perturbative QFT |
| `qm/` | 2 solvers + 2 adapters | Scattering, semiclassical WKB |
| `quantum_mechanics/` | 3 solvers + 3 adapters | Path integrals, propagator, stationary states |
| `statmech/` | 3 solvers + 5 adapters | Equilibrium, Monte Carlo, non-equilibrium |
| Root | 7 | Error mitigation, hybrid classical-quantum, QTT renderers (CPU, GLSL, PyTorch) |

### 2.6 `tensornet/engine/` — Execution Engine, GPU, VM

**93 files, ~36,050 LOC**

| Sub-module | Files | LOC | Description |
|------------|-------|-----|-------------|
| `vm/` | 29 | ~6,500 | QTT Physics VM — IR, operators, runtime (CPU + GPU), telemetry, antenna design, PDE compilers (Maxwell, NS, Schrödinger, Vlasov-Poisson), post-processing |
| `gpu/` | 16 | ~5,400 | Advection, Blackwell opts, fluid dynamics, HIL real-time, kernel autotune, managed memory, mixed precision, multi-GPU TN, NVLink topology, persistent kernels, tensor core |
| `hardware/` | 9 | ~3,200 | Unified HAL — ARM SVE, FPGA, Apple Metal, neuromorphic, Intel oneAPI, photonic, quantum backend, ROCm/HIP |
| `distributed/` | 6 | ~3,000 | MPI/NCCL comms, domain decomposition, GPU manager, parallel solver, scheduler |
| `distributed_tn/` | 5 | ~2,200 | Distributed DMRG, load balancer, MPS ops, parallel TEBD |
| `adaptive/` | 4 | ~2,400 | Bond dimension optimizer, adaptive compression, entanglement analysis |
| `realtime/` | 5 | ~2,700 | Inference engine, kernel fusion, latency optimizer, memory manager |
| `substrate/` | 6 | ~2,500 | Field Oracle API, FieldBundle, Morton ops, frame budget, stats |
| `gateway/` | 6 | ~2,100 | OPERATION VALHALLA Phase 4 — modular grid, onion renderer, orbital command |
| `fuel/` | 3 | ~600 | S3 fetcher, tile compositor |
| `hw/` | 3 | ~1,700 | Verilog/SystemVerilog security analyzer, Yosys netlist analyzer (v1 + v2) |

### 2.7 `tensornet/em/` — Electromagnetics

**24 files, ~15,300 LOC**

| File | LOC | Description |
|------|-----|-------------|
| `chu_limit.py` | 3,347 | 3D QTT topology optimization for antennas |
| `qtt_3d.py` | 1,839 | 3D QTT Maxwell operators |
| `chu_limit_gpu.py` | 1,585 | GPU-native Chu limit |
| `qtt_helmholtz_gpu.py` | 1,414 | GPU-native QTT Helmholtz |
| `qtt_helmholtz.py` | 1,305 | QTT Helmholtz via TT-GMRES |
| `frequency_sweep.py` | 1,254 | QTT frequency sweep |
| `topology_opt.py` | 1,317 | Density-based topology optimization |
| `s_parameters.py` | 1,069 | S-parameter extraction |
| `boundaries.py` | 1,031 | PML for QTT Maxwell |
| `electrostatics.py` | 805 | Poisson-Boltzmann (III.1) |
| Others (14 files) | ~1,300 | Magnetostatics, wave propagation, FDFD, photonics, antenna/microwave + 7 STARK adapters |

### 2.8 `tensornet/platform/` — Platform Layer

**40 files, ~15,474 LOC**

| Sub-area | Key Files | Description |
|----------|-----------|-------------|
| **Data & export** | `data_model.py`, `export.py`, `arrow_export.py`, `mesh_import.py`, `checkpoint.py`, `lakehouse.py`, `timeseries_db.py`, `data_versioning.py` | Mesh, Field, BCs, VTK/HDF5/CSV/Parquet, GMSH import |
| **Solvers** | `solvers.py`, `coupled.py`, `qtt_solver.py`, `qtt.py`, `tci.py`, `acceleration.py` | Time integrators, multi-physics coupling, QTT bridge |
| **V&V** | `vv/` (7 files) | Benchmarks, conservation, convergence, MMS, performance, stability |
| **Platform services** | `protocols.py`, `domain_pack.py`, `deprecation.py`, `security.py`, `lineage.py`, `reproduce.py`, `replay.py`, `experiment_tracker.py`, `federation.py` | PEP 544 protocols, DomainPack plugin, provenance, SBOM |
| **Analysis** | `inverse.py`, `adjoint.py`, `optimization.py`, `uq.py`, `postprocess.py`, `visualize.py` | Inverse problems, Bayesian UQ, gradient optimization |
| **Vertical slices** | `vertical_ode.py`, `vertical_pde.py`, `vertical_vv.py` | End-to-end demo slices |

### 2.9 `tensornet/materials/` — Materials Science & Mechanics

**42 files, ~10,100 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| Root solvers | 7 | First principles (DFT), mechanical properties, microstructure (Cahn-Hilliard), polymers/soft matter, radiation damage, ceramics (UHTC) |
| `mechanics/` | 14 solvers + 6 adapters | Continuum (Neo-Hookean), DEM, fracture (LEFM/EPFM), HHO, IGA (NURBS), mimetic FD, MPM, Noether verification, peridynamics, structural (beam/plate/shell), symplectic (Ruth-4/Yoshida), tribology, variational, VEM, XFEM |
| `manufacturing/` | 2 + 1 adapter | Goldak welding, Scheil solidification, AM melt pool |
| Trace adapters | 7 | STARK adapters for XIV.1–XIV.7 |

### 2.10 `tensornet/applied/` — Applied Physics Domains

**48 files, ~11,800 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| `acoustics/` | 4 | Helmholtz BEM, room acoustics, Lighthill aeroacoustic, Tam-Auriault jet noise, duct acoustics |
| `optics/` | 8 | Physical optics (diffraction, polarization), quantum optics (Jaynes-Cummings), laser physics, ultrafast optics + 4 STARK adapters |
| `intent/` | 8 | NL field steering — LLM-to-solver pipeline, FieldQuery DSL, swarm command |
| `financial/` | 4 | NS solver on order-book density, Coinbase live feed, Unreal Engine bridge |
| `physics/` | 4 | Hypersonic hazard field, trajectory optimizer |
| `medical/` | 2 | Carreau-Yasuda blood flow, stenosis simulation |
| `cyber/` | 2 | DDoS-as-fluid-dynamics, heat equation on graphs |
| `emergency/` | 2 | Rothermel wildfire model, ember transport |
| `particle/` | 3 | Beyond Standard Model — neutrino oscillations, dark matter, GUT |
| `radiation/` | 1 | Flux-limited diffusion, grey/multigroup transport |
| `robotics_physics/` | 2 | Newton-Euler, Featherstone ABA, LCP contact |
| `special_applied/` | 2 | Astrodynamics STARK adapter |

### 2.11 `tensornet/plasma_nuclear/` — Plasma & Nuclear Physics

**36 files, ~8,800 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| `plasma/` | 7 solvers + 7 adapters | Extended MHD, gyrokinetics, space plasma, laser-plasma, magnetic reconnection, dusty plasmas |
| `nuclear/` | 4 solvers + 3 adapters | Nuclear structure, reactions, astrophysics |
| `fusion/` | 9 | DARPA MARRS solid-state fusion — electron screening, phonon trigger, resonant catalysis, superionic dynamics, QTT screening, tokamak equilibrium |

### 2.12 `tensornet/fluids/` — Advanced Fluid Physics

**38 files, ~8,900 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| `coupled/` | 3 solvers + 7 adapters | Coupled MHD, electro-mechanical, thermo-mechanical |
| `fsi/` | 1 | FSI, ALE, flutter, VIV (485 LOC) |
| `multiphase/` | 2 | Cahn-Hilliard + VOF |
| `free_surface/` | 1 | Level-set, thin-film |
| `heat_transfer/` | 2 | Radiative + coupled heat transfer (957 LOC) |
| `mesh_amr/` | 1 | Octree/quadtree AMR (767 LOC) |
| `porous_media/` | 1 | Darcy, Brinkman, Richards |
| `phase_field/` | 2 | Phase-field methods, PFC model |
| `multiscale/` | 2 | FE², homogenisation |
| `computational_methods/` trace adapters | 6 | HPC, inverse problems, large-scale linalg, mesh generation, optimization |
| Top-level trace adapters | 8 | STARK adapters for II.3–II.10 |

### 2.13 `tensornet/life_sci/` — Life Sciences

**33 files, ~7,500 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| `chemistry/` | 7 solvers + 7 adapters | PES/NEB, reaction rates (TST/RRKM), quantum reactive, nonadiabatic (FSSH), photochemistry, spectroscopy (IR/Raman/NMR) |
| `biology/` | 3 | Systems biology (FBA, GRN, Gillespie SSA), social-force ABM |
| `biomedical/` | 2 + 1 adapter | Cardiac electrophysiology (bidomain), compartment PK, hyperelasticity |
| `biophysics/` trace adapters | 7 | Drug design, membrane, neuroscience, nucleic acids, protein structure, systems biology |
| `md/` | 2 | Full MD engine — Verlet, Nosé-Hoover, Parrinello-Rahman, PME, REMD (935 LOC) |
| `membrane_bio/` | 1 | Coarse-grained lipid bilayer, electroporation |

### 2.14 `tensornet/energy_env/` — Energy & Environment

**16 files, ~3,500 LOC**

| Sub-module | Files | Description |
|------------|-------|-------------|
| `energy/` | 5 | Jensen Park wake model, drift-diffusion solar, Newman battery, Butler-Volmer + Unreal Engine wake viz |
| `environmental/` | 3 | Gaussian plume, SCS curve number, storm surge |
| `urban/` | 3 | Procedural voxel city generation, Venturi urban wind solver |
| `agri/` | 2 | Vertical farm microclimate optimization |

### 2.15 `tensornet/infra/` — Infrastructure Modules

**~70 files** spanning deployment, integration, visualization, and orchestration.

| Sub-module | Description |
|------------|-------------|
| `digital_twin/` (6 files, ~3,866 LOC) | Digital twin framework — health monitor, predictive maintenance, reduced-order models (POD/DMD), real-time state sync, twin orchestrator |
| `coordination/` | Multi-component coordination |
| `deployment/` | Deployment tooling |
| `fieldops/` | Field operation utilities |
| `fieldos/` | Field OS services |
| `hyperenv/` | HyperTensor environment management |
| `hypersim/` | Simulation orchestration |
| `hypervisual/` | Visualization services |
| `integration/` | External system integration |
| `oracle/` | Oracle infrastructure |
| `provenance/` | Content-addressed provenance |
| `sdk/` | SDK infrastructure |
| `site/` | Site/web infrastructure |
| `sovereign/` | Sovereign compute infrastructure |
| `zk/` | Zero-knowledge proof infrastructure |

### 2.16 Additional TensorNet Modules

| Module | Description |
|--------|-------------|
| `tensornet/core/` | Core TN operations — MPS, MPO, DMRG, TEBD, contraction |
| `tensornet/algorithms/` | Advanced TN algorithms |
| `tensornet/mpo/` | Matrix Product Operator library |
| `tensornet/mps/` | Matrix Product State library |
| `tensornet/qtt/` | QTT utilities and construction |
| `tensornet/types/` | Type definitions |
| `tensornet/numerics/` | Numerical utilities |
| `tensornet/validation/` | Validation framework |
| `tensornet/certification/` | TPC certification engine |
| `tensornet/sim/` | Simulation orchestration |
| `tensornet/visualization/` | Plotting and rendering |
| `tensornet/benchmark_runner.py` | Benchmark execution runner |

---

## 3. Applications

`apps/` — **207 code files, ~84K LOC** — Deployed or deployable applications.

### 3.1 `apps/oracle/` — Market Prediction Engine *(APPLICATION)*

**25 files, ~12,787 LOC (Python)**

Real-time market prediction engine using QTT tensor networks. Trinity signal
system processing live exchange data from Coinbase, Binance, and Bybit.

| Component | Key Files | Description |
|-----------|-----------|-------------|
| Core engine | `ns_predictor.py` (1,600), `oracle_engine.py`, `live_oracle.py` | NS-based market prediction |
| Exchange feeds | `coinbase_oracle.py`, `binance_oracle.py`, `bybit_oracle.py`, `binance_firehose.py`, `coinbase_firehose.py`, `bybit_firehose.py`, `multi_exchange_oracle.py` | WebSocket market data ingestion |
| Galaxy Trinity | `galaxy_feed_v2.py`, `galaxy_feed_v3.py` | Three firehoses → one signal |
| QTT encoding | `qtt_encoder.py`, `qtt_encoder_cuda.py`, `sketch_encoder.py` | Order book → MPS/holographic compression |
| GPU kernels | `triton_slicer.py`, `cuda_graph_slicer.py`, `batched_kernel.py`, `zero_loop_kernel.py`, `oracle_kernel.py` | Triton/CUDA financial slicers |

### 3.2 `apps/glass_cockpit/` — Atmospheric Observation Layer *(APPLICATION)*

**68 code files (62 Rust + 11 WGSL shaders), ~23,708 LOC**

Sovereign GPU-native atmospheric visualization layer built on `wgpu`. Real-time
globe rendering with QTT tensor field overlays.

| Component | Key Files | Description |
|-----------|-----------|-------------|
| Core renderer | `renderer.rs`, `globe.rs`, `globe_quadtree.rs` | wgpu pipeline, Earth globe with LOD quadtree |
| Tensor viz | `tensor_renderer.rs`, `tensor_field.rs`, `vorticity_renderer.rs`, `vorticity_ghost.rs` | QTT field visualization, vorticity rendering |
| Data | `weather_tensor.rs`, `grib_decoder.rs`, `noaa_fetcher.rs` | GRIB weather data, NOAA integration |
| Flow viz | `streamlines.rs`, `particle_system.rs` | Streamline + particle flow visualization |
| Bridge | `bridge.rs`, `ram_bridge_v2.rs`, `injection_buffer.rs` | RAM bridge to Python engine |
| UI | `hud_overlay.rs`, `camera.rs`, `interaction.rs`, `text.rs`, `text_gpu.rs` | HUD, 3D camera, SDF GPU text |
| Infra | `affinity.rs`, `lod.rs`, `telemetry.rs`, `verification.rs`, `starfield.rs` | E-core pinning, LOD, proof verification |
| Shaders | 11 WGSL files | Globe, grid, particles, SDF text, tensor colormap |

### 3.3 `apps/qtenet/` — QTeneT Physics Engine *(APPLICATION)*

**43 code files, ~12,866 LOC (Python)**

Quantized Tensor Network Physics Engine — O(log N) complexity PDE solver
distributed as a standalone package.

| Component | Key Files | Description |
|-----------|-----------|-------------|
| Solvers | `vlasov_genuine.py` (1,264), `ns3d.py` (1,048), `vlasov6d_genuine.py` (1,005), `vlasov.py`, `euler.py` | 6D Vlasov (1B grid → 200 KB), 3D NS |
| TCI | `tci/from_function.py`, `tci/from_samples.py` | Black-box function → QTT |
| Operators | `shift_nd.py`, `laplacian_nd.py`, `gradient_nd.py` | N-dimensional MPO operators |
| Workflows | `workflows/qtt_turbulence/` (7 files) | DHIT benchmark, turbulence proof, spectral NS3D |
| SDK | `sdk/api.py`, `apps/cli.py` | Public API, CLI entry |
| Benchmarks | `curse_scaling.py` | Curse-of-dimensionality scaling |
| Demos | `holy_grail.py`, `two_stream.py` | 6D plasma, two-stream instability |

### 3.4 `apps/oracle_node/` — Tensor Genesis Oracle Node *(APPLICATION)*

**4 files, ~2,175 LOC** — Domain-agnostic mathematical structure engine.
Raw numbers → signed attestation via OT, SGW, RKHS, PH, GA pipeline.
FastAPI server.

### 3.5 `apps/the_compressor/` — Universal QTT Compressor *(APPLICATION)*

**20 code files** — QTT compression for arbitrary data (NOAA 24h, hybrid
Triton kernels, block SVD, rank sweep, decompression, GPU reconstruction).

### 3.6 `apps/ledger/` — Physics Taxonomy Ledger *(APPLICATION)*

**3 Python files + 168 YAML node definitions** — The canonical machine-readable
registry for all 168 physics taxonomy nodes. Schema-validated via
`schema.yaml`. Dashboard generation.

### 3.7 `apps/sdk_legacy/` — Legacy SDK Distribution *(APPLICATION)*

**25 files** — Python, TypeScript, Conda, Docker, and enterprise SDK
distributions for external consumers.

### 3.8 `apps/global_eye/` — Global Eye *(APPLICATION)*

**Rust application** (`Cargo.toml` + `src/`) — Satellite-scale global
observation.

### 3.9 `apps/sovereign_api/` — Sovereign API *(APPLICATION)*

**2 files** — `sovereign_api.py` + `live_data_provider.py` — Sovereign
compute API surface.

### 3.10 `apps/sovereign_ui/` — Sovereign UI *(APPLICATION)*

**2 files** — Sovereign compute UI frontend.

### 3.11 `apps/exploit_verification/` — Exploit Verification *(APPLICATION)*

**1 file** — Verification harness for exploit engine results.

### 3.12 `apps/golden_demo/` — Golden Demo *(APPLICATION)*

**1 file** — Curated demonstration runner.

### 3.13 `apps/trustless_verify/` — Trustless Verification *(APPLICATION)*

**2 files** — TPC certificate verification tool.

### 3.14 `apps/vlasov_proof/` — Vlasov Proof Generator *(APPLICATION)*

**1 file** — Standalone Vlasov-Poisson proof certificate generator.

### 3.15 `apps/glass_cockpit_root/` — Glass Cockpit Root *(APPLICATION)*

**4 files** — Root-level Glass Cockpit configuration.

---

## 4. Products

`products/` — **154 files, ~78K LOC** — Shippable, deployable product suites.

### 4.1 `products/facial_plastics/` — Facial Plastic Surgery Digital Twin *(PRODUCT)*

**102 Python files + SPA UI + SvelteKit UI, ~50,446 LOC**

Full-featured physics-grounded surgical simulation and planning platform for
facial plastic and reconstructive surgery. Multi-tenant, tested, deployable
with Gunicorn/Caddy.

| Sub-package | Files | LOC | Description |
|-------------|-------|-----|-------------|
| **core/** | 4 | ~1,770 | Types, CaseBundle (content-addressed), config, provenance (SHA-256) |
| **data/** | 8 | ~4,855 | DICOM ingest, surface ingest, photo ingest, anatomy generator (1,917 LOC), paired dataset, case library, curator, synthetic augment |
| **twin/** | 6 | ~2,858 | Twin builder orchestrator, segmentation, landmarks, FEM meshing, registration, materials |
| **plan/** | 5 | ~4,626 | Plan compiler (2,210 LOC), DSL, rhinoplasty operators, facelift operators, blepharoplasty, fillers |
| **sim/** | 9 | ~5,459 | FEM soft tissue (958 LOC), CFD airway, FSI nasal valve, cartilage, anisotropy, aging, orchestrator, sutures, healing |
| **metrics/** | 7 | ~4,571 | Aesthetic proportions, safety, NSGA-II optimizer, cohort analytics, uncertainty (MC/LHS), functional, distributed optimizer |
| **governance/** | 4 | ~1,510 | Multi-tenant isolation, RBAC (surgeon/resident/admin), consent workflow, immutable audit trail |
| **postop/** | 5 | ~2,068 | Bland-Altman validation, ICP alignment, dashboard, Bayesian calibration, outcome ingest |
| **ui/** | 4 + SPA + SvelteKit | ~5,559 | Full backend API (4,041 LOC), WSGI, HTTP server, auth middleware; Three.js 3D viewer; 7-phase SvelteKit UI |
| **tests/** | 28 | ~13,000 | Full test suite: sim, twin, plan, metrics, governance, UI, integration |
| Root | 6 | ~2,325 | CLI, anatomy generator, reports, Gunicorn config, `__init__.py`, `__main__.py` |

### 4.2 `products/fluidelite/` — FluidElite Solver *(PRODUCT)*

GPU kernels directory — pre-compiled solver kernels for the FluidElite
commercial CFD product.

### 4.3 `products/fluidelite-zk/` — FluidElite ZK Verifier *(PRODUCT)*

ZK-verifiable FluidElite deployment — Gevulot keys, Foundry contracts,
container image.

### 4.4 `products/the_compressor/` — The Compressor Data Store *(PRODUCT)*

Pre-computed QTT compression artifacts (NOAA 24h, hybrid, block SVD, auto
venv outputs).

---

## 5. Experiments

`experiments/` — **2,623 files, ~442K LOC** — Research, frontier physics,
demos, and proof-of-concept code.

### 5.1 `experiments/hvac_cfd/` — HVAC-CFD Digital Twin Platform *(APPLICATION)*

**109 Python files + Next.js UI, ~52,345 LOC**

Full HVAC CFD digital twin platform — a complete consulting-grade business
application. Three major subsystems:

| Subsystem | Description |
|-----------|-------------|
| **HyperFOAM** (`Review/hyperfoam/`, 30+ files) | GPU-native CFD solver for HVAC: immersed boundary grid (1,137 LOC), thermal/species transport, k-ε turbulence, multizone (zone graph, data center, fire/smoke, equipment, duplex), optimizer, pipeline, predictive alerts, CAD/IFC import, cleanroom ISO 14644 |
| **Intake** (`intake/`, 15+ files) | Universal document intake — Streamlit app (1,841 LOC), staging UI, PDF/Excel/IFC/image extractors, schema validation, unit conversion ("Sandwich Method"), 3D Plotly visualization, HyperFOAM bridge |
| **Web-UI & Dashboard** | EigenPsi CFD analysis server (976 LOC), Next.js HyperTensor UI with FastAPI backend (944 LOC), Streamlit HVAC dashboard |
| **Benchmarks** (`Review/Tier1/`, 15 files) | Nielsen room benchmarks, QTT-NS solvers, thermal multi-physics |
| **Tests** (`Review/tests/`, 10 files) | Deployment, crucible, validation, eigenpsi API |

### 5.2 `experiments/frontier/` — Frontier Physics *(EXPERIMENT)*

**56 Python files, ~29,528 LOC** — Seven frontier physics domains, each with
validation attestations.

| Domain | Files | LOC | Key Content |
|--------|-------|-----|-------------|
| `01_FUSION/` | 5 | ~2,220 | Landau damping, tokamak geometry, two-stream instability |
| `02_SPACE_WEATHER/` | 6 | ~1,816 | Solar wind bow shock, Sod shock, Alfvén waves |
| `03_SEMICONDUCTOR_PLASMA/` | 5 | ~2,093 | ICP discharge, plasma sheath, etch rate, ion energy distribution |
| `04_PARTICLE_ACCELERATOR/` | 4 | ~1,862 | Beam dynamics, plasma wakefield, RF cavity, space charge |
| `05_QUANTUM_ERROR_CORRECTION/` | 3 | ~1,371 | Surface code, stabilizer formalism, threshold analysis |
| `06_FUSION_CONTROL/` | 3 | ~2,182 | Plasma controller, disruption predictor, control loop |
| `07_GENOMICS/` | 27 | ~17,984 | Full-genome tensor DNA, GPU genomics, clinical classifier, ESM-2 scorer, CRISPR guide RNA, multi-species conservation, RNA structure, ClinVar integration, variant API (FastAPI) |

### 5.3 `experiments/demos/` — Demonstrations *(DEMO)*

**46 runner files + supporting assets, ~22,501 LOC**

| Category | Key Runners | Description |
|----------|-------------|-------------|
| Desktop apps | `hypertensor_hub.py` (1,193), `hypertensor_pro.py` (1,275) | PySide6 + VisPy desktop application |
| Forensics | `forensic_instrument.py` (1,357), `forensic_hub.py` (1,294), `forensic_hub_v2.py` | Weather singularity investigation tools |
| Flagship | `flagship_pipeline.py` (960) | Phase 21-24 full integration proof |
| Millennium | `millennium_hunter.py` (621) | NS singularity numerical analysis ($1M problem) |
| Weather | `weather_viewer.py`, `ingest_noaa_gfs.py`, `blue_marble.py` | 3D atmospheric QTT viewer |
| CFD | `conference_room_cfd.py`, `conference_room_qtt.py`, `cfd_shock.py`, `qtt_shock_tube.py` | Ventilation, shock waves |
| Physics | `provable_physics.py`, `pure_qtt_pde.py`, `resolution_independence.py` | Cryptographic proofs, PDE demos |
| Black Swan | `trap_the_swan.py`, `black_swan_reproduce.py`, `black_swan_945_forensic.py`, `black_swan_1024_confirm.py` | Singularity hunting |
| KH instability | `kelvin_helmholtz_animation.py`, `kelvin_helmholtz_demo.py` | KH vortex roll-up |
| Finance | `oracle_hunt_demo.py` | ORACLE vulnerability hunt |
| Streamlit | `streamlit_app.py` | FluidEliteZK — ZK-verifiable LLM demo |

### 5.4 `experiments/ai_scientist/` — AI Scientist *(EXPERIMENT)*

**6 files, ~2,080 LOC** — AI-driven scientific discovery pipeline:
conjecture → formalize (PhysLean / Lean 4) → prove. LLM-aided formalization.

### 5.5 `experiments/tci_llm/` — TCI Language Model *(EXPERIMENT)*

**10 files, ~2,261 LOC** — Gradient-free language modeling via tensor cross
interpolation. SVD-LLM with 119× improvement. Rank ~50 insight.

### 5.6 `experiments/aave_extraction/` — DeFi Security Research *(TOOL)*

**17 files, ~9,064 LOC** — Ethereum smart contract vulnerability scanning.
QTT-EVM extraction engine, cross-contract exploitation, proxy init analysis,
live 72h WebSocket scanner, kill-chains for Ronin (1,907 ETH), Harmony
(94.79 ETH), Lido (16.47 ETH), Makina Caliber.

### 5.7 `experiments/pwa_engine/` — Partial Wave Analysis *(EXPERIMENT)*

**4 files, ~2,595 LOC** — Thesis-grade PWA engine from Badui (2020) Eq. 5.48.
Gram-matrix-accelerated extended likelihood. Core engine: 2,299 LOC.

### 5.8 `experiments/lux/` — lUX Proof Package Viewer *(APPLICATION)*

**Full TypeScript/Next.js monorepo** — The frontend UI/UX for Facial Plastic
Surgery (FPS) digital twin proof verification.

| Component | Description |
|-----------|-------------|
| `packages/core/` | Core library — proof validation, attestation, test fixtures |
| `packages/ui/` | Next.js frontend — dashboard, gallery, proof workspace, design system (Card, Chip, CodeBlock, DataTable, VerdictSeal, ModeDial, etc.) |
| Routes | `/`, `/gallery`, `/packages/[id]` |
| API | Auth, health, metrics, packages, domains, CSP |
| Screens | Summary, Evidence, Gates, Integrity, Timeline, Compare |
| Testing | Playwright e2e + Vitest + Storybook |
| Infra | Docker Compose, CI, Lighthouse, pnpm workspace |

### 5.9 `experiments/cfd_hvac/` — CFD-HVAC Attestations *(DOCUMENTATION)*

**No Python code** — `README.md` plus Tier 1 QTT CFD and Tier 2 Thermal
Comfort attestation JSON files.

### 5.10 Other Experiments

| Directory | Classification | Description |
|-----------|---------------|-------------|
| `experiments/ahmed_body/` | Experiment | Ahmed body automotive aerodynamics |
| `experiments/benchmarks/` | Benchmark | Performance benchmark suite |
| `experiments/validation/` | Validation | Challenge II validation phases |
| `experiments/physics_standalone/` | Experiment | Standalone physics demos |
| `experiments/notebooks/` | Notebook | Jupyter exploration notebooks |
| `experiments/papers/` | Research | Paper drafts and figures |
| `experiments/scripts/` | Script | Ad-hoc experiment scripts |
| `experiments/visualization/` | Visualization | Rendering experiments |
| `experiments/orbital_frames/` | Experiment | Orbital frame generation |
| `experiments/santa2025/` | Experiment | Santa 2025 challenge |
| `experiments/qtt_3d_ops/` | Experiment | 3D QTT operation experiments |
| `experiments/qtt_aero_output/` | Data | QTT aerodynamics output artifacts |
| `experiments/flash_liquidator/` | Tool | Flash liquidation experiments |
| `experiments/etherfi_oracle_frontrun_poc/` | Tool | EtherFi oracle frontrunning PoC |

---

## 6. Validation Gauntlets

`tools/scripts/gauntlets/` — **38 files, ~37,060 LOC** — Production-grade
validation suites covering every Civilization Stack project and all Trustless
Physics Certificate phases.

### Civilization Stack Gauntlets (20)

| Gauntlet | LOC | Project |
|----------|-----|---------|
| `tomahawk_cfd_gauntlet.py` | 823 | #1 TOMAHAWK — tokamak plasma rampdown, 27K× TT compression |
| `tig011a_multimechanism.py` | 2,377 | #2 TIG-011a — multi-mechanism binding physics |
| `tig011a_dynamic_validation.py` | 1,093 | #2 TIG-011a — 500 ns MD, H-bond, FEP |
| `tig011a_docking_qmmm.py` | 935 | #2 TIG-011a — multi-pose QM/MM |
| `tig011a_dielectric_gauntlet.py` | 605 | #2 TIG-011a — dielectric sweep |
| `tig011a_wiggle_tt.py` | 403 | #2 TIG-011a — energy well test |
| `tig011a_tox_screen.py` | 398 | #2 TIG-011a — in-silico tox |
| `tig011a_attestation.py` | 340 | #2 TIG-011a — crypto attestation |
| `snhff_stochastic_gauntlet.py` | 766 | #3 SnHf-F — quantum well EUV resist |
| `li3incl48br12_superionic_gauntlet.py` | 932 | #4 Li₃InCl₄.₈Br₁.₂ — paddle-wheel resonance |
| `laluh6_odin_gauntlet.py` | 1,083 | #5 LaLuH₆ ODIN — Meissner, zero resistance, Jc |
| `hellskin_gauntlet.py` | 1,075 | #6 HELL-SKIN — 60 MW arc-jet, thermal shock |
| `starheart_gauntlet.py` | 1,177 | #7 STAR-HEART — Q > 10, MHD stability, Lawson |
| `femto_fabricator_gauntlet.py` | 1,202 | #11 FEMTO-FABRICATOR — diamondoid AFM construction |
| `proteome_compiler_gauntlet.py` | 1,111 | #12 PROTEOME COMPILER — function → protein → DNA |
| `metric_engine_gauntlet.py` | 945 | #13 METRIC ENGINE — Schwinger-limit metric engineering |
| `prometheus_gauntlet.py` | 1,707 | #14 PROMETHEUS — IIT Φ consciousness computation |
| `oracle_gauntlet.py` | 791 | #15 ORACLE — warm-temp topological qubits |
| `orbital_forge_gauntlet.py` | 1,192 | #16 ORBITAL FORGE — orbital mechanics, rotating habitats |
| `hermes_gauntlet.py` | 1,276 | #17 HERMES — interstellar signal propagation, QKD |
| `cornucopia_gauntlet.py` | 1,180 | #18 CORNUCOPIA — energy/material/info abundance |
| `chronos_gauntlet.py` | 1,246 | #19 CHRONOS — relativistic time dilation, causality |
| `sovereign_genesis_gauntlet.py` | 1,022 | #20 SOVEREIGN GENESIS — Von Neumann self-replicator |

### Infrastructure & Trustless Physics Gauntlets (15)

| Gauntlet | LOC | Scope |
|----------|-----|-------|
| `trustless_physics_gauntlet.py` | 918 | Phase 0 — TPC format, computation trace, cert gen |
| `trustless_physics_phase1_gauntlet.py` | 793 | Phase 1 — Single-domain MVP (Euler 3D) |
| `trustless_physics_phase2_gauntlet.py` | 973 | Phase 2 — Multi-domain, Lean proofs, customer API |
| `trustless_physics_phase3_gauntlet.py` | 840 | Phase 3 — Prover pool, Gevulot, dashboard |
| `trustless_physics_phase5_gauntlet.py` | 912 | Phase 5 — Tier 1 (4 domains) |
| `trustless_physics_phase6_gauntlet.py` | 864 | Phase 6 — Tier 2A (25 domains) |
| `trustless_physics_phase7_gauntlet.py` | 1,304 | Phase 7 — Tier 2B (40 domains) |
| `trustless_physics_phase8_gauntlet.py` | 1,150 | Phase 8 — Tier 3 (45 domains) |
| `trustless_physics_phase9_gauntlet.py` | 951 | Phase 9 — Tier 4 (26 domains) |
| `trustless_physics_phase10_gauntlet.py` | 492 | Phase 10 — Full-spectrum (140 domains, 5 tiers) |
| `ade_gauntlet.py` | 1,075 | Autonomous Discovery Engine v1 |
| `ade_gauntlet_v2.py` | 1,015 | ADE v2 — regime-aware validation |
| `genesis_benchmark_suite.py` | 872 | Tensor Genesis — all 7 QTT-native layers |
| `qtt_native_gauntlet.py` | 582 | QTT-native — tropical matrix ops, persistence |
| `production_hardening_gauntlet.py` | 640 | Production hardening — logging, profiling, validation |

---

## 7. Tools & Scripts

`tools/scripts/` — **208 files, ~65,684+ LOC total** across 8 sub-directories
plus 68 top-level scripts.

### Top-Level Scripts (68 files, ~28,624 LOC)

| Category | Key Scripts | Description |
|----------|-------------|-------------|
| **Ahmed Body** | `ahmed_body_ib_solver.py` (1,691), `ahmed_body_ib_solver_v1.py`, `ahmed_body_spectrum.py` | Immersed boundary QTT solvers |
| **Civic Aero** | `civic_aero.py` (1,799), `civic_aero_qtt.py` (1,660) | Honda Civic external aero, QTT-native Morton grid |
| **Supersonic** | `mach5_wedge.py` | Mach 5 oblique shock validation |
| **Industrial** | `industrial_qtt_gpu_simulation.py` (1,020) | 1024³ DNS + compression scaling |
| **Trustless** | `trustless_physics.py` (1,765), `run_trustless_ahmed.py` | Hash-chained Merkle-tree crypto attestation |
| **Vlasov** | `vlasov_6d_video.py` (1,153), `generate_vlasov_certificate.py` (1,085) | 6D Vlasov-Maxwell video, STARK proof certs |
| **Reports** | `generate_elite_report.py` (1,575), `generate_pdf_report.py` (1,353), `generate_executive_certificate.py` (821) | PDF/HTML report generation |
| **V&V gates** | `physics_validation.py`, `detect_vv_regression.py`, `determinism_check.py` | Sod shock, oblique shock, SBLI gates |
| **Security** | `security_scan.py`, `sign_manifest.py` (PQC Dilithium2) | Vulnerability scanning, PQC signing |
| **Packaging** | `packaging_gate.py`, `release_check.py`, `generate_sbom.py`, `export_release_zip.py` | Build gates, SBOM (CycloneDX/SPDX), release |
| **DevOps** | `format_lint.py`, `check_import_cycles.py`, `check_docstrings.py`, `check_contract_drift.py`, `check_pwa_regression.py`, `rename_tests.py`, `update_loc_counts.py` | Code quality gates |
| **Benchmarks** | `benchmark_tt_cfd.py`, `gauntlet_vs_nvidia.py`, `scaling_tests.py`, `compare_tenpy.py`, `reproduce.py` | Performance benchmarks |
| **QTT debug** | `qtt_axis_exact.py`, `qtt_axis_verify.py`, `qtt_debug_step.py`, `qtt_deriv_verify.py`, `qtt_single_step.py` | Diagnostic / verification scripts |

### Sub-directories

| Directory | Files | Description |
|-----------|-------|-------------|
| `gauntlets/` | 38 | *(See §6 above)* |
| `research/` | 60 | Yang-Mills proof pipeline (v1/v2), HELL-SKIN thermal solver, STAR-HEART fusion solver, ODIN superconductor solver, live market fluid analysis, NS millennium pipeline, QTT connectome, neuromorphic integration, EUV quantum well, FLU-X001 M2 blocker, sovereign API/daemon, Kida convergence |
| `testing/` | 19 | 100K stress, 4K, CUDA kernel, fusion, GPU, grid shock, implicit, MPO perf, optimization, phase 3/4 integration, planetary, stealth, urban canyon, weather tests |
| `tpc/` | 8 | TPC certificate generators (Euler3D, NS-IMEX, heat, Vlasov, phases 6–9) |
| `profiling/` | 4 | Deep profile, component, ops, 4K render |
| `setup/` | 6 | GPU check, PyTorch check, quick test, realtime renderer, pressure solver |
| `tools/` | 5 | Extension builder, crypto QTT compress/decompress, FluidElite ingest |

### Root-Level Tools

| File | Description |
|------|-------------|
| `tools/dep_graph.py` | Dependency graph generator |
| `tools/forensic_loc_sweep.py` | Forensic LOC sweep v1 |
| `tools/forensic_loc_sweep_v2.py` | Forensic LOC sweep v2 |
| `tools/loc_audit.py` | LOC audit tool |
| `tools/migrate_tensornet_phase5.py` | Phase 5 migration script |
| `tools/sync_versions.py` | Version synchronization |

---

## 8. Proofs & Certificates

`proofs/` — **7,473 files, ~371K LOC** — Trustless Physics Certificates,
ZK proof circuits, conservation proofs, and Yang-Mills research.

| Sub-area | Description |
|----------|-------------|
| **Phase proofs** | `proof_phase_1a.py` through `proof_phase_6.py` — staged TPC generation with result JSON |
| **Phase 21-24** | Dense audit, TDVP conservation, WENO order/shock, TT evolution |
| **Discovery** | `proof_discovery_engine.py` — ADE pipeline validation |
| **DeFi** | `proof_defi_pipeline.py`, `proof_exploit_invariants.py` — DeFi security proofs |
| **Plasma** | `proof_plasma_pipeline.py` — plasma discovery pipeline proof |
| **Markets** | `proof_markets_pipeline.py` — market pipeline proof |
| **Molecular** | `proof_molecular_pipeline.py` — molecular discovery proof |
| **Algorithms** | `proof_algorithms.py`, `proof_decompositions.py` — TN algorithm validation |
| **CFD** | `proof_cfd_conservation.py`, `proof_ns_projection.py` — conservation law proofs |
| **Millennium** | `proof_millennium.py` — NS millennium problem numerical evidence |
| **Production** | `proof_production.py`, `proof_level_3.py` through `proof_level_5.py` — staged maturity gates |
| **Navier-Stokes** | `proofs/navier_stokes/` — dedicated NS proof sub-directory |
| **Conservation** | `proofs/conservation/` — conservation law proof artifacts |
| **Yang-Mills** | `proofs/yang_mills/` — Yang-Mills mass gap research |
| **TPC engine** | `proofs/proof_engine/` — TPC proof generation engine |
| **ZK circuits** | `proofs/zk_circuits/` — zero-knowledge proof circuits |
| **ZK targets** | `proofs/zk_targets/` — verification target contracts |
| **Tenet** | `proofs/tenet_tphy/` — TeneT physics proofs |
| **TPC** | `proofs/tpc/` — certificate storage |

---

## 9. Rust Crates

`crates/` — **15 crates, 716 files, ~204K LOC**

| Crate | Description |
|-------|-------------|
| `tci_core/` | Core TCI (Tensor Cross Interpolation) in Rust |
| `tci_core_rust/` | Pure-Rust TCI implementation |
| `fluidelite/` | FluidElite core solver |
| `fluidelite_core/` | FluidElite core library |
| `fluidelite_circuits/` | ZK circuit definitions |
| `fluidelite_infra/` | FluidElite infrastructure |
| `fluidelite_zk/` | ZK verification for FluidElite |
| `gevulot/` | Gevulot decentralized prover integration |
| `hyper_bridge/` | Python ↔ Rust bridge |
| `hyper_core/` | Core Rust library |
| `hyper_gpu_py/` | GPU-accelerated Python bindings |
| `proof_bridge/` | Proof system bridge |
| `qtt_cem/` | QTT Cross-Entropy Method optimizer |
| `qtt_fea/` | QTT Finite Element Analysis |
| `qtt_opt/` | QTT Optimization library |

---

## 10. Infrastructure & Deployment

### `contracts/` — Solidity Smart Contracts (3 files, ~675 LOC)

| Contract | Description |
|----------|-------------|
| `FluidEliteHalo2Verifier.sol` | Halo2 ZK proof on-chain verifier |
| `HyperTensorBindingVerifier.sol` | Binding proof verifier |
| `ZeroExpansionSemaphoreVerifier.sol` | Semaphore-based anonymity verifier |
| `v1/` | Version 1 contract archive |

### `integrations/` — Game Engine Integrations (1 file)

| Directory | Description |
|-----------|-------------|
| `unity/` | Unity Engine integration |
| `unreal/` | Unreal Engine integration |

### `deploy/` — Deployment Infrastructure

| Directory | Description |
|-----------|-------------|
| `Containerfile` | Container build definition |
| `config/` | Deployment configuration |
| `docker/` | Docker Compose and Dockerfile definitions |
| `telemetry/` | Telemetry collection infrastructure |

---

## 11. HyperTensor Platform Service

`hypertensor/` — **31 files, ~3,900 LOC** — The SaaS platform layer.

| Sub-module | Files | Description |
|------------|-------|-------------|
| `api/` | 6 | FastAPI application, auth, config, routers |
| `billing/` | 3 | Usage metering, invoice generation |
| `cli/` | 3 | Command-line interface |
| `core/` | 7 | Certificates, evidence, executor, hasher, registry, sanitizer |
| `jobs/` | 3 | Job models and persistent store |
| `mcp/` | 2 | Model Context Protocol server |
| `sdk/` | 2 | Python SDK client |

---

## 12. Test Suite

`tests/` — **104 Python files, ~52,258 LOC**

| Category | Key Files | Description |
|----------|-----------|-------------|
| **Audit layers** | `audit_layer_4.py` through `audit_layer_9.py` | Constitutional audit verification |
| **Domain tests** | `test_140_domains.py` | Full 140-domain TPC coverage test |
| **Physics tests** | `test_advanced_physics.py`, `test_boundary_conditions.py`, `test_boundary_layer.py`, `test_ballistics.py` | Physics validation |
| **Integration** | `integration/`, `integration_suite/` | Cross-module integration tests |
| **Benchmarks** | `benchmarks/` | Performance regression tests |
| **Alpha acceptance** | `test_alpha_acceptance.py` | Alpha release gate |
| **Billing** | `test_billing.py` | Billing system tests |

---

## 13. Documentation & Governance

### `docs/` — Documentation Tree

| Sub-directory | Description |
|---------------|-------------|
| `INDEX.md` | Documentation index |
| `ONBOARDING.md` | Developer onboarding guide |
| `PHYSICS_INVENTORY.md` | Physics domain inventory |
| `adr/` | Architecture Decision Records (25+) |
| `api/` | API reference documentation |
| `architecture/` | Architecture documentation |
| `attestations/` | Validation attestation artifacts |
| `audit/`, `audits/` | Audit reports and findings |
| `commercial/` | Commercial documentation |
| `domains/` | Physics domain documentation |
| `evolution/` | Platform evolution history |
| `getting-started/` | Quickstart guides |
| `governance/` | Governance documentation |
| `images/`, `media/` | Visual assets |
| `legacy/` | Legacy documentation archive |
| `operations/` | Operations runbooks |
| `papers/` | Research papers |
| `phases/` | Development phase documentation |
| `product/` | Product documentation |
| `regulatory/` | Regulatory compliance |
| `reports/` | Generated reports |
| `research/` | Research documentation |
| `roadmaps/` | Strategic roadmaps |
| `specifications/` | Technical specifications |
| `strategy/` | Business strategy docs |
| `tutorials/` | Step-by-step tutorials |
| `workflows/` | Workflow documentation |

### Top-Level Governance Documents

| Document | Description |
|----------|-------------|
| `CONSTITUTION.md` | Project constitution and operating principles |
| `CODE_OF_CONDUCT.md` | Community code of conduct |
| `CONTRIBUTING.md` | Contribution guidelines |
| `SECURITY.md` | Security policy |
| `SECURITY_OPERATIONS.md` | Security operations runbook |
| `LICENSE` | Project license |
| `CODEOWNERS` | Code ownership map |
| `CITATION.cff` | Citation metadata |
| `CHANGELOG.md` | Version changelog |
| `PLATFORM_SPECIFICATION.md` | Authoritative platform specification (~2,068 lines) |
| `ARCHITECTURE.md` | System architecture |
| `ROADMAP.md` | Strategic roadmap |
| `README.md` | Repository README |
| `API_SURFACE_FREEZE.md` | API stability guarantees |
| `DETERMINISM_ENVELOPE.md` | Determinism guarantees |
| `ERROR_CODE_MATRIX.md` | Error code reference |
| `FORBIDDEN_OUTPUTS.md` | Output safety constraints |
| `METERING_POLICY.md` | Usage metering policy |
| `PRICING_MODEL.md` | Pricing model |
| `QUEUE_BEHAVIOR_SPEC.md` | Queue behavior specification |
| `OPERATIONS_RUNBOOK.md` | Operations playbook |
| `LAUNCH_READINESS.md` | Launch readiness checklist |
| `CLAIM_REGISTRY.md` | Scientific claim registry |
| `DOMAIN_PACK_AUDIT.md` | Domain pack audit report |
| `CERTIFICATE_TEST_MATRIX.md` | TPC test matrix |

### `challenges/` — Civilization Challenges (6 files)

| Challenge | Description |
|-----------|-------------|
| I | Grid Stability |
| II | Pandemic Preparedness |
| III | Climate Tipping Points |
| IV | Fusion Energy |
| V | Supply Chain |
| VI | Proof of Reality |

---

## 14. Backward-Compatibility Shim Map

The `tensornet/` package provides flat re-export shims so that deeply nested
modules can be imported from short paths. Each shim `__init__.py` re-exports
from the canonical location.

| Shim Path | Canonical Location |
|-----------|--------------------|
| `tensornet/acoustics/` | `tensornet.applied.acoustics` |
| `tensornet/adaptive/` | `tensornet.engine.adaptive` |
| `tensornet/agri/` | `tensornet.energy_env.agri` |
| `tensornet/algorithms/` | `tensornet.core.algorithms` |
| `tensornet/astro/` | `tensornet.applied.special_applied` |
| `tensornet/autonomy/` | `tensornet.aerospace.autonomy` |
| `tensornet/biology/` | `tensornet.life_sci.biology` |
| `tensornet/biomedical/` | `tensornet.life_sci.biomedical` |
| `tensornet/biophysics/` | `tensornet.life_sci.biophysics` |
| `tensornet/certification/` | `tensornet.platform.certification` |
| `tensornet/chemistry/` | `tensornet.life_sci.chemistry` |
| `tensornet/computational_methods/` | `tensornet.fluids.computational_methods` |
| `tensornet/condensed_matter/` | `tensornet.quantum.condensed_matter` |
| `tensornet/coordination/` | `tensornet.infra.coordination` |
| `tensornet/coupled/` | `tensornet.fluids.coupled` |
| `tensornet/cyber/` | `tensornet.applied.cyber` |
| `tensornet/data/` | `tensornet.platform.data` |
| `tensornet/defense/` | `tensornet.aerospace.defense` |
| `tensornet/deployment/` | `tensornet.infra.deployment` |
| `tensornet/digital_twin/` | `tensornet.infra.digital_twin` |
| `tensornet/discovery/` | `tensornet.ml.discovery` |
| `tensornet/distributed/` | `tensornet.engine.distributed` |
| `tensornet/distributed_tn/` | `tensornet.engine.distributed_tn` |
| `tensornet/electronic_structure/` | `tensornet.quantum.electronic_structure` |
| `tensornet/emergency/` | `tensornet.applied.emergency` |
| `tensornet/energy/` | `tensornet.energy_env.energy` |
| `tensornet/environmental/` | `tensornet.energy_env.environmental` |
| `tensornet/exploit/` | `tensornet.aerospace.exploit` |
| `tensornet/fieldops/` | `tensornet.infra.fieldops` |
| `tensornet/fieldos/` | `tensornet.infra.fieldos` |
| `tensornet/financial/` | `tensornet.applied.financial` |
| `tensornet/flight_validation/` | `tensornet.aerospace.flight_validation` |
| `tensornet/free_surface/` | `tensornet.fluids.free_surface` |
| `tensornet/fsi/` | `tensornet.fluids.fsi` |
| `tensornet/fuel/` | `tensornet.engine.fuel` |
| `tensornet/fusion/` | `tensornet.plasma_nuclear.fusion` |
| `tensornet/gateway/` | `tensornet.engine.gateway` |
| `tensornet/geophysics/` | `tensornet.applied.geophysics` |
| `tensornet/gpu/` | `tensornet.engine.gpu` |
| `tensornet/guidance/` | `tensornet.aerospace.guidance` |
| `tensornet/hardware/` | `tensornet.engine.hardware` |
| `tensornet/heat_transfer/` | `tensornet.fluids.heat_transfer` |
| `tensornet/hw/` | `tensornet.engine.hw` |
| `tensornet/hyperenv/` | `tensornet.infra.hyperenv` |
| `tensornet/hypersim/` | `tensornet.infra.hypersim` |
| `tensornet/hypervisual/` | `tensornet.infra.hypervisual` |
| `tensornet/intent/` | `tensornet.applied.intent` |
| `tensornet/integration/` | `tensornet.infra.integration` |
| `tensornet/manufacturing/` | `tensornet.materials.manufacturing` |
| `tensornet/md/` | `tensornet.life_sci.md` |
| `tensornet/mechanics/` | `tensornet.materials.mechanics` |
| `tensornet/medical/` | `tensornet.applied.medical` |
| `tensornet/membrane_bio/` | `tensornet.life_sci.membrane_bio` |
| `tensornet/mesh_amr/` | `tensornet.fluids.mesh_amr` |
| `tensornet/ml_physics/` | `tensornet.ml.ml_physics` |
| `tensornet/ml_surrogates/` | `tensornet.ml.ml_surrogates` |
| `tensornet/multiphase/` | `tensornet.fluids.multiphase` |
| `tensornet/multiscale/` | `tensornet.fluids.multiscale` |
| `tensornet/neural/` | `tensornet.ml.neural` |
| `tensornet/nuclear/` | `tensornet.plasma_nuclear.nuclear` |
| `tensornet/optics/` | `tensornet.applied.optics` |
| `tensornet/oracle/` | `tensornet.infra.oracle` |
| `tensornet/particle/` | `tensornet.applied.particle` |
| `tensornet/phase_field/` | `tensornet.fluids.phase_field` |
| `tensornet/physics/` | `tensornet.applied.physics` |
| `tensornet/plasma/` | `tensornet.plasma_nuclear.plasma` |
| `tensornet/porous_media/` | `tensornet.fluids.porous_media` |
| `tensornet/provenance/` | `tensornet.infra.provenance` |
| `tensornet/qft/` | `tensornet.quantum.qft` |
| `tensornet/qm/` | `tensornet.quantum.qm` |
| `tensornet/quantum_mechanics/` | `tensornet.quantum.quantum_mechanics` |
| `tensornet/racing/` | `tensornet.aerospace.racing` |
| `tensornet/radiation/` | `tensornet.applied.radiation` |
| `tensornet/realtime/` | `tensornet.engine.realtime` |
| `tensornet/relativity/` | `tensornet.applied.relativity` |
| `tensornet/robotics_physics/` | `tensornet.applied.robotics_physics` |
| `tensornet/sdk/` | `tensornet.infra.sdk` |
| `tensornet/semiconductor/` | `tensornet.quantum.semiconductor` |
| `tensornet/sim/` | `tensornet.platform.sim` |
| `tensornet/site/` | `tensornet.infra.site` |
| `tensornet/sovereign/` | `tensornet.infra.sovereign` |
| `tensornet/special_applied/` | `tensornet.applied.special_applied` |
| `tensornet/statmech/` | `tensornet.quantum.statmech` |
| `tensornet/substrate/` | `tensornet.engine.substrate` |
| `tensornet/urban/` | `tensornet.energy_env.urban` |
| `tensornet/visualization/` | `tensornet.platform.visualization` |
| `tensornet/vm/` | `tensornet.engine.vm` |
| `tensornet/zk/` | `tensornet.infra.zk` |

---

## 15. Data & Archive

### `data/` — Datasets & Cached Artifacts

| Content | Description |
|---------|-------------|
| `6GJ8.pdb` | PDB structure file for drug design |
| `ahmed_body_data/`, `ahmed_body_results/`, `ahmed_ib_results/` | Ahmed body simulation data |
| `atlas/` | Rank atlas data |
| `cache/`, `pdb_cache/` | Cached computation results |
| `cases/` | Simulation case data |
| `models/`, `weights/` | Trained model weights |
| `noaa_24h_raw/` | NOAA 24-hour raw weather data |
| `real_data/`, `local_data/`, `sovereign_data/` | Domain-specific datasets |
| `*.json` | Benchmark results, fluid analysis, rank atlas data |
| `shakespeare.txt`, `wikitext2_*.txt` | Text corpora for TCI-LLM experiments |
| `HVAC_Blueprint.png` | HVAC blueprint for intake demo |

### `archive/` — Archived Artifacts

| Content | Description |
|---------|-------------|
| `FRONTIER.zip` | Archived frontier experiments |
| `QTT-FEA.zip`, `QTT-OPT.zip` | Archived QTT optimization/FEA |
| `fluidelite-zk.zip`, `fluidelite.zip` | FluidElite archives |
| `crates.zip`, `proofs.zip`, `demos.zip` | Bulk archives |
| `api_prototype/`, `api_prototype_content/` | API prototype archive |
| `dense/`, `dense_content/` | Dense solver reference implementations |
| `proof_of_concept/` | Early PoC code |
| `vendor/` | Vendored dependencies |

---

## Cross-Reference: Physics Domain Coverage

The 168 physics taxonomy nodes in `apps/ledger/nodes/*.yaml` map across the
TensorNet modules as follows:

| TPC Tier | Domain Count | Primary Module |
|----------|-------------|----------------|
| Tier 0 (Core) | 4 | `tensornet/cfd/` (Euler3D, NS-IMEX, Heat, Vlasov-Poisson) |
| Tier 1 (Fluid) | 8 | `tensornet/cfd/`, `tensornet/fluids/` |
| Tier 2A (25 domains) | 25 | `tensornet/em/`, `tensornet/plasma_nuclear/`, `tensornet/quantum/statmech/` |
| Tier 2B (40 domains) | 40 | `tensornet/materials/mechanics/`, `tensornet/applied/optics/`, `tensornet/quantum/`, `tensornet/fluids/coupled/` |
| Tier 3 (45 domains) | 45 | `tensornet/quantum/`, `tensornet/quantum/electronic_structure/`, `tensornet/quantum/condensed_matter/` |
| Tier 4 (26 domains) | 26 | `tensornet/life_sci/`, `tensornet/quantum/`, `tensornet/applied/` |
| **Total** | **168** | |

---

*This document was compiled by exhaustive file-level traversal of every
directory in the repository. The canonical machine-readable registry for the
168 taxonomy nodes is `apps/ledger/nodes/*.yaml`. For the authoritative
platform specification, see `PLATFORM_SPECIFICATION.md`.*
