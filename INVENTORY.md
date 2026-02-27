# Repository Inventory

> **Auto-generated**: 2026-02-27 | **Tracked files**: ~14,561 | **Languages**: 19 | **First-party LOC**: ~1.51M

Comprehensive index of every application, library, experiment, proof, product, tool, and infrastructure component in the HyperTensor-VM monorepo.

---

## Table of Contents

1. [Applications (`apps/`)](#1-applications)
2. [Rust Crates (`crates/`)](#2-rust-crates)
3. [Products (`products/`)](#3-products)
4. [Hypertensor Platform Service (`hypertensor/`)](#4-hypertensor-platform-service)
5. [Physics Engine Library (`tensornet/`)](#5-physics-engine-library)
6. [Domain Packs — 168 Taxonomy Nodes (`tensornet/packs/`)](#6-domain-packs--168-taxonomy-nodes)
7. [Formal Proofs (`proofs/`)](#7-formal-proofs)
8. [Smart Contracts (`contracts/`)](#8-smart-contracts)
9. [Experiments & R\&D (`experiments/`)](#9-experiments--rd)
10. [Challenges (`challenges/`)](#10-civilization-challenges)
11. [Test Suite (`tests/`)](#11-test-suite)
12. [Integrations (`integrations/`)](#12-integrations)
13. [Deployment (`deploy/`)](#13-deployment)
14. [Developer Tools (`tools/`)](#14-developer-tools)
15. [Documentation (`docs/`)](#15-documentation)
16. [CI/CD (`.github/workflows/`)](#16-cicd)
17. [Data & Artifacts](#17-data--artifacts)
18. [Archive (`archive/`)](#18-archive)
19. [Full 168-Node Taxonomy](#19-full-168-node-taxonomy)

---

## 1. Applications

Standalone, deployable applications and services.

| App | Path | Language | Description |
|-----|------|:--------:|-------------|
| **Glass Cockpit** | `apps/glass_cockpit/` | Python | Sovereign observation layer for atmospheric intelligence — real-time weather/satellite data visualization |
| **Glass Cockpit (Root)** | `apps/glass_cockpit_root/` | Rust | Root-level Glass Cockpit rendering engine |
| **Global Eye** | `apps/global_eye/` | Rust | Global monitoring and surveillance physics platform |
| **Golden Demo** | `apps/golden_demo/` | Rust | TPC end-to-end demo: simulate → prove → verify → visualize |
| **Capability Ledger** | `apps/ledger/` | Python/YAML | 168-node taxonomy registry, dashboard generator, schema validation |
| **Galaxy Feed Oracle** | `apps/oracle/` | Python | Real-time market entropy analysis with GPU-accelerated regime detection |
| **Tensor Genesis Oracle Node** | `apps/oracle_node/` | Python | Domain-agnostic structure engine — "The Universal Truth Machine" |
| **QTeneT** | `apps/qtenet/` | Python | Quantized Tensor Network Physics Engine — O(log N) complexity |
| **Enterprise SDK (Legacy)** | `apps/sdk_legacy/` | Python | HyperTensor Enterprise SDK v1.0.0 |
| **Sovereign API** | `apps/sovereign_api/` | Python | Real-time data provider and sovereign physics API server |
| **Sovereign UI** | `apps/sovereign_ui/` | TypeScript | Frontend dashboard for sovereign physics data |
| **The Compressor** | `apps/the_compressor/` | Python | 63,321× QTT compression engine for satellite/volumetric data |
| **Trustless Verify** | `apps/trustless_verify/` | Rust | TPC binary verifier with Ed25519 signature validation + explorer UI |
| **Vlasov Proof** | `apps/vlasov_proof/` | Rust | Standalone Vlasov equation ZK proof application |
| **Exploit Verification** | `apps/exploit_verification/` | Python/Solidity | Pendle oracle manipulation forensic verification suite |

---

## 2. Rust Crates

Native Rust libraries providing performance-critical computation, ZK circuits, and GPU acceleration.

| Crate | Path | Description |
|-------|------|-------------|
| **fluidelite** | `crates/fluidelite/` | FluidElite CFD optimization and benchmarking framework |
| **fluidelite_circuits** | `crates/fluidelite_circuits/` | Physics ZK circuits: Euler 3D, Navier-Stokes IMEX, thermal diffusion (Halo2) |
| **fluidelite_core** | `crates/fluidelite_core/` | Core tensor primitives: Q16 field, MPS, MPO, operators, physics traits |
| **fluidelite_infra** | `crates/fluidelite_infra/` | Infrastructure: prover pools, Gevulot integration, dashboard, multi-tenancy |
| **fluidelite_zk** | `crates/fluidelite_zk/` | ZK-provable FluidElite inference using Halo2 — deployment-ready |
| **gevulot** | `crates/gevulot/` | Gevulot decentralized prover registration and task definitions |
| **hyper_bridge** | `crates/hyper_bridge/` | RAM Bridge IPC protocol for Python ↔ Rust tensor streaming |
| **hyper_core** | `crates/hyper_core/` | HyperTensor physics engine core: QTT, MPO, CFD operators |
| **hyper_gpu_py** | `crates/hyper_gpu_py/` | PyO3 bindings for CUDA GPU acceleration (97M queries/sec) |
| **proof_bridge** | `crates/proof_bridge/` | Bridge between Python computation traces and ZK proof generation |
| **qtt_cem** | `crates/qtt_cem/` | Computational Electromagnetics via QTT — FDTD Maxwell solver (Q16.16) |
| **qtt_fea** | `crates/qtt_fea/` | Structural Mechanics FEA via QTT — static linear elasticity (Q16.16) |
| **qtt_opt** | `crates/qtt_opt/` | PDE-Constrained Optimization via adjoint methods — SIMP topology optimization |
| **tci_core** | `crates/tci_core/` | Tensor Cross Interpolation for native QTT construction (PyO3) |
| **tci_core_rust** | `crates/tci_core_rust/` | TT-Cross Interpolation core algorithms — pure Rust |

---

## 3. Products

Deployable product packages ready for end users or customers.

| Product | Path | Description |
|---------|------|-------------|
| **Facial Plastics** | `products/facial_plastics/` | Surgical simulation platform — anatomy generation, pre/post-op analysis, clinical governance, UI, Docker deployment |
| **FluidElite** | `products/fluidelite/` | FluidElite compute kernels package |
| **FluidElite-ZK** | `products/fluidelite-zk/` | Gevulot-deployed ZK prover image for trustless physics verification |
| **The Compressor** | `products/the_compressor/` | Pre-compressed QTT datasets (NOAA 24h, hybrid, block SVD outputs) |

---

## 4. Hypertensor Platform Service

The runtime platform — API server, CLI, billing, SDK, and MCP server.

| Module | Path | Description |
|--------|------|-------------|
| **API Server** | `hypertensor/api/` | FastAPI app with auth, config, and routers (capabilities, contracts, health, jobs, validate) |
| **Billing** | `hypertensor/billing/` | Metering and invoice generation for platform usage |
| **CLI** | `hypertensor/cli/` | Command-line interface for job submission and management |
| **Core** | `hypertensor/core/` | Executor, certificates, hasher, sanitizer, evidence collection, registry |
| **Jobs** | `hypertensor/jobs/` | Job models and persistent store |
| **MCP Server** | `hypertensor/mcp/` | Model Context Protocol server for AI-assisted physics |
| **SDK** | `hypertensor/sdk/` | Python SDK client for programmatic platform access |

---

## 5. Physics Engine Library

`tensornet/` — the core physics engine with **117 sub-modules** spanning every domain.

### Core Infrastructure

| Module | Path | Files | Description |
|--------|------|------:|-------------|
| **core** | `tensornet/core/` | — | Core tensor operations, decompositions, and data structures |
| **qtt** | `tensornet/qtt/` | 12 | Quantized Tensor Train compression, decomposition, arithmetic |
| **mps** | `tensornet/mps/` | 3 | Matrix Product State representations and operations |
| **mpo** | `tensornet/mpo/` | 6 | Matrix Product Operator construction and application |
| **algorithms** | `tensornet/algorithms/` | 15 | DMRG, TEBD, TDVP, TCI, variational methods |
| **numerics** | `tensornet/numerics/` | 13 | Numerical methods: quadrature, interpolation, differentiation |
| **types** | `tensornet/types/` | 15 | Type system for physics quantities, units, dimensions |

### Physics Domains

| Module | Path | Files | Domain |
|--------|------|------:|--------|
| **cfd** | `tensornet/cfd/` | 117 | Computational Fluid Dynamics — NS, Euler, RANS, LES, WENO, shock capturing |
| **fluids** | `tensornet/fluids/` | 11 | General fluid mechanics: multiphase, reactive, free-surface |
| **free_surface** | `tensornet/free_surface/` | 2 | Free surface and interfacial flow |
| **multiphase** | `tensornet/multiphase/` | 2 | Multiphase flow modeling |
| **em** | `tensornet/em/` | 20 | Computational Electromagnetics — FDTD, PML, antenna, CEM-QTT |
| **quantum** | `tensornet/quantum/` | 16 | Quantum computing: circuits, error correction, simulation |
| **quantum_mechanics** | `tensornet/quantum_mechanics/` | 2 | Schrödinger equation, semiclassical methods |
| **qm** | `tensornet/qm/` | 2 | Quantum mechanics primitives |
| **qft** | `tensornet/qft/` | 2 | Quantum field theory |
| **condensed_matter** | `tensornet/condensed_matter/` | 2 | DFT, Hubbard model, superconductors |
| **electronic_structure** | `tensornet/electronic_structure/` | 2 | Band structure, DFT, beyond-DFT methods |
| **plasma** | `tensornet/plasma/` | 2 | Plasma physics — Vlasov-Poisson, MHD, kinetic theory |
| **plasma_nuclear** | `tensornet/plasma_nuclear/` | 5 | Plasma-nuclear coupling, fusion physics |
| **fusion** | `tensornet/fusion/` | 2 | Fusion energy: tokamak, MARRS, disruption prediction |
| **nuclear** | `tensornet/nuclear/` | 2 | Nuclear structure, reactions, astrophysics |
| **particle** | `tensornet/particle/` | 2 | Particle physics: lattice QCD, pQFT |
| **mechanics** | `tensornet/mechanics/` | 2 | Structural mechanics and elasticity |
| **materials** | `tensornet/materials/` | 11 | Materials science: phase-field, radiation damage, polymers |
| **heat_transfer** | `tensornet/heat_transfer/` | 2 | Conduction, convection, radiation heat transfer |
| **acoustics** | `tensornet/acoustics/` | 2 | Acoustics and vibration |
| **optics** | `tensornet/optics/` | 2 | Physical optics, laser physics, photonics |
| **relativity** | `tensornet/relativity/` | 2 | General and special relativity, geodesics |
| **statmech** | `tensornet/statmech/` | 2 | Statistical mechanics: equilibrium, non-equilibrium |
| **geophysics** | `tensornet/geophysics/` | 2 | Seismology, mantle convection, geomagnetism |
| **astro** | `tensornet/astro/` | 11 | Astrophysics: stellar evolution, gravitational waves, cosmology |
| **chemistry** | `tensornet/chemistry/` | 2 | Reaction kinetics, thermodynamics |
| **biology** | `tensornet/biology/` | 2 | Computational biology |
| **biophysics** | `tensornet/biophysics/` | 1 | Biophysics modeling |
| **membrane_bio** | `tensornet/membrane_bio/` | 2 | Membrane biophysics |
| **neural** | `tensornet/neural/` | 1 | Computational neuroscience |
| **phase_field** | `tensornet/phase_field/` | 2 | Phase-field modeling |
| **porous_media** | `tensornet/porous_media/` | 2 | Porous media flow |
| **radiation** | `tensornet/radiation/` | 2 | Radiation transport and damage |
| **fsi** | `tensornet/fsi/` | 2 | Fluid-structure interaction |
| **coupled** | `tensornet/coupled/` | 2 | Multi-physics coupling |
| **multiscale** | `tensornet/multiscale/` | 2 | Multiscale methods |
| **md** | `tensornet/md/` | 2 | Molecular dynamics |

### Genesis Layers (8 Meta-Primitives)

| Module | Path | Description |
|--------|------|-------------|
| **QTT-OT** | `tensornet/genesis/ot/` | Trillion-point Sinkhorn optimal transport — O(r³ log N) |
| **QTT-SGW** | `tensornet/genesis/sgw/` | Billion-node spectral graph wavelets with Chebyshev filters |
| **QTT-RMT** | `tensornet/genesis/rmt/` | Random matrix eigenvalue statistics without dense storage |
| **QTT-TG** | `tensornet/genesis/tropical/` | Shortest paths via tropical min-plus semiring algebra |
| **QTT-RKHS** | `tensornet/genesis/rkhs/` | Trillion-sample Gaussian processes with QTT kernel matrices |
| **QTT-PH** | `tensornet/genesis/topology/` | Persistent homology (Betti numbers) at scale |
| **QTT-GA** | `tensornet/genesis/ga/` | Clifford algebras Cl(p,q,r) — Cl(50) in KB, not PB |
| **QTT-Aging** | `tensornet/genesis/aging/` | Biological aging as rank growth; reversal as rank reduction |

### Applied Verticals & Industry

| Module | Path | Files | Domain |
|--------|------|------:|--------|
| **aerospace** | `tensornet/aerospace/` | 7 | Aerodynamics, flight validation |
| **applied** | `tensornet/applied/` | 15 | Applied physics across domains |
| **biomedical** | `tensornet/biomedical/` | 2 | Biomedical engineering |
| **medical** | `tensornet/medical/` | 2 | Medical physics |
| **life_sci** | `tensornet/life_sci/` | 8 | Life sciences: drug discovery, pharmacokinetics |
| **energy** | `tensornet/energy/` | 2 | Energy systems |
| **energy_env** | `tensornet/energy_env/` | 6 | Energy and environment |
| **environmental** | `tensornet/environmental/` | 2 | Environmental modeling |
| **manufacturing** | `tensornet/manufacturing/` | 2 | Manufacturing processes |
| **semiconductor** | `tensornet/semiconductor/` | 1 | Semiconductor physics |
| **financial** | `tensornet/financial/` | 1 | Financial physics models |
| **racing** | `tensornet/racing/` | 1 | Motorsport physics |
| **defense** | `tensornet/defense/` | 1 | Defense applications |
| **urban** | `tensornet/urban/` | 1 | Urban physics and infrastructure |
| **agri** | `tensornet/agri/` | 1 | Agricultural modeling |
| **cyber** | `tensornet/cyber/` | 1 | Cybersecurity physics |
| **emergency** | `tensornet/emergency/` | 1 | Emergency response modeling |
| **robotics_physics** | `tensornet/robotics_physics/` | 2 | Robotics physics simulation |
| **autonomy** | `tensornet/autonomy/` | 1 | Autonomous systems |
| **guidance** | `tensornet/guidance/` | 2 | Guidance, navigation, and control |
| **fuel** | `tensornet/fuel/` | 1 | Fuel systems and combustion |
| **special_applied** | `tensornet/special_applied/` | 1 | Special-purpose applied physics |
| **digital_twin** | `tensornet/digital_twin/` | 1 | Digital twin framework |
| **discovery** | `tensornet/discovery/` | 1 | Physics discovery engine |

### Platform & Infrastructure

| Module | Path | Files | Description |
|--------|------|------:|-------------|
| **platform** | `tensornet/platform/` | 35 | Platform orchestration, job pipeline, capability registry |
| **engine** | `tensornet/engine/` | 13 | Simulation engine: scheduler, execution, state management |
| **infra** | `tensornet/infra/` | 17 | Infrastructure: logging, metrics, deployment, configuration |
| **sim** | `tensornet/sim/` | 8 | Simulation framework |
| **substrate** | `tensornet/substrate/` | 1 | Field Oracle substrate — canonical `sample/slice/step/stats` API |
| **fieldops** | `tensornet/fieldops/` | 1 | Physics operators as composable FieldGraph nodes |
| **fieldos** | `tensornet/fieldos/` | 1 | Field OS layer |
| **validation** | `tensornet/validation/` | 1 | V&V (Verification & Validation) framework |
| **certification** | `tensornet/certification/` | 1 | Physics certification engine |
| **provenance** | `tensornet/provenance/` | 1 | Data provenance and lineage tracking |
| **sovereign** | `tensornet/sovereign/` | 1 | Sovereign data pipeline |
| **oracle** | `tensornet/oracle/` | 2 | Oracle subsystem |
| **zk** | `tensornet/zk/` | 1 | Zero-knowledge proof integration |
| **vm** | `tensornet/vm/` | 2 | Register-based virtual machine |
| **packs** | `tensornet/packs/` | 24 | 20 domain packs (168 taxonomy nodes) |

### ML, Visualization, and Compute

| Module | Path | Files | Description |
|--------|------|------:|-------------|
| **ml** | `tensornet/ml/` | 7 | Machine learning for physics |
| **ml_physics** | `tensornet/ml_physics/` | 2 | ML-physics hybrid models |
| **ml_surrogates** | `tensornet/ml_surrogates/` | 1 | ML surrogate models |
| **cuda** | `tensornet/cuda/` | 10 | CUDA GPU kernels |
| **gpu** | `tensornet/gpu/` | 2 | GPU compute abstraction |
| **hw** | `tensornet/hw/` | 1 | Hardware interface |
| **hardware** | `tensornet/hardware/` | 1 | Hardware configuration |
| **distributed** | `tensornet/distributed/` | 1 | Distributed computing |
| **distributed_tn** | `tensornet/distributed_tn/` | 2 | Distributed tensor networks |
| **hypervisual** | `tensornet/hypervisual/` | 1 | Real-time QTT field rendering (Glass Cockpit) |
| **visualization** | `tensornet/visualization/` | 1 | Data visualization framework |
| **shaders** | `tensornet/shaders/` | 1 | GPU shader programs |
| **mesh_amr** | `tensornet/mesh_amr/` | 2 | Mesh generation and adaptive mesh refinement |
| **adaptive** | `tensornet/adaptive/` | 1 | Adaptive methods |
| **realtime** | `tensornet/realtime/` | 1 | Real-time simulation |
| **benchmarks** | `tensornet/benchmarks/` | 1 | Performance benchmarks |

### Remaining Modules

| Module | Path | Files | Description |
|--------|------|------:|-------------|
| **computational_methods** | `tensornet/computational_methods/` | 1 | General computational methods |
| **coordination** | `tensornet/coordination/` | 1 | Multi-solver coordination |
| **data** | `tensornet/data/` | 1 | Data management |
| **deployment** | `tensornet/deployment/` | 1 | Deployment utilities |
| **exploit** | `tensornet/exploit/` | 1 | DeFi exploit detection physics |
| **flight_validation** | `tensornet/flight_validation/` | 1 | Flight validation benchmarks |
| **gateway** | `tensornet/gateway/` | 1 | API gateway |
| **hyperenv** | `tensornet/hyperenv/` | 1 | HyperEnvironment simulation |
| **hypersim** | `tensornet/hypersim/` | 1 | HyperSim multi-domain simulator |
| **integration** | `tensornet/integration/` | 1 | Cross-module integration |
| **intent** | `tensornet/intent/` | 1 | Natural-language intent engine |
| **physics** | `tensornet/physics/` | 1 | General physics utilities |
| **sdk** | `tensornet/sdk/` | 1 | Internal SDK |
| **simulation** | `tensornet/simulation/` | 1 | Simulation primitives |
| **site** | `tensornet/site/` | 1 | Documentation site generator |

---

## 6. Domain Packs — 168 Taxonomy Nodes

Each domain pack corresponds to a physics vertical. All 168 nodes are registered in `apps/ledger/nodes/*.yaml`.

| Pack | Domain | Nodes | Module |
|------|--------|------:|--------|
| PHY-I | Classical Mechanics | 8 | `tensornet/packs/pack_i.py` |
| PHY-II | Classical Fluids (CFD) | 10 | `tensornet/packs/pack_ii.py` |
| PHY-III | Electromagnetics (CEM) | 7 | `tensornet/packs/pack_iii.py` |
| PHY-IV | Optics | 7 | `tensornet/packs/pack_iv.py` |
| PHY-V | Statistical Mechanics | 6 | `tensornet/packs/pack_v.py` |
| PHY-VI | Quantum Mechanics | 10 | `tensornet/packs/pack_vi.py` |
| PHY-VII | Quantum Many-Body | 13 | `tensornet/packs/pack_vii.py` |
| PHY-VIII | Electronic Structure | 10 | `tensornet/packs/pack_viii.py` |
| PHY-IX | Condensed Matter | 8 | `tensornet/packs/pack_ix.py` |
| PHY-X | Nuclear & Particle | 9 | `tensornet/packs/pack_x.py` |
| PHY-XI | Plasma Physics | 10 | `tensornet/packs/pack_xi.py` |
| PHY-XII | Astrophysics | 10 | `tensornet/packs/pack_xii.py` |
| PHY-XIII | Geophysics | 8 | `tensornet/packs/pack_xiii.py` |
| PHY-XIV | Materials Science | 8 | `tensornet/packs/pack_xiv.py` |
| PHY-XV | Chemical Physics | 8 | `tensornet/packs/pack_xv.py` |
| PHY-XVI | Biophysics | 8 | `tensornet/packs/pack_xvi.py` |
| PHY-XVII | Computational Methods | 6 | `tensornet/packs/pack_xvii.py` |
| PHY-XVIII | Multi-Physics Coupling | 8 | `tensornet/packs/pack_xviii.py` |
| PHY-XIX | Quantum Computing | 8 | `tensornet/packs/pack_xix.py` |
| PHY-XX | Special & Applied | 6 | `tensornet/packs/pack_xx.py` |
| | **Total** | **168** | |

Full node-level listing: [§19 Full 168-Node Taxonomy](#19-full-168-node-taxonomy).

---

## 7. Formal Proofs

Mathematical and cryptographic verification spanning Lean 4, Python proof engines, and ZK circuits.

### Lean 4 Formal Proofs (27 Conservation + 6 Core = 33 first-party `.lean` files)

| Proof | Path | Domain |
|-------|------|--------|
| **EulerConservation** | `proofs/conservation/euler/` | Euler equation conservation laws |
| **FluidConservation** | `proofs/conservation/fluid/` | General fluid conservation |
| **PlasmaConservation** | `proofs/conservation/plasma/` | Vlasov/plasma conservation |
| **QuantumManyBodyConservation** | `proofs/conservation/qmb/` | Quantum many-body conservation |
| **QuantumMechanicsConservation** | `proofs/conservation/quantum_mechanics/` | QM conservation laws |
| **QuantumInfoConservation** | `proofs/conservation/quantum_info/` | Quantum information |
| **QuantumInfoExtConservation** | `proofs/conservation/quantum_info_ext/` | Extended quantum info |
| **EMConservation** | `proofs/conservation/em/` | Electromagnetic conservation |
| **MechanicsConservation** | `proofs/conservation/mechanics/` | Classical mechanics |
| **ThermalConservation** | `proofs/conservation/thermal/` | Thermal/heat conservation |
| **StatMechConservation** | `proofs/conservation/statmech/` | Statistical mechanics |
| **StatMechStochasticConservation** | `proofs/conservation/statmech_stochastic/` | Stochastic stat mech |
| **MaterialsConservation** | `proofs/conservation/materials/` | Materials science |
| **SolidStateConservation** | `proofs/conservation/solid_state/` | Solid state physics |
| **ElectronicStructureConservation** | `proofs/conservation/electronic_structure/` | Electronic structure |
| **OpticsConservation** | `proofs/conservation/optics/` | Optics |
| **NuclearParticleConservation** | `proofs/conservation/nuclear_particle/` | Nuclear & particle |
| **AstroConservation** | `proofs/conservation/astro/` | Astrophysics |
| **GeophysicsConservation** | `proofs/conservation/geophysics/` | Geophysics |
| **BiophysicsConservation** | `proofs/conservation/biophysics/` | Biophysics |
| **ChemicalPhysicsConservation** | `proofs/conservation/chemical_physics/` | Chemical physics |
| **ChemPhysicsIterConservation** | `proofs/conservation/chem_physics_iter/` | Iterative chem physics |
| **AppliedPhysicsConservation** | `proofs/conservation/applied_physics/` | Applied physics |
| **ComputationalMethodsConservation** | `proofs/conservation/computational_methods/` | Comp methods |
| **CoupledConservation** | `proofs/conservation/coupled/` | Coupled systems |
| **SpecialRelativityConservation** | `proofs/conservation/special_relativity/` | Special relativity |
| **VlasovConservation** | `proofs/conservation/vlasov/` | Vlasov equation |
| **NavierStokes v1** | `proofs/navier_stokes/v1/` | Navier-Stokes conservation |
| **NavierStokes v2** | `proofs/navier_stokes/v2/` | Navier-Stokes regularity |
| **YangMills (multiple)** | `proofs/yang_mills/` | Yang-Mills mass gap — 6 Lean variants |

### Python Proof Engine

| Module | Path | Description |
|--------|------|-------------|
| **Proof Engine** | `proofs/proof_engine/` | Certificate generation, convergence proofs, Coq/Lean/Isabelle export, dashboard |
| **TPC Library** | `proofs/tpc/` | Trustless Physics Certificate format, constants, registry, generator |
| **Yang-Mills Core** | `proofs/yang_mills/core/` | SU(2) lattice gauge, Hamiltonian, DMRG, transfer matrix, scaling analysis |
| **Proof Pipeline** | `proofs/proof_phase_*.py` | 8 phased proof scripts (1a → 6) plus production, plasma, summary |

### ZK Circuit References

| Circuit | Path | Description |
|---------|------|-------------|
| **Circom ECDSA** | `proofs/zk_circuits/circom_ecdsa/` | ECDSA signature circuits |
| **Semaphore** | `proofs/zk_circuits/semaphore/` | Privacy-preserving identity circuits |
| **Tornado** | `proofs/zk_circuits/tornado/` | Tornado Cash circuits (reference) |
| **Worldcoin** | `proofs/zk_circuits/worldcoin_circuits/` | Worldcoin identity circuits (reference) |
| **Worldcoin ID** | `proofs/zk_circuits/worldcoin_id/` | World ID verification (reference) |

### ZK Target References

| Target | Path | Description |
|--------|------|-------------|
| **DeGate/Loopring** | `proofs/zk_targets/degate_protocols/` | Loopring v3 DEX circuits |
| **Light Protocol** | `proofs/zk_targets/light-protocol/` | Compressed state proofs |
| **Linea** | `proofs/zk_targets/linea-circuits/` | Linea zkEVM circuits |
| **Noir** | `proofs/zk_targets/noir-lang/` | Noir DSL circuits |
| **OpenTitan** | `proofs/zk_targets/opentitan/` | Hardware security module |

---

## 8. Smart Contracts

On-chain verification and API contracts.

| Contract | Path | Description |
|----------|------|-------------|
| **FluidEliteHalo2Verifier** | `contracts/FluidEliteHalo2Verifier.sol` | On-chain Halo2 proof verification for FluidElite physics |
| **HyperTensorBindingVerifier** | `contracts/HyperTensorBindingVerifier.sol` | Groth16 on-chain verifier for ZK drug-binding affinity proofs (EIP-197) |
| **ZeroExpansionSemaphoreVerifier** | `contracts/ZeroExpansionSemaphoreVerifier.sol` | Semaphore identity verification for zero-expansion proofs |
| **API Contract v1** | `contracts/v1/` | OpenAPI spec, JSON schema, and SPEC.md for the platform API |

---

## 9. Experiments & R\&D

Research experiments, validation pipelines, demos, and proof-of-concept work.

### Validation Pipelines (Challenge II — Pandemic Preparedness)

| Pipeline | Path | Description |
|----------|------|-------------|
| **TIG-011a MD Validation** | `experiments/validation/tig011a_md_validation.py` | Phase 1: Molecular dynamics of TIG-011a binding (Velocity Verlet, Nosé-Hoover, MM-GBSA) |
| **10K Drug Library** | `experiments/validation/challenge_ii_phase2_library.py` | Phase 2: 10,000-candidate drug library (5 targets, scaffold × R-group) |
| **Binding Atlas** | `experiments/validation/challenge_ii_phase3_atlas.py` | Phase 3: Pre-computed atlas (40 structures, QTT compression) |
| **Pandemic Response** | `experiments/validation/challenge_ii_phase4_pandemic.py` | Phase 4: 7-target pandemic response pipeline (48-hour turnaround) |
| **ZK Binding Proofs** | `experiments/validation/challenge_ii_phase5_zk_proofs.py` | Phase 5: Zero-knowledge proofs, Merkle trees, Fiat-Shamir, on-chain verifier |

### Frontier — Trillion-Dollar Physics Applications

| Frontier | Path | Description |
|----------|------|-------------|
| **01 Fusion** | `experiments/frontier/01_FUSION/` | Fusion energy validation demo |
| **02 Space Weather** | `experiments/frontier/02_SPACE_WEATHER/` | Alfvén waves, bow shock, Sod shock tube |
| **03 Semiconductor Plasma** | `experiments/frontier/03_SEMICONDUCTOR_PLASMA/` | ICP discharge, plasma sheath, ion energy |
| **04 Particle Accelerator** | `experiments/frontier/04_PARTICLE_ACCELERATOR/` | Beam dynamics, RF cavity, space charge, wakefield |
| **05 Quantum Error Correction** | `experiments/frontier/05_QUANTUM_ERROR_CORRECTION/` | Surface codes, stabilizer formalism, threshold analysis |
| **06 Fusion Control** | `experiments/frontier/06_FUSION_CONTROL/` | Control loop, disruption predictor |
| **07 Genomics** | `experiments/frontier/07_GENOMICS/` | ClinVar, ENCODE, genome validation attestations |

### Demo Runners (46 demos)

| Demo | Path | Description |
|------|------|-------------|
| **Flagship Pipeline** | `experiments/demos/runners/flagship_pipeline.py` | End-to-end platform showcase |
| **Holy Grail Video** | `experiments/demos/runners/holy_grail_video.py` | Video-quality physics visualization |
| **HyperTensor Hub** | `experiments/demos/runners/hypertensor_hub.py` | Platform hub demo |
| **HyperTensor Pro** | `experiments/demos/runners/hypertensor_pro.py` | Pro-tier feature demo |
| **Physics-First Drug Design** | `experiments/demos/runners/physics_first_drug_design.py` | Drug discovery via physics |
| **Oracle Hunt** | `experiments/demos/runners/oracle_hunt_demo.py` | DeFi oracle manipulation detection |
| **Black Swan Forensics** | `experiments/demos/runners/black_swan_945_forensic.py` | Flash crash forensic analysis |
| **Trap the Swan** | `experiments/demos/runners/trap_the_swan.py` | Flash crash prediction |
| **Blue Marble** | `experiments/demos/runners/blue_marble.py` | Earth observation |
| **Kelvin-Helmholtz** | `experiments/demos/runners/kelvin_helmholtz_demo.py` | KH instability simulation |
| **CFD Shock** | `experiments/demos/runners/cfd_shock.py` | Shock tube demonstration |
| **Conference Room CFD** | `experiments/demos/runners/conference_room_cfd.py` | HVAC airflow simulation |
| **Millennium Hunter** | `experiments/demos/runners/millennium_hunter.py` | Millennium Prize problem hunter |
| **Weather Viewer** | `experiments/demos/runners/weather_viewer.py` | Weather data visualization |
| **World Data Slicer** | `experiments/demos/runners/world_data_slicer.py` | Global data slicing tool |
| **Resolution Independence** | `experiments/demos/runners/resolution_independence.py` | QTT resolution-independent demo |
| **Provable Physics** | `experiments/demos/runners/provable_physics.py` | Trustless physics demo |
| **Memory Wall** | `experiments/demos/runners/memory_wall.py` | Memory wall breaker demo |
| **Intent Demo** | `experiments/demos/runners/intent_demo.py` | Natural-language physics interface |
| **Layer 7 Physics RL** | `experiments/demos/runners/layer7_physics_rl.py` | Reinforcement learning for physics |
| **Layer 9 Engine Integration** | `experiments/demos/runners/layer9_engine_integration.py` | Full engine integration |
| *...and 25 more* | `experiments/demos/runners/` | See directory for complete list |

### Research Scripts

| Script | Path | Description |
|--------|------|-------------|
| **Helmholtz Rank Test** | `experiments/scripts/phase0_helmholtz_rank_test.py` | Phase 0 rank analysis |
| **Helmholtz QTT Solve** | `experiments/scripts/phase1_helmholtz_qtt_solve.py` | Phase 1 QTT solution |
| **Chu Limit 1024** | `experiments/scripts/run_chu_1024.py` | 1024-mode Chu limit computation |
| **Chu Limit Challenge** | `experiments/scripts/run_chu_limit_challenge.py` | Maximum Chu limit test |
| **Convergence Study** | `experiments/scripts/run_convergence_study.py` | Systematic convergence analysis |
| **Exascale Invention Sweep** | `experiments/scripts/run_exascale_invention_sweep.py` | Exascale-scale invention search |
| **Fast Invention Sweep** | `experiments/scripts/run_fast_invention_sweep.py` | Fast invention candidate search |
| **Kelvin-Helmholtz Run** | `experiments/scripts/run_kelvin_helmholtz.py` | KH instability execution |
| **Parametric Sweep** | `experiments/scripts/run_parametric_sweep.py` | Multi-parameter sweep |
| **GPU Bench** | `experiments/scripts/run_gpu_bench.sh` | GPU performance benchmarks |
| **Gradient Diagnostics** | `experiments/scripts/diag_gradient.py` | Gradient flow diagnostics |
| **Wave Port Validation** | `experiments/scripts/validate_wave_port.py` | CEM wave port validation |
| **Exascale Attestation Gen** | `experiments/scripts/generate_exascale_attestation.py` | Attestation generation |

### Other Experiments

| Experiment | Path | Description |
|------------|------|-------------|
| **AAVE Extraction** | `experiments/aave_extraction/` | DeFi AAVE protocol extraction forensics |
| **Ahmed Body** | `experiments/ahmed_body/` | NVIDIA Ahmed body aerodynamics (QTT pipeline) |
| **AI Scientist** | `experiments/ai_scientist/` | Automated physics discovery: conjecture → formalize → prove |
| **CFD-HVAC** | `experiments/cfd_hvac/` | Building ventilation CFD analysis suite |
| **Clawdbot** | `experiments/clawdbot/` | Personal AI assistant |
| **EtherFi Oracle Frontrun** | `experiments/etherfi_oracle_frontrun_poc/` | Oracle frontrunning proof-of-concept (Foundry) |
| **Flash Liquidator** | `experiments/flash_liquidator/` | DeFi flash liquidation toolkit (Foundry) |
| **HVAC-CFD** | `experiments/hvac_cfd/` | HyperFOAM HVAC execution charter and audit |
| **lUX** | `experiments/lux/` | lUX experimental framework |
| **Notebooks** | `experiments/notebooks/` | 6 Jupyter notebooks (Bose-Hubbard, Heisenberg, TEBD, TFIM, demo, integration) |
| **Orbital Frames** | `experiments/orbital_frames/` | Orbital mechanics frame renders |
| **Physics Standalone** | `experiments/physics_standalone/` | Standalone physics benchmarks and regression tests |
| **PWA Engine** | `experiments/pwa_engine/` | Partial Wave Analysis engine (Eq. 5.48, Badui dissertation) |
| **QTT 3D Ops** | `experiments/qtt_3d_ops/` | QTT-native 3D CFD operations |
| **Santa 2025** | `experiments/santa2025/` | Kaggle Santa 2025 competition submissions |
| **TCI-LLM** | `experiments/tci_llm/` | Gradient-free language modeling via Tensor Cross Interpolation |
| **Visualization** | `experiments/visualization/` | Report generation and rendering scripts |

---

## 10. Civilization Challenges

Six grand challenges positioning HyperTensor-VM against humanity's most critical problems.

| # | Challenge | Path | Status |
|:-:|-----------|------|:------:|
| I | **Grid Stability** | `challenges/challenge_I_grid_stability.md` | Defined |
| II | **Pandemic Preparedness** | `challenges/challenge_II_pandemic_preparedness.md` | Phase 1–5 Complete |
| III | **Climate Tipping Points** | `challenges/challenge_III_climate_tipping_points.md` | Defined |
| IV | **Fusion Energy** | `challenges/challenge_IV_fusion_energy.md` | Defined |
| V | **Supply Chain** | `challenges/challenge_V_supply_chain.md` | Defined |
| VI | **Proof of Reality** | `challenges/challenge_VI_proof_of_reality.md` | Defined |

---

## 11. Test Suite

104 test files spanning unit tests, integration tests, audit layers, and benchmarks.

### Integration Tests

| Test | Path | Description |
|------|------|-------------|
| Advection MMS | `tests/integration/test_advection_mms.py` | Method of Manufactured Solutions — advection |
| CFD Physics | `tests/integration/test_cfd_physics.py` | CFD correctness |
| DMRG Physics | `tests/integration/test_dmrg_physics.py` | DMRG accuracy |
| Euler 2D MMS | `tests/integration/test_euler2d_mms.py` | Euler 2D manufactured solutions |
| Euler 2D Physics | `tests/integration/test_euler2d_physics.py` | Euler 2D conservation |
| Euler 3D MMS | `tests/integration/test_euler3d_mms.py` | Euler 3D manufactured solutions |
| Flagship Pipeline | `tests/integration/test_flagship_pipeline.py` | End-to-end pipeline |
| Lid-Driven Cavity | `tests/integration/test_lid_driven_cavity.py` | Classic CFD benchmark |
| Poisson MMS | `tests/integration/test_poisson_mms.py` | Poisson manufactured solutions |
| Shu-Osher | `tests/integration/test_shu_osher_benchmark.py` | Shu-Osher shock benchmark |
| Taylor-Green | `tests/integration/test_taylor_green_benchmark.py` | Taylor-Green vortex |

### Audit Layers

| Layer | Path | Scope |
|:-----:|------|-------|
| 4 | `tests/audit_layer_4.py` | Audit layer 4 |
| 5 | `tests/audit_layer_5.py` | Audit layer 5 |
| 6 | `tests/audit_layer_6.py` | Audit layer 6 |
| 7 | `tests/audit_layer_7.py` | Audit layer 7 |
| 8 | `tests/audit_layer_8.py` | Audit layer 8 |
| 9 | `tests/audit_layer_9.py` | Audit layer 9 |

### Performance Benchmarks

| Benchmark | Path | Description |
|-----------|------|-------------|
| Optimized Pipeline | `tests/benchmarks/optimized_pipeline_benchmark.py` | Full pipeline benchmark |
| QTT Comprehensive | `tests/benchmarks/qtt_comprehensive_benchmark.py` | QTT operation benchmarks |
| QTT Full Pipeline | `tests/benchmarks/qtt_full_pipeline_benchmark.py` | End-to-end QTT benchmark |
| QTT Native | `tests/benchmarks/qtt_native_benchmark.py` | Native QTT ops benchmark |
| QTT Render | `tests/benchmarks/qtt_render_benchmark.py` | Rendering benchmark |

### Domain Tests (selected — 80+ unit tests)

Key test files include: `test_navier_stokes`, `test_fusion`, `test_quantum_physics`, `test_pwa_engine`, `test_exploit_hunter`, `test_financial`, `test_fire`, `test_ballistics`, `test_vlasov_genuine`, `test_racing`, `test_certificate_integrity`, `test_trustless_certificate`, `test_packs`, `test_140_domains`, `test_platform`, `test_billing`, `test_provenance`, and many more. Full listing: `tests/` directory.

---

## 12. Integrations

Game engine and visualization integrations.

| Integration | Path | Description |
|-------------|------|-------------|
| **Unity** | `integrations/unity/` | Unity package: Editor + Runtime HyperTensor plugin |
| **Unreal Engine** | `integrations/unreal/` | Unreal plugin with Python bridge for real-time physics |

---

## 13. Deployment

Infrastructure for containerized deployment and observability.

| Component | Path | Description |
|-----------|------|-------------|
| **Containerfile** | `deploy/Containerfile` | Main container image |
| **Config Container** | `deploy/config/Containerfile` | Configuration container |
| **Docker Compose** | `deploy/docker/` | Docker deployment orchestration |
| **Telemetry Stack** | `deploy/telemetry/` | Prometheus + alerts: `docker-compose.yml`, `prometheus.yml`, `alerts.yml` |

---

## 14. Developer Tools

Internal tooling for analytics, migrations, and reporting.

| Tool | Path | Description |
|------|------|-------------|
| **LOC Audit** | `tools/loc_audit.py` | Lines-of-code analysis across the monorepo |
| **Dep Graph** | `tools/dep_graph.py` | Dependency graph generator |
| **Sync Versions** | `tools/sync_versions.py` | Version synchronization across packages |
| **Forensic LOC Sweep** | `tools/forensic_loc_sweep.py` | Forensic code authorship analysis |
| **Forensic LOC Sweep v2** | `tools/forensic_loc_sweep_v2.py` | Enhanced forensic analysis |
| **Migrate TensorNet Phase 5** | `tools/migrate_tensornet_phase5.py` | Phase 5 migration script |
| **PDF Report Generator** | `tools/scripts/generate_pdf_report.py` | PDF report generation |
| **Update LOC Counts** | `tools/scripts/update_loc_counts.py` | Automated LOC metric updates |
| **Mach 5 Wedge** | `tools/scripts/mach5_wedge.py` | Mach 5 wedge flow calculator |
| **QTT Derivative Verify** | `tools/scripts/qtt_deriv_verify.py` | QTT derivative verification |
| **Packaging Gate** | `tools/scripts/packaging_gate.py` | Release packaging validation |
| **Wedge Flow Demo** | `tools/scripts/wedge_flow_demo.py` | Wedge flow demonstration |
| **Profile Performance** | `tools/scripts/profile_performance.py` | Performance profiling |
| **Phase 4 Integration** | `tools/scripts/run_phase4_integration.sh` | Phase 4 integration runner |
| **Glass Cockpit Launcher** | `tools/scripts/Start-GlassCockpit.ps1` | PowerShell Glass Cockpit starter |

---

## 15. Documentation

326+ Markdown documents across 22 subdirectories.

| Directory | Files | Topics |
|-----------|------:|--------|
| `docs/adr/` | 25 | Architecture Decision Records (ADR-0001 through ADR-0025) |
| `docs/api/` | 87 | API reference documentation |
| `docs/architecture/` | 14 | System architecture documents |
| `docs/attestations/` | 1 | Attestation documentation |
| `docs/audit/` | 4 | Security and code audit reports |
| `docs/audits/` | 9 | Extended audit documents |
| `docs/commercial/` | 3 | Commercial strategy and execution |
| `docs/domains/` | 11 | Domain-specific documentation (per physics vertical) |
| `docs/evolution/` | 1 | OS evolution strategy |
| `docs/getting-started/` | 3 | Quickstart guides |
| `docs/governance/` | 14 | Governance policies and registries |
| `docs/legacy/` | 15 | Legacy documentation |
| `docs/operations/` | 10 | Operations runbooks |
| `docs/papers/` | — | Research papers (PDF) |
| `docs/phases/` | 8 | Development phase documentation |
| `docs/product/` | 4 | Product documentation |
| `docs/regulatory/` | 1 | Regulatory compliance |
| `docs/reports/` | 50 | Validation and analysis reports |
| `docs/research/` | 42 | Research notes and findings |
| `docs/roadmaps/` | 4 | Product and technical roadmaps |
| `docs/specifications/` | 2 | Technical specifications |
| `docs/strategy/` | 5 | Strategic planning (Commercial Execution, OS Evolution, etc.) |
| `docs/tutorials/` | 3 | Tutorials |
| `docs/workflows/` | 6 | Workflow documentation |

### Top-Level Documents

| Document | Description |
|----------|-------------|
| `PLATFORM_SPECIFICATION.md` | Canonical platform specification (2,069 lines, 25 sections) |
| `ARCHITECTURE.md` | System architecture overview |
| `README.md` | Repository README |
| `ROADMAP.md` | Development roadmap |
| `CHANGELOG.md` | Version changelog |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CODE_OF_CONDUCT.md` | Code of conduct |
| `SECURITY.md` | Security policy |
| `LICENSE` | Apache 2.0 license |
| `NOTICE` | Third-party notice |
| `CITATION.cff` | Citation metadata |
| `CODEOWNERS` | Code ownership (289 lines) |

---

## 16. CI/CD

11 GitHub Actions workflows.

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR | Main CI pipeline |
| `nightly.yml` | Cron | Nightly full regression |
| `release.yml` | Tag | Release packaging and publishing |
| `docs.yml` | Push | Documentation build and deploy |
| `contracts-ci.yml` | Push/PR | Smart contract compilation and tests |
| `exploit-engine.yml` | Push/PR | Exploit detection engine tests |
| `facial-plastics-ci.yml` | Push/PR | Facial Plastics product CI |
| `hardening.yml` | Push/PR | Security hardening checks |
| `audit-gates.yml` | Push/PR | Audit gate validation |
| `ledger-validation.yml` | Push/PR | Capability ledger schema/integrity (168 nodes) |
| `vv-validation.yml` | Push/PR | V&V suite validation |

---

## 17. Data & Artifacts

### Data (`data/`)

| Directory | Description |
|-----------|-------------|
| `data/ahmed_body_data/` | Ahmed body aerodynamics input data |
| `data/ahmed_body_results/` | Ahmed body simulation results |
| `data/ahmed_ib_results/` | Ahmed immersed-boundary results |
| `data/atlas/` | Benchmark atlas reference data for taxonomy nodes |
| `data/cache/` | Computation cache |
| `data/cases/` | Test case definitions |
| `data/local_data/` | Local experimental data |
| `data/models/` | Pre-trained models |
| `data/pdb_cache/` | Protein Data Bank structure cache |
| `data/real_data/` | Real-world validation datasets |
| `data/sovereign_data/` | Sovereign data pipeline outputs |
| `data/weights/` | Model weights |

### Artifacts (`artifacts/`)

| Directory | Description |
|-----------|-------------|
| `artifacts/attestations/` | Signed attestation files |
| `artifacts/build-info/` | Build metadata |
| `artifacts/certificates/` | Trust certificates |
| `artifacts/evidence/` | V&V evidence packages |
| `artifacts/logs/` | Execution logs |
| `artifacts/outputs/` | Computation outputs |
| `artifacts/phase6/` – `artifacts/phase8/` | Phase-specific artifacts |
| `artifacts/profile_results/` | Performance profiles |
| `artifacts/results/` | General results |
| `artifacts/traces/` | Computation traces for ZK proving |

---

## 18. Archive

Historical and superseded components preserved for reference.

| Item | Description |
|------|-------------|
| `archive/api_prototype/` | Early API prototype |
| `archive/build_artifacts/` | Historical build outputs |
| `archive/dense/` | Pre-QTT dense implementations |
| `archive/proof_of_concept/` | Early proof-of-concept code |
| `archive/vendor/` | Vendored dependencies |
| `archive/*.zip` | Archived packages: FRONTIER, QTT-FEA, QTT-OPT, CEM-QTT, FluidElite-ZK, TCI-LLM, proofs, demos |

---

## 19. Full 168-Node Taxonomy

Every physics node in the capability ledger (`apps/ledger/nodes/*.yaml`).

### PHY-I — Classical Mechanics (8 nodes)

| Node | Name |
|------|------|
| PHY-I.1 | Newtonian Particle Dynamics |
| PHY-I.2 | Lagrangian/Hamiltonian Mechanics |
| PHY-I.3 | Continuum Mechanics |
| PHY-I.4 | Structural Mechanics |
| PHY-I.5 | Nonlinear Dynamics and Chaos |
| PHY-I.6 | Acoustics and Vibration |
| PHY-I.7 | Continuum Mechanics (extended) |
| PHY-I.8 | Chaos Theory |

### PHY-II — Classical Fluids / CFD (10 nodes)

| Node | Name |
|------|------|
| PHY-II.1 | Incompressible Navier-Stokes |
| PHY-II.2 | Compressible Flow |
| PHY-II.3 | Turbulence |
| PHY-II.4 | Multiphase Flow |
| PHY-II.5 | Reactive Flow / Combustion |
| PHY-II.6 | Rarefied Gas / Kinetic |
| PHY-II.7 | Shallow Water / Geophysical Fluids |
| PHY-II.8 | Non-Newtonian / Complex Fluids |
| PHY-II.9 | Porous Media |
| PHY-II.10 | Free Surface / Interfacial |

### PHY-III — Electromagnetics / CEM (7 nodes)

| Node | Name |
|------|------|
| PHY-III.1 | Electrostatics |
| PHY-III.2 | Magnetostatics |
| PHY-III.3 | Full Maxwell Time-Domain |
| PHY-III.4 | Frequency-Domain EM |
| PHY-III.5 | Wave Propagation |
| PHY-III.6 | Computational Photonics |
| PHY-III.7 | Antennas and Microwaves |

### PHY-IV — Optics (7 nodes)

| Node | Name |
|------|------|
| PHY-IV.1 | Physical Optics |
| PHY-IV.2 | Quantum Optics |
| PHY-IV.3 | Laser Physics |
| PHY-IV.4 | Ultrafast Optics |
| PHY-IV.5 | Nonlinear Optics |
| PHY-IV.6 | Quantum Optics (extended) |
| PHY-IV.7 | Photonic Crystal |

### PHY-V — Statistical Mechanics (6 nodes)

| Node | Name |
|------|------|
| PHY-V.1 | Equilibrium Statistical Mechanics |
| PHY-V.2 | Non-Equilibrium Statistical Mechanics |
| PHY-V.3 | Molecular Dynamics |
| PHY-V.4 | Monte Carlo Methods |
| PHY-V.5 | Heat Transfer |
| PHY-V.6 | Lattice Models and Spin Systems |

### PHY-VI — Quantum Mechanics (10 nodes)

| Node | Name |
|------|------|
| PHY-VI.1 | Time-Independent Schrödinger |
| PHY-VI.2 | TDSE Propagation |
| PHY-VI.3 | Scattering Theory |
| PHY-VI.4 | Semiclassical Methods |
| PHY-VI.5 | Path Integrals |
| PHY-VI.6 | Strongly Correlated |
| PHY-VI.7 | Mesoscopic Physics |
| PHY-VI.8 | Surface Physics |
| PHY-VI.9 | Disordered Systems |
| PHY-VI.10 | Phase Transitions |

### PHY-VII — Quantum Many-Body (13 nodes)

| Node | Name |
|------|------|
| PHY-VII.1 | Tensor Network Methods |
| PHY-VII.2 | Quantum Spin Systems |
| PHY-VII.3 | Strongly Correlated Electrons |
| PHY-VII.4 | Topological Phases |
| PHY-VII.5 | MBL and Disorder |
| PHY-VII.6 | Lattice Gauge Theory |
| PHY-VII.7 | Open Quantum Systems |
| PHY-VII.8 | Non-Equilibrium Quantum Dynamics |
| PHY-VII.9 | Quantum Impurity |
| PHY-VII.10 | Bosonic Many-Body |
| PHY-VII.11 | Fermionic Systems |
| PHY-VII.12 | Nuclear Many-Body |
| PHY-VII.13 | Ultracold Atoms |

### PHY-VIII — Electronic Structure (10 nodes)

| Node | Name |
|------|------|
| PHY-VIII.1 | DFT |
| PHY-VIII.2 | Beyond-DFT Correlated Methods |
| PHY-VIII.3 | Semi-Empirical and Tight-Binding |
| PHY-VIII.4 | Excited States |
| PHY-VIII.5 | Response Properties |
| PHY-VIII.6 | Relativistic Electronic Structure |
| PHY-VIII.7 | Quantum Embedding |
| PHY-VIII.8 | Response Functions |
| PHY-VIII.9 | Band Structure |
| PHY-VIII.10 | Ab Initio MD |

### PHY-IX — Condensed Matter (8 nodes)

| Node | Name |
|------|------|
| PHY-IX.1 | Phonons and Lattice Dynamics |
| PHY-IX.2 | Band Structure and Transport |
| PHY-IX.3 | Magnetism |
| PHY-IX.4 | Superconductivity |
| PHY-IX.5 | Disordered Systems |
| PHY-IX.6 | Surfaces and Interfaces |
| PHY-IX.7 | Defects in Solids |
| PHY-IX.8 | Ferroelectrics and Multiferroics |

### PHY-X — Nuclear & Particle (9 nodes)

| Node | Name |
|------|------|
| PHY-X.1 | Nuclear Structure |
| PHY-X.2 | Nuclear Reactions |
| PHY-X.3 | Nuclear Astrophysics |
| PHY-X.4 | Lattice QCD |
| PHY-X.5 | Perturbative QFT |
| PHY-X.6 | Beyond Standard Model |
| PHY-X.7 | Dark Matter |
| PHY-X.8 | Neutrino Physics |
| PHY-X.9 | Partial Wave Analysis |

### PHY-XI — Plasma Physics (10 nodes)

| Node | Name |
|------|------|
| PHY-XI.1 | Ideal MHD |
| PHY-XI.2 | Resistive/Extended MHD |
| PHY-XI.3 | Kinetic Theory Plasma |
| PHY-XI.4 | Gyrokinetics |
| PHY-XI.5 | Magnetic Reconnection |
| PHY-XI.6 | Laser-Plasma Interaction |
| PHY-XI.7 | Dusty Plasmas |
| PHY-XI.8 | Space and Astrophysical Plasma |
| PHY-XI.9 | Ion Acoustic Waves |
| PHY-XI.10 | Plasma Instabilities |

### PHY-XII — Astrophysics (10 nodes)

| Node | Name |
|------|------|
| PHY-XII.1 | Stellar Structure and Evolution |
| PHY-XII.2 | Compact Objects |
| PHY-XII.3 | Gravitational Waves |
| PHY-XII.4 | Cosmological Simulations |
| PHY-XII.5 | CMB and Early Universe |
| PHY-XII.6 | Radiative Transfer Astrophysical |
| PHY-XII.7 | Accretion |
| PHY-XII.8 | Radiation Transport |
| PHY-XII.9 | Dark Energy |
| PHY-XII.10 | CMB |

### PHY-XIII — Geophysics (8 nodes)

| Node | Name |
|------|------|
| PHY-XIII.1 | Seismology |
| PHY-XIII.2 | Mantle Convection |
| PHY-XIII.3 | Geomagnetism and Dynamo |
| PHY-XIII.4 | Atmospheric Physics |
| PHY-XIII.5 | Oceanography |
| PHY-XIII.6 | Glaciology |
| PHY-XIII.7 | Volcanology |
| PHY-XIII.8 | Geodesy |

### PHY-XIV — Materials Science (8 nodes)

| Node | Name |
|------|------|
| PHY-XIV.1 | First-Principles Materials Design |
| PHY-XIV.2 | Mechanical Properties |
| PHY-XIV.3 | Phase-Field Methods |
| PHY-XIV.4 | Microstructure Evolution |
| PHY-XIV.5 | Radiation Damage |
| PHY-XIV.6 | Polymers and Soft Matter |
| PHY-XIV.7 | Ceramics and High-Temperature Materials |
| PHY-XIV.8 | Cell Signaling |

### PHY-XV — Chemical Physics (8 nodes)

| Node | Name |
|------|------|
| PHY-XV.1 | Potential Energy Surfaces |
| PHY-XV.2 | Reaction Rate Theory |
| PHY-XV.3 | Quantum Reaction Dynamics |
| PHY-XV.4 | Nonadiabatic Dynamics |
| PHY-XV.5 | Photochemistry |
| PHY-XV.6 | Catalysis |
| PHY-XV.7 | Spectroscopy |
| PHY-XV.8 | Combustion |

### PHY-XVI — Biophysics (8 nodes)

| Node | Name |
|------|------|
| PHY-XVI.1 | Protein Structure and Dynamics |
| PHY-XVI.2 | Drug Design and Binding |
| PHY-XVI.3 | Membrane Biophysics |
| PHY-XVI.4 | Nucleic Acids |
| PHY-XVI.5 | Systems Biology |
| PHY-XVI.6 | Neuroscience Computational |
| PHY-XVI.7 | Metamaterials |
| PHY-XVI.8 | Phase-Field Modeling |

### PHY-XVII — Computational Methods (6 nodes)

| Node | Name |
|------|------|
| PHY-XVII.1 | Optimization |
| PHY-XVII.2 | Inverse Problems |
| PHY-XVII.3 | ML for Physics |
| PHY-XVII.4 | Mesh Generation and Adaptivity |
| PHY-XVII.5 | Linear Algebra Large-Scale |
| PHY-XVII.6 | High-Performance Computing |

### PHY-XVIII — Multi-Physics Coupling (8 nodes)

| Node | Name |
|------|------|
| PHY-XVIII.1 | Fluid-Structure Interaction |
| PHY-XVIII.2 | Thermo-Mechanical Coupling |
| PHY-XVIII.3 | Electro-Mechanical Coupling |
| PHY-XVIII.4 | Magnetohydrodynamics Coupled |
| PHY-XVIII.5 | Chemically Reacting Flows |
| PHY-XVIII.6 | Radiation-Hydrodynamics |
| PHY-XVIII.7 | Multiscale Methods |
| PHY-XVIII.8 | Data Assimilation |

### PHY-XIX — Quantum Computing (8 nodes)

| Node | Name |
|------|------|
| PHY-XIX.1 | Quantum Circuit Simulation |
| PHY-XIX.2 | Quantum Error Correction |
| PHY-XIX.3 | Quantum Algorithms |
| PHY-XIX.4 | Quantum Simulation |
| PHY-XIX.5 | Quantum Cryptography and Communication |
| PHY-XIX.6 | Quantum Sensing |
| PHY-XIX.7 | Quantum Simulation (extended) |
| PHY-XIX.8 | Quantum Cryptography |

### PHY-XX — Special & Applied (6 nodes)

| Node | Name |
|------|------|
| PHY-XX.1 | Relativistic Mechanics |
| PHY-XX.2 | General Relativity Numerical |
| PHY-XX.3 | Astrodynamics |
| PHY-XX.4 | Robotics Physics |
| PHY-XX.5 | Acoustics Applied |
| PHY-XX.6 | Biomedical Engineering |

---

## Summary Statistics

| Metric | Count |
|--------|------:|
| Tracked files | ~14,561 |
| First-party LOC | ~1.51M |
| Languages | 19 |
| Applications (`apps/`) | 15 |
| Rust crates (`crates/`) | 15 |
| Products (`products/`) | 4 |
| Platform modules (`hypertensor/`) | 7 |
| TensorNet modules (`tensornet/`) | 117 |
| Domain packs | 20 |
| Taxonomy nodes | 168 |
| Genesis layers | 8 |
| Lean 4 formal proofs | 33 |
| Smart contracts | 3 (+1 API spec) |
| Experiments/R&D areas | 20+ |
| Frontier applications | 7 |
| Demo runners | 46 |
| Research scripts | 13 |
| Jupyter notebooks | 6 |
| Civilization challenges | 6 |
| Test files | 104 |
| CI/CD workflows | 11 |
| Game engine integrations | 2 |
| Documentation files (.md) | 326+ |
| ADRs | 25 |

---

*This document was compiled by exhaustive traversal of every directory and file in the repository. The canonical machine-readable registry for the 168 taxonomy nodes is `apps/ledger/nodes/*.yaml`. For the authoritative platform specification, see `PLATFORM_SPECIFICATION.md`.*
