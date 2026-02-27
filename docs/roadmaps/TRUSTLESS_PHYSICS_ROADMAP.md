# Trustless Physics Roadmap — 140-Domain STARK Proof Coverage

**Document ID**: TRUSTLESS-PHYSICS-ROADMAP  
**Version**: 1.0.0  
**Date**: February 8, 2026  
**Authority**: Principal Investigator  
**Cross-References**:
- [PLATFORM_SPECIFICATION.md](../../PLATFORM_SPECIFICATION.md) — Tenet-TPhy §Trustless Physics Certificates
- [TOOLBOX.md](../research/TOOLBOX.md) — 333-module arsenal catalog
- [computational_physics_coverage_assessment.md](../research/computational_physics_coverage_assessment.md) — 140/140 domain audit
- [TRUSTLESS_PHYSICS_EXECUTION_GUIDE.md](../../Tenet-TPhy/TRUSTLESS_PHYSICS_EXECUTION_GUIDE.md) — Original execution plan
- [ROADMAP.md](ROADMAP.md) — Main project roadmap (Layers 0–9)

---

## Executive Summary

This roadmap extends Tenet-TPhy from its current 3-solver proof pipeline (Euler 3D, NS-IMEX, Thermal/Vlasov) to full STARK proof coverage of all **140 computational physics sub-domains** across 20 categories. Each domain receives a cryptographic Trustless Physics Certificate (TPC) with three verification layers:

| Layer | Name | Mechanism | Reusable Infrastructure |
|:-----:|------|-----------|------------------------|
| A | Mathematical Truth | Lean 4 formal proofs of governing equations | `proofs/yang_mills/lean/` — extend per domain |
| B | Computational Integrity | STARK/Halo2 proof of computation trace | `crates/fluidelite_zk/`, `crates/proof_bridge/` — reusable |
| C | Physical Fidelity | Attested benchmark validation + hardware | `tpc/generator.py`, gauntlet framework — reusable |

**Key insight**: The STARK AIR (`ThermalAir`) and TPC infrastructure are **physics-agnostic**. The 8 transition constraints (energy conservation, dt/α constancy, step increment, hash chain continuity) and 20-column trace layout apply to any time-stepping PDE solver. Extending to new domains is primarily a matter of:
1. Writing the solver-specific computation trace adapter
2. Defining the conservation law assertions for Layer A (Lean 4)
3. Configuring the domain-specific benchmark suite for Layer C

---

## Existing Infrastructure Inventory

### Already Built — Do Not Rebuild

The following components exist and are production-tested per `PLATFORM_SPECIFICATION.md` and `TOOLBOX.md`:

| Component | Location | LOC | Tests | Status |
|-----------|----------|----:|------:|:------:|
| TPC binary format | `tpc/format.py` | 1,163 | 25/25 | ✅ Phase 0 |
| TPC certificate generator | `tpc/generator.py` | 511 | — | ✅ Phase 0 |
| TPC constants | `tpc/constants.py` | 73 | — | ✅ Phase 0 |
| Computation trace logger | `tensornet/core/trace.py` | 1,013 | — | ✅ Phase 0 |
| Proof bridge (Rust) | `crates/proof_bridge/` | 1,718 | 12/12 | ✅ Phase 0 |
| Standalone verifier | `apps/trustless_verify/` | 965 | — | ✅ Phase 0 |
| Halo2 circuit framework | `crates/fluidelite_zk/src/circuit/` | ~2K | — | ✅ Phase 1 |
| Hybrid prover (STARK+Halo2) | `crates/fluidelite_zk/src/halo2_hybrid_prover.rs` | — | — | ✅ |
| Groth16 prover | `crates/fluidelite_zk/src/groth16_prover.rs` | — | — | ✅ |
| GPU Halo2 prover | `crates/fluidelite_zk/src/gpu_halo2_prover.rs` | — | — | ✅ |
| Zero-expansion prover v3 | `crates/fluidelite_zk/src/zero_expansion_prover_v3.rs` | — | — | ✅ |
| Trustless REST API | `crates/fluidelite_zk/src/trustless_api.rs` | 860 | — | ✅ Phase 2 |
| Certificate authority | `crates/fluidelite_zk/src/certificate_authority.rs` | — | — | ✅ |
| Multi-timestep prover | `crates/fluidelite_zk/src/multi_timestep.rs` | — | — | ✅ |
| Proof profiler | `crates/fluidelite_zk/src/proof_profiler.rs` | — | — | ✅ |
| Gevulot prover binary | `crates/fluidelite_zk/src/bin/gevulot_prover.rs` | — | — | ✅ Phase 3 |
| Certificate generator bin | `crates/fluidelite_zk/src/bin/generate_certificate.rs` | — | — | ✅ |
| Multi-GPU support | `crates/fluidelite_zk/src/multi_gpu.rs` | — | — | ✅ |
| Rate limiter | `crates/fluidelite_zk/src/rate_limit.rs` | — | — | ✅ Phase 3 |
| Deployment (Containerfile) | `deploy/` | ~1,300 | — | ✅ Phase 2 |

### Existing Lean 4 Proofs

| File | LOC | Domain Covered |
|------|----:|---------------|
| `thermal_conservation_proof/ThermalConservation.lean` | ~120 | Energy conservation, rank bounds, CG termination |
| `vlasov_conservation_proof/VlasovConservation.lean` | ~150 | L² norm, rank bounds, hash chain, Landau damping |
| `navier_stokes_proof/NavierStokes.lean` | ~200 | NS regularity |
| `navier_stokes_proof_v2/NavierStokesRegularity.lean` | 78 | NS regularity (extended) |
| `yang_mills_proof/YangMills.lean` | ~200 | Yang-Mills mass gap |
| `yang_mills_unified_proof/YangMillsUnified.lean` | 113 | Unified YM proof |
| `verified_yang_mills_proof/YangMillsVerified.lean` | 88 | Verified YM |
| `elite_yang_mills_proof_v2/YangMillsMultiEngine.lean` | 94 | Multi-engine YM |

### Existing Gauntlets & Attestations

| Gauntlet | Tests | Attestation |
|----------|------:|-------------|
| `trustless_physics_gauntlet.py` (Phase 0) | 25/25 Python + 12/12 Rust | `TRUSTLESS_PHYSICS_PHASE0_ATTESTATION.json` |
| `trustless_physics_phase1_gauntlet.py` | 24/24 | `TRUSTLESS_PHYSICS_PHASE1_ATTESTATION.json` |
| `trustless_physics_phase2_gauntlet.py` | 45/45 | `TRUSTLESS_PHYSICS_PHASE2_ATTESTATION.json` |
| `trustless_physics_phase3_gauntlet.py` | 40/40 | `TRUSTLESS_PHYSICS_PHASE3_ATTESTATION.json` |
| — | — | `TRUSTLESS_PHYSICS_PHASE4_ATTESTATION.json` |
| — | — | `TRUSTLESS_PHYSICS_FINAL_ATTESTATION.json` |

### Existing Solver Coverage (724+ physics files, ~227K LOC)

All 140 domains have production solvers. See [computational_physics_coverage_assessment.md](../research/computational_physics_coverage_assessment.md) for the complete inventory.

---

## Tiering Strategy

Domains are grouped by proof difficulty — the distance between "solver exists" and "STARK proof + TPC certificate issued":

| Tier | Description | Domains | Effort/Domain | Key Challenge |
|:----:|-------------|:-------:|:-------------:|---------------|
| **1** | Wire-up only — solver + STARK exist | 4 | ~2 days | Trace adapter + Lean shell |
| **2** | Time-stepping PDE — conservation laws map directly to STARK AIR | 65 | ~3–5 days | Per-domain conservation assertions + benchmarks |
| **3** | Iterative/eigenvalue — non-time-stepping (SCF, NRG, optimization) | 45 | ~5–8 days | SCF-to-STARK adapter, convergence proofs |
| **4** | Stochastic/ML/special — PRNG seeds, neural weights, probabilistic | 26 | ~8–15 days | Statistical proof adapters, seed attestation |

---

## Phase Architecture

```
Phase 5 ──── Tier 1 Wire-Up (4 domains)           ← Existing solvers + STARK, just connect
Phase 6 ──── Tier 2A: Core PDE (25 domains)        ← Fluid + Thermal + EM + Plasma  
Phase 7 ──── Tier 2B: Extended PDE (40 domains)     ← Mechanics + Materials + Coupled + Geo
Phase 8 ──── Tier 3: Iterative/Eigenvalue (45)      ← QM, Electronic Structure, Condensed Matter
Phase 9 ──── Tier 4: Stochastic/Special (26)        ← Monte Carlo, ML, Stochastic, Applied
Phase 10 ─── Full-Spectrum Certification            ← 140/140 TPC certificates, audit
```

> **Note**: Phases 0–4 are complete (see `TRUSTLESS_PHYSICS_EXECUTION_GUIDE.md` and attestation JSONs). This roadmap begins at Phase 5.

---

## Phase 5: Tier 1 Wire-Up (4 Domains)

**Duration**: 1 week  
**LOC**: ~800 new (adapters + Lean shells)  
**Prerequisite**: None — all infrastructure exists

These domains already have QTT solvers with STARK-compatible computation traces. The work is glue code: trace adapter → proof bridge → TPC generator → gauntlet test.

| # | Domain | Solver Source | Conservation Law | Existing Lean |
|:-:|--------|-------------|-----------------|:-------------:|
| 1 | Euler 3D (Compressible Flow, II.2) | `tensornet/cfd/euler_3d.py` | Mass + Momentum + Energy | — |
| 2 | NS-IMEX (Incompressible NS, II.1) | `tensornet/cfd/ns_2d.py`, `ns_3d.py` | Momentum + Divergence-free | `NavierStokes.lean` ✅ |
| 3 | Heat Equation (V.5) | `tensornet/thermal/heat_transfer.py` | Energy conservation | `ThermalConservation.lean` ✅ |
| 4 | Vlasov-Poisson 6D (XI.3) | `tensornet/cfd/fast_vlasov_5d.py` | L² norm + Phase-space volume | `VlasovConservation.lean` ✅ |

### Per-Domain Deliverables

For each domain:
1. **Trace adapter** (`tensornet/<domain>/trace_adapter.py`, ~50 LOC) — hooks solver to `ComputationTrace`
2. **Lean 4 conservation shell** (if not existing) — formal statement of governing conservation laws
3. **TPC generation script** (`tools/scripts/tpc/generate_<domain>.py`, ~80 LOC)
4. **Gauntlet test** (extend `trustless_physics_phase5_gauntlet.py`)

### Exit Criteria

- [ ] 4/4 domains produce valid `.tpc` certificates
- [ ] `trustless-verify` passes all 4 certificates
- [ ] `trustless_physics_phase5_gauntlet.py` — all tests pass
- [ ] `TRUSTLESS_PHYSICS_PHASE5_ATTESTATION.json` generated

---

## Phase 6: Tier 2A — Core PDE Domains (25 Domains)

**Duration**: 4 weeks  
**LOC**: ~5,000 new (25 × ~200 LOC avg)  
**Prerequisite**: Phase 5 complete, template validated

These are time-stepping PDE solvers with explicit conservation laws that map directly to the existing STARK AIR constraints. Each follows the template established in Phase 5.

### Template Per Domain

```
1. trace_adapter.py          — Hook solver → ComputationTrace
2. conservation.lean         — Lean 4 formal conservation law proof
3. benchmark_config.json     — Reference solution, error thresholds
4. generate_tpc.py           — TPC generation script
5. gauntlet_test.py          — Automated validation
```

### Domain List

#### II. Fluid Dynamics (8 remaining)

| # | Sub-domain | Solver | Conservation Law for Layer A |
|:-:|-----------|--------|------------------------------|
| 5 | Turbulence (II.3) | `tensornet/cfd/turbulence.py` | TKE budget, enstrophy |
| 6 | Multiphase Flow (II.4) | `tensornet/fluids/multiphase.py` | Phase mass, total energy, interface area |
| 7 | Reactive Flow (II.5) | `tensornet/cfd/reactive_ns.py` | Species mass, total energy, atom counts |
| 8 | Rarefied Gas (II.6) | `tensornet/fluids/rarefied.py` | Number density, kinetic energy |
| 9 | Shallow Water (II.7) | CivStack Hermes | Mass, momentum, potential vorticity |
| 10 | Non-Newtonian (II.8) | `tensornet/fluids/non_newtonian.py` | Momentum, stress–strain energy |
| 11 | Porous Media (II.9) | `tensornet/fluids/porous_media.py` | Fluid mass, Darcy flux |
| 12 | Free Surface (II.10) | `tensornet/fluids/free_surface.py` | Volume, total energy, surface area |

#### III. Electromagnetism (7 domains)

| # | Sub-domain | Solver | Conservation Law for Layer A |
|:-:|-----------|--------|------------------------------|
| 13 | Electrostatics (III.1) | `tensornet/em/electrostatics.py` | Gauss's law, total charge |
| 14 | Magnetostatics (III.2) | `tensornet/em/magnetostatics.py` | ∇·B = 0, magnetic flux |
| 15 | Full Maxwell FDTD (III.3) | `crates/cem-qtt/` | Poynting energy, charge conservation |
| 16 | Frequency-Domain EM (III.4) | `tensornet/em/frequency_domain.py` | Power balance, reciprocity |
| 17 | EM Wave Propagation (III.5) | `tensornet/em/wave_propagation.py` | Poynting energy, CFL stability |
| 18 | Computational Photonics (III.6) | `tensornet/em/computational_photonics.py` | Photon number, energy flux |
| 19 | Antenna & Microwave (III.7) | `tensornet/em/antenna_microwave.py` | Radiated power, input impedance |

#### V. Thermodynamics (3 remaining)

| # | Sub-domain | Solver | Conservation Law for Layer A |
|:-:|-----------|--------|------------------------------|
| 20 | Non-Equilibrium StatMech (V.2) | `tensornet/statmech/non_equilibrium.py` | Free energy, detailed balance |
| 21 | Molecular Dynamics (V.3) | `tensornet/md/molecular_dynamics.py` | Total energy, momentum, angular momentum |
| 22 | Lattice Spin (V.6) | `tensornet/mps/hamiltonians.py` | Total spin, energy |

#### XI. Plasma Physics (7 remaining)

| # | Sub-domain | Solver | Conservation Law for Layer A |
|:-:|-----------|--------|------------------------------|
| 23 | Ideal MHD (XI.1) | CivStack TOMAHAWK | Mass + Momentum + Energy + ∇·B = 0 |
| 24 | Resistive MHD (XI.2) | `tensornet/plasma/extended_mhd.py` | Magnetic helicity, total energy |
| 25 | Gyrokinetics (XI.4) | `tensornet/plasma/gyrokinetics.py` | Phase-space volume, energy |
| 26 | Magnetic Reconnection (XI.5) | `tensornet/plasma/reconnection.py` | Total energy, magnetic flux |
| 27 | Laser-Plasma (XI.6) | `tensornet/plasma/laser_plasma.py` | Photon number, energy |
| 28 | Dusty Plasma (XI.7) | `tensornet/plasma/dusty_plasma.py` | Particle count, total energy |
| 29 | Space Plasma (XI.8) | `tensornet/plasma/space_plasma.py` | Cosmic ray number, magnetic flux |

### Exit Criteria

- [ ] 25/25 domains produce valid `.tpc` certificates
- [ ] Conservation law coverage ≥ 90% per domain
- [ ] `trustless_physics_phase6_gauntlet.py` — all tests pass
- [ ] `TRUSTLESS_PHYSICS_PHASE6_ATTESTATION.json` generated

---

## Phase 7: Tier 2B — Extended PDE Domains (40 Domains)

**Duration**: 6 weeks  
**LOC**: ~10,000 new (40 × ~250 LOC avg — slightly more complex adapters)  
**Prerequisite**: Phase 6 template proven at scale

### I. Classical Mechanics (6 domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 30 | Newtonian Dynamics (I.1) | `tensornet/guidance/trajectory.py` | Linear + angular momentum, energy |
| 31 | Lagrangian/Hamiltonian (I.2) | `tensornet/mechanics/symplectic.py` | Hamiltonian, symplectic form |
| 32 | Continuum (I.3) | `tensornet/mechanics/continuum.py` | Linear momentum, energy, angular momentum |
| 33 | Structural (I.4) | `tensornet/mechanics/structural.py` | Virtual work, energy |
| 34 | Nonlinear Dynamics (I.5) | CivStack Dynamics | Lyapunov function, attractor bounds |
| 35 | Acoustics (I.6) | `tensornet/mechanics/acoustics.py` | Acoustic energy, reciprocity |

### IV. Optics & Photonics (4 domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 36 | Physical Optics (IV.1) | `tensornet/optics/physical_optics.py` | Poynting flux, coherence |
| 37 | Quantum Optics (IV.2) | `tensornet/optics/quantum_optics.py` | Photon number, trace(ρ) = 1 |
| 38 | Laser Physics (IV.3) | `tensornet/optics/laser_physics.py` | Population inversion, energy balance |
| 39 | Ultrafast Optics (IV.4) | `tensornet/optics/ultrafast_optics.py` | Pulse energy, photon number |

### XII. Astrophysics & Cosmology (6 domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 40 | Stellar Structure (XII.1) | `tensornet/astro/stellar_structure.py` | Hydrostatic equilibrium, luminosity |
| 41 | Compact Objects (XII.2) | `tensornet/astro/compact_objects.py` | TOV mass-energy, geodesic constants |
| 42 | Gravitational Waves (XII.3) | `tensornet/astro/gravitational_waves.py` | Energy flux, angular momentum |
| 43 | Cosmological Sims (XII.4) | `tensornet/astro/cosmological_sims.py` | Total mass, energy, Friedmann constraint |
| 44 | CMB (XII.5) | `tensornet/astro/cmb_early_universe.py` | Photon number, entropy |
| 45 | Radiative Transfer (XII.6) | `tensornet/astro/radiative_transfer.py` | Radiative energy, photon number |

### XIII. Geophysics (6 domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 46 | Seismology (XIII.1) | `tensornet/geophysics/seismology.py` | Elastic energy, wave energy |
| 47 | Mantle Convection (XIII.2) | `tensornet/geophysics/mantle_convection.py` | Mass, thermal energy, Nusselt |
| 48 | Geodynamo (XIII.3) | `tensornet/geophysics/geodynamo.py` | Magnetic energy, ∇·B = 0 |
| 49 | Atmospheric (XIII.4) | `tensornet/atmosphere/atmospheric_physics.py` | Mass, enthalpy, ozone column |
| 50 | Oceanography (XIII.5) | `tensornet/ocean/oceanography.py` | Mass, salt, heat |
| 51 | Glaciology (XIII.6) | `tensornet/geophysics/glaciology.py` | Ice volume, thermal energy |

### XIV. Materials Science (7 domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 52 | First-Principles Design (XIV.1) | `tensornet/materials/first_principles_design.py` | Total energy, pressure–volume |
| 53 | Mechanical Properties (XIV.2) | `tensornet/materials/mechanical.py` | Elastic energy, symmetry |
| 54 | Phase-Field (XIV.3) | `tensornet/materials/phase_field.py` | Free energy decrease, mass |
| 55 | Microstructure (XIV.4) | `tensornet/materials/microstructure.py` | Phase fractions, energy |
| 56 | Radiation Damage (XIV.5) | `tensornet/materials/radiation_damage.py` | Displaced atoms, energy deposition |
| 57 | Polymers (XIV.6) | `tensornet/materials/polymers_soft_matter.py` | Free energy, chain number |
| 58 | Ceramics (XIV.7) | `tensornet/materials/ceramics.py` | Mass, thermal energy |

### XVIII. Coupled Physics (7 domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 59 | FSI (XVIII.1) | `tensornet/coupled/fsi.py` | Total (fluid + solid) energy, momentum |
| 60 | Thermo-Mechanical (XVIII.2) | `tensornet/coupled/thermo_mechanical.py` | Thermal + elastic energy |
| 61 | Electro-Mechanical (XVIII.3) | `tensornet/coupled/electro_mechanical.py` | Electrostatic + elastic energy |
| 62 | Coupled MHD (XVIII.4) | `tensornet/coupled/coupled_mhd.py` | Kinetic + magnetic energy, ∇·B = 0 |
| 63 | Reacting Flows (XVIII.5) | `tensornet/cfd/reactive_ns.py` | Species + total energy |
| 64 | Radiation-Hydro (XVIII.6) | `tensornet/radiation/radiation_hydro.py` | Total (radiation + matter) energy |
| 65 | Multiscale (XVIII.7) | `tensornet/coupled/multiscale.py` | Fine↔coarse energy consistency |

### XV. Chemical Physics (4 time-stepping domains)

| # | Sub-domain | Solver | Conservation Law |
|:-:|-----------|--------|-----------------|
| 66 | Nonadiabatic Dynamics (XV.4) | `tensornet/chemistry/nonadiabatic.py` | Total energy (electronic + nuclear) |
| 67 | Photochemistry (XV.5) | `tensornet/chemistry/photochemistry.py` | Oscillator strength sum, energy |
| 68 | Quantum Reactive (XV.3) | `tensornet/chemistry/quantum_reactive.py` | Total probability, energy |
| 69 | Spectroscopy (XV.7) | `tensornet/chemistry/spectroscopy.py` | Sum rules, energy levels |

### Exit Criteria

- [ ] 40/40 domains produce valid `.tpc` certificates
- [ ] `trustless_physics_phase7_gauntlet.py` — all tests pass
- [ ] `TRUSTLESS_PHYSICS_PHASE7_ATTESTATION.json` generated
- [ ] Cumulative: 69/140 domains certified

---

## Phase 8: Tier 3 — Iterative / Eigenvalue Domains (45 Domains)

**Duration**: 8 weeks  
**LOC**: ~15,000 new  
**Prerequisite**: Phase 7 complete

These domains use iterative convergence (SCF, DMRG, NRG, optimization) rather than explicit time-stepping. The STARK AIR must be adapted:

### SCF-to-STARK Adapter

**New component**: Generic adapter that maps iterative self-consistent-field loops to STARK traces.

```
SCF iteration i:
  input_hash[i]  = H(density_matrix[i])
  output_hash[i] = H(density_matrix[i+1])
  residual[i]    = ||ρ[i+1] - ρ[i]||
  constraint:      residual[i] < residual[i-1]  (monotone convergence)
  constraint:      output_energy ≤ input_energy   (variational principle)
```

**Deliverable**: `tensornet/core/scf_trace_adapter.py` (~500 LOC)  
**Reuse**: All Tier 3 domains use this adapter

### Eigenvalue-to-STARK Adapter

For Lanczos/Davidson/DMRG eigensolvers:

```
Krylov step k:
  input_hash[k]  = H(Krylov_basis[k])
  output_hash[k] = H(Krylov_basis[k+1])
  constraint:      basis orthogonality ||<v_k|v_j>|| < ε for j < k
  constraint:      Ritz value convergence (monotone decrease)
```

**Deliverable**: `tensornet/core/eigenvalue_trace_adapter.py` (~400 LOC)

### Domain List

#### VI. Quantum Mechanics (5 domains)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 70 | TISE (VI.1) | Eigenvalue | Energy eigenvalue convergence, norm = 1 |
| 71 | TDSE (VI.2) | Time-step | Probability conservation, unitarity |
| 72 | Scattering (VI.3) | Eigenvalue | Optical theorem, unitarity of S-matrix |
| 73 | Semiclassical (VI.4) | Time-step | Action stationarity, Maslov index |
| 74 | Path Integrals (VI.5) | Stochastic (→ Phase 9) or Time-step | Partition function, detailed balance |

#### VII. Quantum Many-Body (13 domains)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 75 | Tensor Network / DMRG (VII.1) | Eigenvalue | Energy variational bound, entanglement entropy |
| 76 | Quantum Spin (VII.2) | Eigenvalue | Total S_z, energy |
| 77 | Strongly Correlated (VII.3) | SCF (DMFT) | Self-energy convergence, spectral sum rule |
| 78 | Topological (VII.4) | Eigenvalue | Chern number (integer), TEE |
| 79 | MBL & Disorder (VII.5) | Eigenvalue | Level statistics, participation ratio |
| 80 | Lattice Gauge (VII.6) | Stochastic (HMC) | Gauss's law, plaquette average |
| 81 | Open Quantum (VII.7) | Time-step | Trace(ρ) = 1, positivity |
| 82 | Non-Equilibrium QM (VII.8) | Time-step | Energy, Lieb-Robinson bound |
| 83 | Kondo/Impurity (VII.9) | Eigenvalue (NRG) | Friedel sum rule, spectral weight |
| 84 | Bosonic (VII.10) | SCF/Time-step | Particle number, energy |
| 85 | Fermionic (VII.11) | SCF | Particle number, energy, Luttinger |
| 86 | Nuclear Many-Body (VII.12) | Eigenvalue (CI) | Nucleon number, angular momentum |
| 87 | Ultracold Atoms (VII.13) | Time-step (GPE) | Atom number, energy |

#### VIII. Electronic Structure (7 domains)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 88 | DFT (VIII.1) | SCF | Total energy variational, electron count |
| 89 | Beyond-DFT (VIII.2) | SCF | Correlation energy, size-consistency |
| 90 | Tight Binding (VIII.3) | Eigenvalue | Band filling, charge neutrality |
| 91 | Excited States (VIII.4) | SCF/Eigenvalue | Oscillator strength sum rule, f-sum |
| 92 | Response Properties (VIII.5) | SCF (DFPT) | Kramers-Kronig, sum rules |
| 93 | Relativistic Electronic (VIII.6) | SCF | Charge conservation, current continuity |
| 94 | Quantum Embedding (VIII.7) | SCF (DFT+DMFT) | Total electron count, energy partition |

#### IX. Solid State / Condensed Matter (8 domains)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 95 | Phonons (IX.1) | Eigenvalue | Dynamical matrix Hermiticity, acoustic sum rule |
| 96 | Band Structure (IX.2) | Eigenvalue | Bloch periodicity, charge neutrality |
| 97 | Classical Magnetism (IX.3) | Time-step (LLG) | |M| conservation, energy decrease |
| 98 | Superconductivity (IX.4) | SCF | Gap equation convergence, particle number |
| 99 | Disordered Systems (IX.5) | Eigenvalue | Normalisation, spectral weight |
| 100 | Surfaces & Interfaces (IX.6) | SCF | Charge neutrality, slab convergence |
| 101 | Defects (IX.7) | SCF | Formation energy convergence |
| 102 | Ferroelectrics (IX.8) | Time-step/SCF | Polarisation, free energy |

#### X. Nuclear & Particle (6 domains)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 103 | Nuclear Structure (X.1) | Eigenvalue (CI) | Nucleon number, parity, J |
| 104 | Nuclear Reactions (X.2) | Eigenvalue (R-matrix) | Unitarity, threshold behavior |
| 105 | Nuclear Astrophysics (X.3) | Time-step (network) | Baryon number, energy |
| 106 | Lattice QCD (X.4) | Stochastic (HMC) | Gauge invariance, plaquette |
| 107 | Perturbative QFT (X.5) | Algebraic | Ward identity, renormalisation group |
| 108 | Beyond SM (X.6) | Time-step/algebraic | Quantum numbers, unitarity |

#### XV. Chemical Physics (3 remaining)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 109 | PES (XV.1) | SCF (Born-Opp.) | Total energy, gradient consistency |
| 110 | Reaction Rate (XV.2) | Eigenvalue (TST) | Detailed balance, partition function |
| 111 | Catalysis (XV.6) | SCF/Time-step | Atom count, energy |

#### XIX. Quantum Information (2 remaining — 3 already Tier 2)

| # | Sub-domain | Adapter | Conservation/Convergence Law |
|:-:|-----------|---------|------------------------------|
| 112 | Quantum Circuit (XIX.1) | Time-step (gate) | Unitarity, trace |
| 113 | QEC (XIX.2) | Eigenvalue (syndrome) | Logical qubit fidelity |
| 114 | Quantum Algorithms (XIX.3) | Time-step (VQE) | Energy variational bound |

### Exit Criteria

- [ ] SCF-to-STARK adapter validated on DFT + DMFT
- [ ] Eigenvalue-to-STARK adapter validated on DMRG + Lanczos
- [ ] 45/45 domains produce valid `.tpc` certificates
- [ ] `trustless_physics_phase8_gauntlet.py` — all tests pass
- [ ] `TRUSTLESS_PHYSICS_PHASE8_ATTESTATION.json` generated
- [ ] Cumulative: 114/140 domains certified

---

## Phase 9: Tier 4 — Stochastic / ML / Special Domains (26 Domains)

**Duration**: 8 weeks  
**LOC**: ~12,000 new  
**Prerequisite**: Phase 8 complete

These domains involve stochastic processes (PRNG-seeded Monte Carlo, Gillespie SSA), machine learning models, or fundamentally different computation patterns that require specialized STARK adapters.

### Stochastic-to-STARK Adapter

For Monte Carlo / Gillespie / PIMC methods, the STARK proves:

```
MC sweep s:
  seed_hash[s]     = H(PRNG_state[s])
  accept_count[s]  = number of accepted moves
  observable[s]    = measured quantity (e.g. energy)
  constraint:       seed_hash chain is deterministic (given initial seed)
  constraint:       acceptance satisfies detailed balance
  constraint:       running average converges within stated error bars
```

**Deliverable**: `tensornet/core/stochastic_trace_adapter.py` (~600 LOC)

### ML-to-STARK Adapter

For PINN / FNO / Neural network potential methods:

```
Training step t:
  weight_hash[t]   = H(θ[t])
  loss[t]          = L(θ[t])
  gradient_hash[t] = H(∇L[t])
  constraint:       loss[t] ≤ loss[t-1] (with momentum/learning rate tolerance)
  constraint:       final_loss < threshold
  inference:        input_hash → output_hash via forward(θ_final)
```

**Deliverable**: `tensornet/core/ml_trace_adapter.py` (~400 LOC)

### Domain List

#### V. StatMech — Stochastic (2 domains)

| # | Sub-domain | Adapter | Verifiable Property |
|:-:|-----------|---------|---------------------|
| 115 | Equilibrium MC (V.1) | Stochastic | Detailed balance, convergence |
| 116 | Monte Carlo General (V.4) | Stochastic | Ergodicity, estimator variance |

#### XVI. Biophysics (6 domains)

| # | Sub-domain | Adapter | Verifiable Property |
|:-:|-----------|---------|---------------------|
| 117 | Protein Structure (XVI.1) | SCF/Stoch. | RMSD convergence, Ramachandran |
| 118 | Drug Design (XVI.2) | Stochastic (FEP) | Free energy convergence, overlap |
| 119 | Membrane (XVI.3) | Stochastic (CG-MD) | Area/lipid, bilayer thickness |
| 120 | Nucleic Acids (XVI.4) | SCF/Stoch. | Base pairing energy, MFE |
| 121 | Systems Biology (XVI.5) | Stochastic (Gillespie) | Mass action, stoichiometry |
| 122 | Neuroscience (XVI.6) | Time-step (ODE) | Membrane potential bounds, current |

#### XVII. Computational Methods (6 domains)

| # | Sub-domain | Adapter | Verifiable Property |
|:-:|-----------|---------|---------------------|
| 123 | Optimization (XVII.1) | Eigenvalue/SCF | Objective decrease, constraint satisfaction |
| 124 | Inverse Problems (XVII.2) | SCF | Residual decrease, regularization |
| 125 | ML for Physics (XVII.3) | ML | Loss convergence, physics residual |
| 126 | Mesh Generation (XVII.4) | Algebraic | Element quality, conformity |
| 127 | Large-Scale LinAlg (XVII.5) | Eigenvalue | Residual decrease, orthogonality |
| 128 | HPC (XVII.6) | Meta-adapter | Reproducibility, bit-exact |

#### XIX. Quantum Information (2 remaining)

| # | Sub-domain | Adapter | Verifiable Property |
|:-:|-----------|---------|---------------------|
| 129 | Quantum Simulation (XIX.4) | Time-step/DMRG | Energy, entanglement |
| 130 | Quantum Crypto (XIX.5) | Algebraic | Key rate, entropy bound |

#### XX. Special / Applied (10 domains)

| # | Sub-domain | Adapter | Verifiable Property |
|:-:|-----------|---------|---------------------|
| 131 | Special Relativity (XX.1) | Time-step | 4-momentum conservation |
| 132 | Numerical GR (XX.2) | Time-step | Hamiltonian + momentum constraints |
| 133 | Astrodynamics (XX.3) | Time-step | Orbital energy, Jacobi constant |
| 134 | Robotics Physics (XX.4) | Time-step | Joint torque bounds, energy |
| 135 | Applied Acoustics (XX.5) | Time-step | Acoustic energy, reciprocity |
| 136 | Biomedical (XX.6) | Time-step/SCF | Action potential, drug mass balance |
| 137 | Environmental (XX.7) | Time-step | Pollutant mass balance, energy |
| 138 | Energy Systems (XX.8) | Time-step/SCF | Charge balance, energy |
| 139 | Manufacturing (XX.9) | Time-step | Enthalpy balance, mass |
| 140 | Semiconductor (XX.10) | SCF (drift-diff.) | Current continuity, carrier count |

### Exit Criteria

- [ ] Stochastic-to-STARK adapter validated on Metropolis + Gillespie
- [ ] ML-to-STARK adapter validated on PINN + FNO
- [ ] 26/26 domains produce valid `.tpc` certificates  
- [ ] `trustless_physics_phase9_gauntlet.py` — all tests pass
- [ ] `TRUSTLESS_PHYSICS_PHASE9_ATTESTATION.json` generated
- [ ] Cumulative: 140/140 domains certified

---

## Phase 10: Full-Spectrum Certification & Audit

**Duration**: 4 weeks  
**LOC**: ~3,000 new (audit tooling, certificate index, documentation)  
**Prerequisite**: Phase 9 complete

### 10.1 Certificate Index & Registry

Build a machine-readable registry of all 140 TPC certificates:

```json
{
  "version": "1.0.0",
  "total_domains": 140,
  "certified": 140,
  "certificates": [
    {
      "domain": "II.1",
      "name": "Incompressible Navier-Stokes",
      "solver": "tensornet/cfd/ns_3d.py",
      "tpc_path": "certificates/II_1_ns.tpc",
      "lean_theorems": ["ns_mass_conservation", "ns_momentum_conservation", "ns_divergence_free"],
      "proof_system": "STARK",
      "verification_time_s": 0.8,
      "benchmark": "lid_driven_cavity_Re1000",
      "phase": 5
    }
  ]
}
```

**Deliverable**: `tpc/registry.py` + `certificates/index.json`

### 10.2 Cross-Domain Regression Suite

A single command that re-generates and re-verifies all 140 certificates:

```bash
python tools/scripts/gauntlets/trustless_physics_full_spectrum_gauntlet.py
```

Produces:
- 140 `.tpc` certificates
- 140 verification reports
- Aggregate attestation JSON with pass/fail per domain
- Performance summary (total proof time, total verify time, aggregate proof size)

### 10.3 Lean Proof Library Consolidation

Consolidate all per-domain Lean 4 proofs into a structured Lean project:

```
lean_trustless_physics/
├── lakefile.lean
├── TrustlessPhysics/
│   ├── FluidDynamics/
│   │   ├── EulerConservation.lean
│   │   ├── NSConservation.lean
│   │   ├── MultiphaseConservation.lean
│   │   └── ...
│   ├── Electromagnetism/
│   │   ├── MaxwellConservation.lean
│   │   └── ...
│   ├── QuantumMechanics/
│   │   └── ...
│   └── ...
└── README.md
```

### 10.4 Regulatory Compliance Mapping

For each of the 140 domains, document:
- Which Layer A theorems map to which regulatory V&V requirements
- Which Layer B proof system provides which computational integrity guarantee
- Which Layer C benchmarks satisfy which industry standards

### Exit Criteria

- [ ] `certificates/index.json` with 140/140 entries
- [ ] Full-spectrum gauntlet: 140/140 pass
- [ ] Lean project builds cleanly (`lake build`)
- [ ] Regulatory mapping complete for Tier 1 frameworks (MIL-STD-3022, FAA AC 25.571)
- [ ] `TRUSTLESS_PHYSICS_FULL_SPECTRUM_ATTESTATION.json` generated
- [ ] Tag: `vX.0.0-trustless-physics-140`

---

## Summary — Effort Estimates

| Phase | Domains | Duration | New LOC | Cumulative Domains |
|:-----:|:-------:|:--------:|--------:|:------------------:|
| 0–4 | 3 | Complete | ~26,300 | 3 |
| **5** | 4 | 1 week | ~800 | 7 |
| **6** | 25 | 4 weeks | ~5,000 | 32 |
| **7** | 40 | 6 weeks | ~10,000 | 72 |
| **8** | 45 | 8 weeks | ~15,000 | 117 |
| **9** | 26 | 8 weeks | ~12,000 | 143* |
| **10** | — | 4 weeks | ~3,000 | 140 (audited) |
| **Total** | **140** | **~31 weeks** | **~72,100** | **140/140** |

*\*3 domains overlap between Tier 2 and Tier 4 (Path Integrals, Lattice Gauge, Lattice QCD use both time-stepping and stochastic modes)*

### Compression Opportunities

| Optimization | Saving | Mechanism |
|--------------|:------:|-----------|
| Template automation (cookiecutter) | 30% | Generate trace adapter + Lean shell + gauntlet from YAML spec |
| Parallel domain work (4-person team) | 4× | Independent domains can proceed concurrently |
| Adapter reuse across domains | 20% | SCF adapter covers ~30 domains; time-step adapter covers ~60 |
| AI-assisted Lean proof generation | 25% | Pattern-match against existing conservation proofs |

**Optimistic estimate (4-person team + tooling)**: ~4 months  
**Conservative estimate (2-person team)**: ~8 months

---

## Reusable Infrastructure — Do Not Duplicate

This section serves as the canonical "toolbox check" — any new work must verify it doesn't rebuild these:

| What | Where | Reuse Across |
|------|-------|-------------|
| STARK AIR (ThermalAir) | `fluidelite-circuits` (referenced via PLATFORM_SPEC) | All time-stepping domains |
| TPC format + generator | `tpc/format.py`, `tpc/generator.py` | All 140 domains |
| Computation trace logger | `tensornet/core/trace.py` | All 140 domains |
| Proof bridge | `crates/proof_bridge/` | All 140 domains |
| Standalone verifier | `apps/trustless_verify/` | All 140 domains |
| Halo2 circuit base | `crates/fluidelite_zk/src/circuit/` | All circuit-based proofs |
| Hybrid prover | `crates/fluidelite_zk/src/halo2_hybrid_prover.rs` | All hybrid STARK+Halo2 proofs |
| Groth16 prover | `crates/fluidelite_zk/src/groth16_prover.rs` | All Groth16 proofs |
| GPU acceleration | `crates/fluidelite_zk/src/gpu*.rs` | All GPU-accelerated proofs |
| Trustless REST API | `crates/fluidelite_zk/src/trustless_api.rs` | All API-driven certificate generation |
| Genesis integration | `crates/fluidelite_zk/src/genesis_prover.rs` | All genesis primitive proofs |
| Certificate authority | `crates/fluidelite_zk/src/certificate_authority.rs` | All PKI operations |
| Multi-timestep prover | `crates/fluidelite_zk/src/multi_timestep.rs` | All time-stepping domains |
| Proof profiler | `crates/fluidelite_zk/src/proof_profiler.rs` | All profiling/benchmarking |
| Gevulot integration | `crates/fluidelite_zk/src/bin/gevulot_prover.rs` | All decentralized verification |
| Deployment infra | `deploy/` | All on-premise deployments |
| Gauntlet framework | `tools/scripts/gauntlets/` | All validation suites |
| Attestation format | `docs/attestations/*.json` | All attestation outputs |
| Lean thermal proof | `thermal_conservation_proof/ThermalConservation.lean` | Heat transfer domains |
| Lean Vlasov proof | `vlasov_conservation_proof/VlasovConservation.lean` | Kinetic/plasma domains |
| Lean NS proof | `navier_stokes_proof/NavierStokes.lean` | All NS-derived domains |

---

## Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|:-:|------|:------:|:----------:|-----------|
| R1 | SCF-to-STARK adapter fails for non-monotone convergence (e.g., DMFT with restarts) | High | Medium | Fall back to proving input→output consistency only (weaker but valid) |
| R2 | Stochastic proofs have large proof size (many MC sweeps) | Medium | High | Prove statistical summary only: seed chain + final observable ± error bar |
| R3 | Lean proof coverage lags behind STARK proof generation | Low | High | Issue certificates with Layer A = "partial" (Layer B covers computation integrity) |
| R4 | Some solvers are non-deterministic (floating-point order) | High | Medium | Fix solver determinism first, or prove within floating-point tolerance envelope |
| R5 | 140-domain gauntlet takes too long to run | Medium | Medium | Parallelize, use smaller grids for CI, full-size for release gauntlets |
| R6 | Regulatory bodies unfamiliar with ZK proofs | High | High | "Belt and suspenders" — traditional V&V + TPC certificate together |

---

## Dependencies

```
Phase 5 ← Phases 0-4 (complete)
Phase 6 ← Phase 5 (template validated)
Phase 7 ← Phase 6 (template proven at scale)
Phase 8 ← Phase 7 + SCF/Eigenvalue adapters
Phase 9 ← Phase 8 + Stochastic/ML adapters
Phase 10 ← Phase 9 (all 140 certified)
```

All phases after Phase 5 can run domains in parallel within the phase. Cross-phase dependencies are sequential.

---

*This roadmap is a living document. Update as phases complete and new infrastructure is identified.*

*Cross-reference before every phase:*
- *[PLATFORM_SPECIFICATION.md](../../PLATFORM_SPECIFICATION.md) — check for new infrastructure*
- *[TOOLBOX.md](../research/TOOLBOX.md) — check for new modules*
- *Reusable Infrastructure table above — verify no duplication*

*© 2026 Tigantic Holdings LLC. All rights reserved.*
