# ADR-0007: Domain-Pack Framework and Anchor Vertical Slices

| Field      | Value                                          |
|------------|------------------------------------------------|
| Status     | Accepted                                       |
| Date       | 2025-01-23                                     |
| Authors    | Platform Team                                  |
| Supersedes | —                                              |
| Phase      | 3 — Domain-Pack Framework & First Verticals    |

## Context

Phase 2 delivered a V&V harness that can validate *any* solver implementing the
platform protocols (ADR-0006).  Phase 3's mission is to stand up the **domain-
pack framework** — the mechanism by which physics domains are packaged, loaded,
and validated — and prove it end-to-end with six anchor vertical slices.

Commercial_Execution.md specifies Phase 3's exit gate:

> *20 domain packs scaffolded to V0.1 (taxonomy node, ProblemSpec, Solver stub).
> 6 anchor packs graduating to V0.4 with full vertical-slice validation
> (convergence, conservation, determinism).*

## Decision

### 1. Pack Architecture

Each domain pack lives in `ontic/packs/pack_{id}.py` and:

- Subclasses the `DomainPack` ABC from `ontic.platform.domain_pack`.
- Overrides `pack_id`, `pack_name`, `taxonomy_ids`, `problem_specs()`, `solvers()`,
  and `version` (as `@property`).
- Registers itself via the `DomainRegistry` singleton on import.

A shared `ontic/packs/_base.py` provides common utilities: `BaseProblemSpec`,
`run_ode_problem()`, `compute_linf_error()`, `convergence_order()`, `make_1d_state()`.

### 2. Taxonomy Coverage

20 domain packs spanning 167 taxonomy nodes:

| ID    | Domain                    | Version | Nodes | Anchor? |
|-------|---------------------------|---------|-------|---------|
| I     | Classical Mechanics       | 0.1.0   |  8    | No      |
| II    | Fluid Dynamics            | 0.4.0   | 10    | Yes     |
| III   | Electromagnetism          | 0.4.0   |  7    | Yes     |
| IV    | Solid Mechanics           | 0.1.0   |  8    | No      |
| V     | Thermo & Stat Mech        | 0.4.0   |  6    | Yes     |
| VI    | Quantum Field Theory      | 0.1.0   |  8    | No      |
| VII   | Quantum Many-Body         | 0.4.0   | 13    | Yes     |
| VIII  | Density Functional Theory | 0.4.0   | 10    | Yes     |
| IX    | Astrophysics              | 0.1.0   |  8    | No      |
| X     | Geophysics                | 0.1.0   |  8    | No      |
| XI    | Plasma Physics            | 0.4.0   | 10    | Yes     |
| XII   | Biophysics                | 0.1.0   |  8    | No      |
| XIII  | Chemical Kinetics         | 0.1.0   |  8    | No      |
| XIV   | Acoustics                 | 0.1.0   |  8    | No      |
| XV    | Nonlinear Dynamics        | 0.1.0   |  8    | No      |
| XVI   | Control Theory            | 0.1.0   |  8    | No      |
| XVII  | Optimization              | 0.1.0   |  8    | No      |
| XVIII | Signal Processing         | 0.1.0   |  8    | No      |
| XIX   | Materials Science         | 0.1.0   |  8    | No      |
| XX    | Climate Modelling         | 0.1.0   |  7    | No      |

### 3. Anchor Validation Criteria (V0.4 Gates)

Each anchor must demonstrate **all four** gates:

1. **Accuracy** — L∞ (or domain-equivalent metric) below a physics-appropriate
   threshold at the finest tested resolution.
2. **Convergence** — Observed spatial/temporal order ≥ expected theoretical order
   (typically ≥ 2.0) via grid-refinement study.
3. **Conservation / Consistency** — Monotone error decay, mass/energy conservation,
   or domain-specific consistency check (e.g., SCF convergence, particle number).
4. **Determinism** — Bit-identical results across two independent runs with the
   same seed.

### 4. Anchor Solver Details

| Pack  | PDE / Model                | Method                                | Key Parameters               |
|-------|----------------------------|---------------------------------------|------------------------------|
| II    | Viscous Burgers            | FVM + conservative central flux + RK4 | N=[128,256,512], ν=0.1       |
| III   | 1-D Maxwell (FDTD)         | Yee leapfrog, PEC BCs                 | N=[400,800,1600], CFL=0.8    |
| V     | Advection–Diffusion        | FVM + central advection/diffusion, RK4| N=[128,256,512], α=0.01      |
| VII   | Heisenberg spin chain      | MPS + 2nd-order Suzuki–Trotter TEBD   | N=8,12; χ=1,4,16,32; τ=0.01 |
| VIII  | 1-D Kohn–Sham SCF         | FD kinetic + soft-Coulomb + direct Hartree | N=[200,400,800], Z=2   |
| XI    | Vlasov–Poisson (Landau)    | Strang splitting + cubic semi-Lagrangian  | (32×64)→(128×256)       |

### 5. Compliance Validation

`domain_pack.py` was extended with `check_compliance(pack)` which verifies:
- `pack_id` and `pack_name` are non-empty strings.
- `taxonomy_ids` is non-empty and each tid has both a ProblemSpec and a Solver.
- Each ProblemSpec satisfies the platform `ProblemSpec` protocol (name, ndim,
  parameters, governing_equations, field_names, observable_names).
- Version is a valid semver string ≥ "0.1.0".

### 6. Testing Strategy

- **Fast suite** (112 tests, <4 s): Discovery, registration, compliance,
  structural checks for all 20 packs, plus smoke tests for each anchor's solver.
- **Slow suite** (6 tests, ~5 min): Full anchor validation running the
  `run_*_vertical_slice()` function of each anchor with convergence verification.
- Marker: `@pytest.mark.slow` separates the two tiers.

## Consequences

- **Platform V1.0** — With the domain-pack framework proven end-to-end, the
  platform exits Phase 3 at version 1.0.0.
- **Scalability** — Any future domain can be added by creating a single pack file
  that subclasses `DomainPack`; the registry, compliance checker, and test
  parametrization handle the rest.
- **V0.4 → V0.5 pathway** — Anchors are ready for ADR-0006's V&V harness
  (MMS injection, standard convergence study, conservation monitors) in Phase 4.
- **Scaffolds → anchors** — Each V0.1 scaffold pack has the full protocol
  skeleton; physics implementation is the only remaining work.

## Status

Accepted.  All 20 packs registered, 6 anchors validated at V0.4, 147 tests
passing.  Platform version bumped to 1.0.0.
