# ADR-0005: Vertical Slices as Phase Exit Gates

**Date:** 2026-02-08  
**Status:** Accepted  
**Deciders:** Platform Architecture  

## Context

Phase 1 of the Commercial Execution Plan requires:

> *At least one PDE and one ODE solver traverse the full stack
> (spec → run → output → regression) at Platform V0.4.*

Defining "traverse the full stack" precisely and automatically is essential to
prevent subjective sign-off.

## Decision

Each phase exit gate is demonstrated by one or more **vertical slices** —
small, self-contained programs that exercise every layer of the platform in
sequence:

1. `ProblemSpec` declaration
2. `StructuredMesh` construction
3. `InitialCondition` generation
4. `Discretization` (for PDEs) or Hamiltonian RHS (for ODEs)
5. `TimeIntegrator.solve()` with observables
6. `ReproducibilityContext` (seed lock + environment capture + artifact hashing)
7. `save_checkpoint` / `load_checkpoint` round-trip
8. Quantitative gates (error vs exact, conservation, grid refinement, determinism)

The Phase 1 exit gate is satisfied by:

| Slice | Problem | Integrator | Gates |
|---|---|---|---|
| `vertical_ode` | Harmonic oscillator | Störmer-Verlet | Energy < 1e-8 rel, phase < 0.5°, deterministic, checkpoint |
| `vertical_pde` | 1-D heat equation | RK4 + FVM | L∞ < 1e-4, refinement ratio ≈ 4, monotone decay, deterministic, checkpoint |

## Consequences

- Future phases extend this pattern: Phase 2 adds MMS convergence slices,
  Phase 3 adds multi-domain coupling slices, Phase 5 adds QTT-accelerated
  slices.
- CI runs vertical slices on every PR — failure blocks merge.
- Slices double as onboarding examples and regression baselines.
