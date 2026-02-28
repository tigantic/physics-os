# ADR-0006: Verification & Validation Harness Architecture

| Field      | Value                  |
|------------|------------------------|
| Status     | Accepted               |
| Date       | 2025-01-22             |
| Authors    | Platform Team          |
| Supersedes | —                      |
| Phase      | 2 — V&V Harness        |

## Context

Phase 1 delivered the core platform substrate (protocols, data model, solvers,
reproducibility, checkpointing) with two vertical slices that manually coded their
own convergence checks, error measurement, and conservation monitoring.

For the platform to scale to 140 domain-pack nodes, **every solver must use the
same V&V toolbox** — bespoke per-solver validation is unsustainable and error-prone.

Phase 2's exit gate states:

> *Any solver at V0.4 must be able to reach V0.5 using standard harness outputs
> — no bespoke validation.*

## Decision

We introduce a `ontic.platform.vv` subpackage with six complementary modules:

| Module         | Responsibility                                                    |
|----------------|-------------------------------------------------------------------|
| `mms`          | Method of Manufactured Solutions: inject source terms, verify code |
| `convergence`  | Grid refinement & timestep refinement studies with order detection |
| `conservation` | Monitor conserved quantities (mass, energy, divergence-free, Lp)  |
| `stability`    | CFL checker, blow-up detector, stiffness estimator                |
| `performance`  | Per-step timing, memory tracking, strong/weak scaling studies      |
| `benchmarks`   | Registry of canonical problems with golden outputs & thresholds   |

### Design Principles

1. **Observable-compatible**: Conservation monitors implement the `Observable`
   protocol and plug directly into `TimeIntegrator.solve(observables=[...])`.

2. **Solver-agnostic**: All modules work with any solver that conforms to the
   Phase 1 protocols (SimulationState, RHSCallable, TimeIntegrator).

3. **Factory pattern for benchmarks**: Benchmark problems provide a
   `setup_fn(mesh, golden) → (integrator, state, rhs, t_span, dt)` factory,
   decoupling benchmark definition from solver implementation.

4. **Dataclass results**: Every module returns frozen dataclass results
   (`MMSConvergenceResult`, `RefinementResult`, `ConservationReport`,
   `StabilityVerdict`, `PerformanceReport`, `BenchmarkResult`) with
   `.summary()` methods for human-readable output.

5. **No bespoke code per solver**: A solver author provides two things:
   (a) the RHS function, (b) a solver factory.  Everything else — convergence
   measurement, conservation tracking, stability monitoring — comes from the
   standard harness.

### Module Architecture

```
ontic/platform/vv/
├── __init__.py        # Package exports
├── mms.py             # ManufacturedSolution, MMSProblem, mms_convergence_study
├── convergence.py     # RefinementStudy, grid/timestep_refinement_study
├── conservation.py    # ConservedQuantity ABC, MassIntegral, EnergyIntegral, ...
├── stability.py       # StabilityCheck ABC, CFLChecker, BlowupDetector, ...
├── performance.py     # PerformanceHarness, ScalingStudy
└── benchmarks.py      # BenchmarkRegistry, BenchmarkProblem, GoldenOutput
```

### V-State Gate Mapping

| V-State | Gate                                   | Harness Module(s)           |
|---------|----------------------------------------|-----------------------------|
| V0.4    | Vertical slice passes                  | (Phase 1 — manual)          |
| V0.5    | MMS convergence + grid refinement PASS | mms, convergence            |
| V0.5    | Conservation monotonicity verified     | conservation                |
| V0.5    | Stability checks pass (CFL, blow-up)   | stability                   |
| V0.6    | Benchmark golden-output comparison     | benchmarks                  |
| V0.6    | Performance profiled, scaling measured  | performance                 |

## Consequences

**Positive:**
- Uniform V&V across all 140 nodes — no bespoke validation code.
- Automated promotion pipeline: solver → V0.4 → harness → V0.5 → benchmark → V0.6.
- Rich diagnostic output (summaries, reports) for debugging failed promotions.
- Factory pattern enables registering new benchmarks without modifying harness code.

**Negative:**
- Additional learning curve for solver authors (mitigated by vertical_vv demo).
- MMS requires the solver author to derive a source term (can be aided by symbolic
  differentiation in future phases).
- Performance harness uses RSS-based memory tracking, which is approximate for
  GPU workloads (will switch to torch.cuda instrumentation when GPU paths mature).

**Risks:**
- Conservation thresholds may need per-domain tuning (e.g., dissipative vs
  conservative systems). Mitigated by per-quantity threshold overrides.
- Timestep refinement is non-trivial for CFL-constrained diffusion PDEs where
  temporal and spatial errors scale identically. Mitigated by allowing ODE-based
  temporal convergence tests.
