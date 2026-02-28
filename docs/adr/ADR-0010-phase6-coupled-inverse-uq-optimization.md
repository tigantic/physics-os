# ADR-0010: Phase 6 — Coupled Physics, Inverse Problems, UQ & Optimization

**Status:** Accepted  
**Date:** 2026-02-09  
**Decision Makers:** Platform team  

## Context

Phases 0–5 built the platform substrate, full taxonomy coverage, and QTT
acceleration.  All work so far focused on single-physics forward solves.
Real engineering workflows require:

1. **Multi-physics coupling** — e.g., thermal-structural, fluid-structure.
2. **Inverse problems** — parameter identification from observations.
3. **Uncertainty quantification** — propagating input uncertainty to outputs.
4. **Optimization** — topology/shape optimization with physics constraints.
5. **Provenance tracking** — audit trail for multi-stage workflows.

These capabilities are prerequisites for Phase 7 Productization.

## Decision

### 1. Coupling Orchestrator (`ontic/platform/coupled.py`)

**Architecture:** Strategy-pattern coupler with pluggable strategies.

- `CoupledField` — Declares which field transfers between which solvers,
  with optional transfer function and relaxation factor.
- `CouplingInterface` — Groups coupled fields with convergence criteria.
- `CouplerBase` (ABC) — Provides the time-stepping loop (`solve()`).
- `MonolithicCoupler` — Single-pass field transfer per time step.
- `PartitionedCoupler` — Sub-iteration convergence with configurable
  strategy (Gauss-Seidel, Jacobi, Strang).  Supports under-relaxation
  and convergence monitoring on user-specified fields.

**Design rationale:** Partitioned coupling is chosen as the default because
it allows reuse of existing single-physics solvers without modification.
Monolithic coupling is provided for problems where partitioned iteration
does not converge (e.g., added-mass instability in FSI).

### 2. Adjoint / Sensitivity (`ontic/platform/adjoint.py`)

- `CostFunctional` (ABC) — Scalar objective with `evaluate()` and
  `dJ_dstate()` methods.
- `L2TrackingCost` — L2 distance between simulation output and target data.
- `AdjointSolver` — PyTorch autograd-based discrete adjoint.  **Finite-
  difference fallback** added to handle solvers that break the autograd
  graph (common for operator-splitting / explicit methods).
- `CheckpointedAdjoint` — Memory-efficient variant using Griewank-style
  checkpointing.  Same finite-difference fallback.

### 3. Inverse Problem Toolkit (`ontic/platform/inverse.py`)

- `Regularizer` (ABC) with `TikhonovRegularizer` (L2) and `TVRegularizer`
  (total variation).
- `InverseProblem` — Combines forward solver, cost, and regularizer.
  Provides `total_cost()` and `total_gradient()`.
- `GradientDescentSolver` — Simple momentum-free gradient descent.
- `LBFGSSolver` — Wraps `torch.optim.LBFGS` with configurable line search.
- `BayesianInversion` — Laplace approximation: MAP estimate via L-BFGS,
  Hessian via finite differences, posterior as N(MAP, H⁻¹), log-evidence
  estimate.

### 4. UQ Toolkit (`ontic/platform/uq.py`)

- `ParameterDistribution` — Supports uniform, normal, lognormal with
  inverse-CDF sampling.
- `MonteCarloUQ` — Brute-force sampling with user-defined state modifier
  and QoI extractors.  Returns mean, variance, std, percentiles.
- `LatinHypercubeUQ` — Space-filling LHS via stratified sampling +
  inverse CDF.  Extends `MonteCarloUQ`.
- `PolynomialChaosExpansion` — Non-intrusive PCE with Legendre basis,
  total-order multi-index, least-squares coefficient regression,
  and analytical mean/variance from converged coefficients.

### 5. Optimization Toolkit (`ontic/platform/optimization.py`)

- `Constraint` — Name + evaluate/gradient callables for g(x) ≤ 0.
- `volume_fraction_constraint()` — Standard constraint for topology
  optimization.
- `ConstrainedOptimizer` — Augmented-Lagrangian method: outer loop
  updates multipliers and penalty; inner loop runs gradient descent.
- `TopologyOptimization` — SIMP (Solid Isotropic Material with
  Penalization) with OC update.  Includes density filter for
  regularization and bisection-based volume constraint enforcement.

### 6. Data Lineage DAG (`ontic/platform/lineage.py`)

- `LineageEvent` — Typed enumeration of 12 event kinds (forward solve,
  adjoint, coupling, QTT compress/decompress, TCI, inverse, UQ,
  optimization, checkpoint, custom).
- `LineageNode` — SHA-256 hashes of inputs/outputs, wall-clock time,
  parent references, arbitrary metadata.
- `LineageDAG` — O(1) lookup, topological ordering, ancestor/descendant
  queries, event filtering, JSON serialization/deserialization.
- `LineageTracker` — Context-manager API (`record()`) and fire-and-forget
  API (`record_instant()`) for recording provenance events.

## Consequences

- 6 new platform modules (total ~2000 lines of production code).
- 27 new tests, all passing.
- All 200+ existing tests unaffected.
- Finite-difference fallback in the adjoint solver ensures robustness
  with any forward solver, at the cost of O(n_params) extra forward
  evaluations.
- The UQ toolkit is non-intrusive — any existing solver can be wrapped
  without modification.
- Lineage DAG enables full audit trail for regulatory/certification
  workflows.
- Future work: operator-overloading intrusive UQ, shape optimization,
  adjoint-based adaptive mesh refinement, distributed coupling via MPI.
