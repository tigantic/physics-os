# ADR-0008: Phase 4 — Full Taxonomy Baseline Coverage

**Status:** Accepted  
**Date:** 2025-01-24  
**Decision Makers:** Platform team  

## Context

Phase 3 delivered 6 anchor packs at V0.4 and 14 scaffold packs at V0.1.
The scaffold packs contained `_GenericAdvectionSolver` or `_ScaffoldWaveSolver`
stubs that ran generic (wrong-physics) advection or identity operations.  Anchor
packs II, III, and VII each had real anchor nodes but also scaffold stubs for
non-anchor nodes.

Phase 4's exit gate requires **100 % of taxonomy nodes at ≥ V0.2 Correctness**
and **Tier A nodes at ≥ V0.4 Validated** (the latter was partially achieved via
anchor vertical slices in Phase 3 and extends to all Tier A nodes in subsequent
phases).

## Decision

1. **Extend `_base.py`** with a reusable V0.2 solver pattern library:
   `ODEReferenceSolver`, `PDE1DReferenceSolver`, `EigenReferenceSolver`,
   `MonteCarloReferenceSolver`, and `validate_v02()`.

2. **Rewrite all 14 pure-scaffold packs** (I, IV, VI, IX, X, XII–XX) with
   real physics solvers implementing the canonical test case for each node.
   Each pack version bumped from `"0.1.0"` to `"0.2.0"`.

3. **Replace 26 scaffold solvers in anchor packs** II (9 nodes), III (6 nodes),
   and VII (11 nodes) with dedicated physics solvers while preserving the
   anchor vertical-slice code verbatim.

4. **Fix pre-existing defense module bug** (`callable | None` → `Callable | None`
   with `from __future__ import annotations`).

5. **Update capability ledger**: 167 YAML entries, 7 at V0.4, 160 at V0.2.
   31 new entries created, 4 orphan entries removed.

6. **Generate coverage dashboard** (`docs/COVERAGE_DASHBOARD.md`).

## Consequences

- **167 / 167 nodes at ≥ V0.2 Correctness** — exit gate satisfied.
- 7 anchor nodes validated at V0.4 (II.1, III.3, V.1, VII.1, VII.2, VIII.1, XI.1).
- 19 Tier A nodes all at ≥ V0.2; 5 of those at V0.4 via vertical slices.
- Zero scaffold solvers remain in the codebase.
- 257 tests pass (185 core + 72 ancillary), 1 skipped (empty scaffold set).
- Solver implementations cover: ODE integration, PDE finite differences,
  exact diagonalization, transfer-matrix methods, self-consistent field
  iteration, Monte Carlo sampling, analytical solutions, Strang splitting,
  Lindblad master equations, and beyond.

## Risks

- V0.2 solvers use first- or second-order schemes on coarse grids.  They
  reproduce reference solutions within tolerance but are not production-grade
  for high-resolution simulations.
- Quantum many-body solvers (Pack VII) use exact diagonalization limited to
  small system sizes (N ≤ 12).  Scalability requires MPS/DMRG at V0.4+.
