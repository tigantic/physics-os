# ADR-0009: Phase 5 — QTT / TN Acceleration as First-Class Capability

**Status:** Accepted  
**Date:** 2026-02-09  
**Decision Makers:** Platform team  

## Context

Phases 0–4 established the platform substrate (data model, protocols, solvers,
domain packs, full taxonomy coverage at ≥ V0.2) but all forward solvers operate
on dense tensors.  The repository already contains mature QTT / TN primitives:

- `ontic/cfd/pure_qtt_ops.py` (1069 lines): TT-SVD, rounding, arithmetic.
- `ontic/cfd/qtt_tci.py` (1271 lines): Python TCI engine.
- `ontic/core/` + `ontic/algorithms/`: MPS, MPO, DMRG, TEBD, TDVP.
- `apps/qtenet/`: N-D QTT operators, NS3D / Vlasov / Euler solvers.

**Gap identified:** Zero bridge between QTT primitives and the platform data
model / protocols.  No domain pack can natively consume QTT fields, and there
is no policy framework to decide when QTT acceleration is beneficial vs.
harmful (rank explosion → accuracy loss).

## Decision

### 1. QTT Bridge Layer (`ontic/platform/qtt.py`)

- `QTTFieldData` — QTT analog of `FieldData`: stores TT cores, `n_qubits`,
  compression ratio, max rank.
- `field_to_qtt()` — Compress `FieldData` → `QTTFieldData` via TT-SVD with
  configurable max rank and tolerance.
- `qtt_to_field()` — Reconstruct `FieldData` from `QTTFieldData` by core
  contraction.
- `QTTOperator` — Wraps an MPO and satisfies `OperatorProto`.  Provides both
  `apply(dense)` and `apply_qtt(QTTFieldData)` paths.
- `QTTDiscretization` — Satisfies `Discretization` protocol for QTT-native
  discretizations.

### 2. TCI Decomposition Engine (`ontic/platform/tci.py`)

- `TCIConfig` / `TCIResult` — Frozen dataclasses for TCI parameters and
  outputs.
- `tci_from_function()` — Delegates to `ontic.cfd.qtt_tci`, falls back
  to TT-SVD on failure.  Handles heterogeneous output formats (lists, numpy,
  tensors).
- `tci_from_field()` — Field → QTT via interpolation-based TCI.
- `tci_error_vs_rank()` — Produces error-vs-rank curves for QTT Enablement
  Policy validation.

### 3. Acceleration Policy (`ontic/platform/acceleration.py`)

- `AccelerationMode` enum: DENSE | QTT | FALLBACK | HYBRID.
- `AccelerationMetrics` — Per-step metrics (rank, compression, error, time).
- `RankGrowthReport` — Aggregated report with peak/median rank, explosion
  detection, speedup measurement.
- `AccelerationPolicy` — Configurable thresholds for rank cap, error budget,
  compression floor, growth rate.  `should_use_qtt()` governs per-step
  mode selection; `validate_enablement()` checks post-solve criteria.

### 4. QTT Solver Wrapper (`ontic/platform/qtt_solver.py`)

- `QTTAcceleratedSolver` — Generic wrapper around any `Solver`-protocol solver.
  Compresses fields before each step, decompresses after, records metrics,
  and triggers fallback per the policy.
- `QTTSimulationState` — Pairs dense state with QTT-compressed fields.

### 5. V0.6 QTT-Accelerated Domain Solvers (`ontic/packs/qtt_accelerated.py`)

Four anchor-adjacent V0.6 solvers demonstrating QTT acceleration:

| Solver | Pack | Physics | QTT Strategy |
|--------|------|---------|-------------|
| `QTTBurgersSolver` | II (PHY-II.1) | 1-D viscous Burgers | Shift-MPO RK4 in TT format, mean-value linearization |
| `QTTAdvDiffSolver` | V (PHY-V.5) | Linear advection-diffusion | Full QTT operator application |
| `QTTMaxwellSolver` | III (PHY-III.3) | 1-D FDTD Maxwell | Dense leapfrog + QTT compression monitoring |
| `QTTVlasovSolver` | XI (PHY-XI.1) | 1D-1V Vlasov-Poisson | 2-D phase-space QTT + Strang-split semi-Lagrangian |

Each solver provides: `rank_growth_report()`, `error_vs_rank()`, automatic
dense fallback, and a `DOMAIN_OF_VALIDITY` string.

### MPO-MPS Contraction Fix

The einsum for MPO-MPS site contraction was corrected from `"ais,boji->abojs"`
(incorrect index overlap) to `"asb,cpsd->acpbd"` (proper physical-index
contraction).  This affects both `QTTOperator._apply_mpo_to_cores()` and
`_apply_shift_to_cores()`.

## Consequences

- 5 new platform modules, 1 new pack module (total ~1860 lines of production code).
- 28 new tests, all passing.
- Existing 200+ tests unaffected.
- QTT bridge is the single integration point between QTT primitives and the
  platform protocol layer.
- The acceleration policy is intentionally conservative: rank cap 128,
  error budget 1e-4, mandatory warmup before fallback decisions.
- Future work: QTT-native DMRG time integration, adaptive rank allocation,
  GPU QTT kernels.
