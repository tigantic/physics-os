# ADR-0004: Canonical Platform Protocols

**Date:** 2026-02-08  
**Status:** Accepted  
**Deciders:** Platform Architecture  

## Context

The HyperTensor-VM codebase contains three independent `Field` classes,
two `Operator` hierarchies, and zero common `Solver` interface.  Over 40
domain-specific solvers exist — each with different constructor signatures,
different step/advance method names, and different state representations.

Solver dispatch, benchmark orchestration, and domain-pack composition are
impossible without a shared contract.

## Decision

Introduce six **PEP 544 `Protocol` interfaces** in `ontic.platform.protocols`:

| Protocol | Purpose |
|---|---|
| `ProblemSpec` | Declares the continuous problem (equations, domain, BC/IC, observables) |
| `Discretization` | Maps a continuous problem onto a discrete system |
| `OperatorProto` | A discrete operator: `apply(field, **kwargs) → field` |
| `Solver` | `step(state, dt)` + `solve(state, t_span, dt)` |
| `Observable` | `compute(state) → Tensor` |
| `Workflow` | End-to-end: `run(**overrides) → WorkflowResult` |

These are **structural protocols**, not ABC subclasses.  Any existing class
that already has the right shape satisfies the protocol without modification.

## Consequences

- All new domain-pack code must satisfy `Solver` and `ProblemSpec` at minimum.
- Existing solvers can be retrofitted incrementally (add a `step` method).
- The three `Field` classes remain for now; unification is deferred to
  Phase 3 (the data-model bridge is via `SimulationState`, which wraps
  field data in `FieldData` containers agnostic to the underlying class).
- Observable protocol enables the V&V harness to work across all domains.
