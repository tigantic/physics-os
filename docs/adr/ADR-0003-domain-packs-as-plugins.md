# ADR-0003: Domain Packs as Plugins

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-08 |
| **Author** | @core-team |

## Context

The platform currently has ~70 `tensornet/` subdirectories containing solver code for all 140 taxonomy nodes. These are flat Python modules imported directly — there is no plugin interface, no interface compliance enforcement, and no standardized registration mechanism.

This creates several problems:

1. **No interface conformance.** Solvers can define ad-hoc APIs, making composition impossible.
2. **No isolation.** A broken import in `tensornet/plasma/` breaks unrelated downstream code.
3. **No independent versioning.** All 140 nodes share one version number via `tensornet` package.
4. **No enforcement.** Anyone can merge a solver that doesn't use the canonical field/operator abstractions.

## Decision

Domain packs (Packs I–XX) will be restructured as **plugins** that register with the core platform via a `DomainPack.register()` interface.

Requirements for every domain pack:

1. **Implement `DomainPack` protocol:** expose `register()`, `list_solvers()`, `get_problem_spec(node_id)`.
2. **Use canonical abstractions:** `ProblemSpec`, `Discretization`, `Operator`, `Solver`, `Observable` from the core platform.
3. **Use canonical output schema:** all solver outputs conform to platform I/O format (HDF5/Zarr with embedded metadata).
4. **Use canonical test harness:** entry points for unit, regression, benchmark, and MMS tests.
5. **Pass interface compliance checks in CI:** a linter/validator confirms protocol implementation.

The transition is phased:

- Phase 0–1: Define the `DomainPack` protocol and canonical interfaces.
- Phase 3: Migrate existing flat modules to pack structure (one pack per taxonomy category).
- Phase 7: Packs are independently releasable.

## Consequences

- **Easier:** Composing multi-physics workflows across packs.
- **Easier:** Independent pack development, testing, and release.
- **Easier:** CI can enforce interface compliance automatically.
- **Harder:** Migration of 140 existing modules to the plugin architecture.
- **Harder:** Developers must learn the pack registration protocol.
- **Risk:** Over-engineering the plugin interface too early. Mitigate by keeping the protocol minimal and evolving via ADRs.
