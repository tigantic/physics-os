# ADR-0015: Whitelist-Only IP Sanitization

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

The HyperTensor-VM API accepts user-defined simulation parameters including mesh dimensions, material properties, boundary conditions, and solver tolerances. Several parameters map directly to memory allocation sizes, iteration counts, or file paths. A permissive input model creates attack surface for:

1. **Resource exhaustion**: Mesh dimensions of 2^64 causing OOM.
2. **Path traversal**: Material database paths escaping the data directory.
3. **Injection**: Specially crafted parameter names/values in log output.
4. **Forbidden outputs**: Results that violate `FORBIDDEN_OUTPUTS.md` constraints (denormalized floats, NaN propagation).

A blacklist approach (reject known-bad patterns) is inherently incomplete — novel attack vectors bypass the list.

## Decision

**All input validation uses whitelist (allowlist) filtering exclusively.** Specifically:

1. Every API parameter has an explicit type, range, and enum constraint defined in `PLATFORM_SPECIFICATION.md` §5.
2. Mesh dimensions: integer ∈ [8, 1048576], must be power-of-2.
3. Solver tolerances: float ∈ [1e-15, 1e-1].
4. Material properties: enum from `data/materials/*.yaml` registry only.
5. File paths: resolved and verified to be within `$HYPERTENSOR_DATA_ROOT` — no `..`, no symlink escape.
6. String parameters: ASCII printable only, max 256 characters, regex `^[a-zA-Z0-9_\-\.]+$`.
7. Any value outside the whitelist returns HTTP 422 with the specific constraint violation.

The `ERROR_CODE_MATRIX.md` defines all rejection codes.

## Consequences

- **Easier:** Security audits — every parameter's valid domain is documented and testable.
- **Easier:** Deterministic behavior — no surprising edge-case inputs reach the solver.
- **Harder:** Adding new parameters requires explicit schema updates (not just "pass through").
- **Risk:** Overly restrictive ranges block legitimate use cases. Mitigated by per-domain-pack override schemas with elevated review requirements.
