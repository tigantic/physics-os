# Spec: Operator Versioning

## Objective
Operators (MPO builders) must be **versioned** so solver results remain reproducible and debuggable.

## Operator identity
An operator is identified by:
- `name` (e.g., `laplacian`, `shift`, `gradient`)
- `scheme` (e.g., `cd2`, `upwind1`, `weno5`)
- `version` (semantic, e.g., `v1`, `v2`)
- `grid_layout` (binary quantics layout assumptions)

## Requirements
1. Each operator builder must return metadata capturing the identity.
2. Solver runs must emit operator identities into a run manifest.

## Suggested structure
- `qtenet.operators.laplacian(..., scheme='cd2', version='v1')`
- returns `(mpo, meta)` where `meta` is JSON-serializable.

## Backward compatibility
- `v1` operators must remain available once shipped.
- breaking changes require new `version`.
