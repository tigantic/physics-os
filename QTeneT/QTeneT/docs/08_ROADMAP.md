# Roadmap (Draft)

## Phase 0 — Extraction
- [ ] Freeze an authoritative inventory of QTT artifacts (this repo)
- [ ] Identify canonical implementations (choose winners per primitive)

## Phase 1 — Stabilize Core
- [ ] Single `QTTTensor` container type
- [ ] Deterministic decomposition + rounding API
- [ ] Golden tests for: point-eval, shift, laplacian, hadamard

## Phase 2 — Operators + Solvers
- [ ] MPO operator library with versioned schemes
- [ ] IMEX/TDVP integrator packages

## Phase 3 — Codec + Services
- [ ] Standard QTT container format (schema, versioning)
- [ ] Query service (HTTP/gRPC) + mmap local mode

## Phase 4 — Enterprise
- [ ] SECURITY.md, threat model
- [ ] CI: lint/typecheck/tests/bench gates
- [ ] Observability: structured logs + run manifests
