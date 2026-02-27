# Release Notes — v4.0.0 Baseline

**Tag**: `v4.0.0`
**Commit**: `569ff1da5334a0614179f1d06d6508ff479e6cd0`
**Date**: 2026-02-24
**Branch**: `release/v4.0.x` (hardening branch)
**Status**: Scope-frozen baseline for paid private alpha

---

## What This Release Contains

HyperTensor Runtime Access Layer — a 3-surface product for licensed
execution access and evidence guarantees over compression-native
physics compute.

### Surfaces

| Surface | Entry Point                        | Status   |
|---------|------------------------------------|----------|
| API     | `hypertensor.api.app:app`          | Verified |
| SDK     | `hypertensor.sdk.client`           | Verified |
| MCP     | `hypertensor.mcp.server`           | Verified |
| CLI     | `python -m hypertensor.cli`        | Verified |

### Core Layer (server-side only, never distributed)

| Module                     | Purpose                                      |
|----------------------------|----------------------------------------------|
| `hypertensor.core.hasher`  | Content-addressed SHA-256 over canonical JSON |
| `hypertensor.core.registry`| 7-domain compiler registry                   |
| `hypertensor.core.executor`| Bridges jobs to tensornet.vm runtime          |
| `hypertensor.core.sanitizer`| IP-safe output filtering                    |
| `hypertensor.core.evidence`| Validation reports + claim-witness generation |
| `hypertensor.core.certificates`| Ed25519 signing + verification            |

### Job System

| Component                   | Purpose                                     |
|-----------------------------|---------------------------------------------|
| `hypertensor.jobs.models`   | State machine (6 states, 6 job types)       |
| `hypertensor.jobs.store`    | Thread-safe in-memory store + idempotency   |

### Contracts

| File                              | Purpose                                |
|-----------------------------------|----------------------------------------|
| `contracts/v1/SPEC.md`           | Product boundary, job model, schemas   |
| `contracts/v1/envelope.schema.json`| JSON Schema for artifact envelopes   |

---

## Supported Job Types

| Type                    | Code                       | Status   |
|-------------------------|----------------------------|----------|
| Full Pipeline           | `full_pipeline`            | Verified |
| Physics VM Execution    | `physics_vm_execution`     | Verified |
| Rank Atlas Benchmark    | `rank_atlas_benchmark`     | Verified |
| Rank Atlas Diagnostic   | `rank_atlas_diagnostic`    | Verified |
| Validation Only         | `validation`               | Verified |
| Attestation Only        | `attestation`              | Verified |

## Supported Physics Domains

| Domain                 | Key                    | Dimensions |
|------------------------|------------------------|------------|
| Viscous Burgers        | `burgers`              | 1D         |
| Maxwell TE Mode        | `maxwell`              | 1D         |
| Maxwell 3D             | `maxwell_3d`           | 3D         |
| Schrödinger            | `schrodinger`          | 1D         |
| Advection-Diffusion    | `advection_diffusion`  | 1D         |
| Vlasov-Poisson         | `vlasov_poisson`       | 2D         |
| Navier-Stokes 2D       | `navier_stokes_2d`     | 2D         |

## API Endpoints (9 total)

| Method | Path                           | Auth | Verified |
|--------|--------------------------------|------|----------|
| POST   | `/v1/jobs`                     | Yes  | ✓        |
| GET    | `/v1/jobs/{id}`                | Yes  | ✓        |
| GET    | `/v1/jobs/{id}/result`         | Yes  | ✓        |
| GET    | `/v1/jobs/{id}/validation`     | Yes  | ✓        |
| GET    | `/v1/jobs/{id}/certificate`    | Yes  | ✓        |
| POST   | `/v1/validate`                 | No   | ✓        |
| GET    | `/v1/capabilities`             | No   | ✓        |
| GET    | `/v1/contracts/{version}`      | No   | ✓        |
| GET    | `/v1/health`                   | No   | ✓        |

## End-to-End Verification Record

**Test performed**: 2026-02-24
**Job**: `full_pipeline` on Burgers domain, n_bits=6, n_steps=50

| Stage                          | Result                              |
|--------------------------------|-------------------------------------|
| Job submission                 | 201 Created, job_id assigned        |
| State progression              | queued → running → succeeded → validated → attested |
| Wall time                      | 4.5s total lifecycle                |
| Certificate                    | Ed25519 signed, 3 claims           |
| Stateless re-validation        | envelope_valid ✓                   |
| Hash verification              | result_hash_verified ✓             |
| Signature verification         | certificate_signature_valid ✓      |

---

## Scope Freeze

This release is the **frozen baseline** for the paid private alpha
program.  The following scope constraints are in effect:

- No new physics domains
- No new opcodes or compiler modifications
- No new API endpoints or surface area
- No schema changes without version bump
- Bug fixes and hardening only on `release/v4.0.x`

## What Happens Next

All post-baseline work is tracked in `LAUNCH_READINESS.md` and flows
through the launch gate checklist before reaching the release branch.

---

## File Manifest (31 files, 3965 lines)

```
contracts/v1/SPEC.md
contracts/v1/envelope.schema.json
hypertensor/__init__.py
hypertensor/api/__init__.py
hypertensor/api/app.py
hypertensor/api/auth.py
hypertensor/api/config.py
hypertensor/api/routers/__init__.py
hypertensor/api/routers/capabilities.py
hypertensor/api/routers/contracts.py
hypertensor/api/routers/health.py
hypertensor/api/routers/jobs.py
hypertensor/api/routers/validate.py
hypertensor/cli/__init__.py
hypertensor/cli/__main__.py
hypertensor/cli/main.py
hypertensor/core/__init__.py
hypertensor/core/certificates.py
hypertensor/core/evidence.py
hypertensor/core/executor.py
hypertensor/core/hasher.py
hypertensor/core/registry.py
hypertensor/core/sanitizer.py
hypertensor/jobs/__init__.py
hypertensor/jobs/models.py
hypertensor/jobs/store.py
hypertensor/mcp/__init__.py
hypertensor/mcp/server.py
hypertensor/sdk/__init__.py
hypertensor/sdk/client.py
```
