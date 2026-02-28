# Release Notes — v4.0.0 Baseline

**Tag**: `v4.0.0`
**Commit**: `569ff1da5334a0614179f1d06d6508ff479e6cd0`
**Date**: 2026-02-24
**Branch**: `release/v4.0.x` (hardening branch)
**Status**: Scope-frozen baseline for paid private alpha

---

## What This Release Contains

Physics OS Platform Shell — a 3-surface product for licensed
execution access and evidence guarantees over compression-native
physics compute.

### Surfaces

| Surface | Entry Point                        | Status   |
|---------|------------------------------------|----------|
| API     | `physics_os.api.app:app`          | Verified |
| SDK     | `physics_os.sdk.client`           | Verified |
| MCP     | `physics_os.mcp.server`           | Verified |
| CLI     | `python -m physics_os.cli`        | Verified |

### Core Layer (server-side only, never distributed)

| Module                     | Purpose                                      |
|----------------------------|----------------------------------------------|
| `physics_os.core.hasher`  | Content-addressed SHA-256 over canonical JSON |
| `physics_os.core.registry`| 7-domain compiler registry                   |
| `physics_os.core.executor`| Bridges jobs to ontic.vm runtime          |
| `physics_os.core.sanitizer`| IP-safe output filtering                    |
| `physics_os.core.evidence`| Validation reports + claim-witness generation |
| `physics_os.core.certificates`| Ed25519 signing + verification            |

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
physics_os/__init__.py
physics_os/api/__init__.py
physics_os/api/app.py
physics_os/api/auth.py
physics_os/api/config.py
physics_os/api/routers/__init__.py
physics_os/api/routers/capabilities.py
physics_os/api/routers/contracts.py
physics_os/api/routers/health.py
physics_os/api/routers/jobs.py
physics_os/api/routers/validate.py
physics_os/cli/__init__.py
physics_os/cli/__main__.py
physics_os/cli/main.py
physics_os/core/__init__.py
physics_os/core/certificates.py
physics_os/core/evidence.py
physics_os/core/executor.py
physics_os/core/hasher.py
physics_os/core/registry.py
physics_os/core/sanitizer.py
physics_os/jobs/__init__.py
physics_os/jobs/models.py
physics_os/jobs/store.py
physics_os/mcp/__init__.py
physics_os/mcp/server.py
physics_os/sdk/__init__.py
physics_os/sdk/client.py
```
