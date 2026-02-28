# Physics OS Platform Shell Contract — v1

**Version**: 1.0.0
**Status**: Active
**Date**: 2026-02-24

---

## 1. Product Boundary

### 1.1 What Is Exposed

| Surface        | Exposed To Client                                                       |
|----------------|-------------------------------------------------------------------------|
| Job submission | Domain selection, physical parameters, resolution, boundary conditions  |
| Execution      | Job state transitions, progress tokens                                  |
| Results        | Physical observables (fields, energies, forces), conservation metrics   |
| Diagnostics    | Validation reports, benchmark verdicts, consistency checks              |
| Evidence       | Signed artifact bundles, trust certificates, witness predicates         |
| Contracts      | Versioned JSON schemas, API spec, capability manifest                   |

### 1.2 What Is NOT Exposed

| Internal                         | Containment Rule                                      |
|----------------------------------|-------------------------------------------------------|
| Runtime source code              | Never distributed.  Licensed execution only.          |
| Compiler implementations         | Server-side only.  No client libraries contain logic. |
| Operator kernels / algebra       | No intermediate representations in responses.         |
| Bond dimensions / TT cores       | Stripped before serialization.                         |
| Singular value spectra            | Stripped before serialization.                         |
| Compression ratios / rank data   | Stripped before serialization.                         |
| Scaling classifications          | Internal diagnostic only.                             |
| IR instructions / opcodes        | Internal diagnostic only.                             |
| Stack traces / internal errors   | Machine-readable error codes only.                    |
| Class names / file paths         | Redacted from all responses.                          |

---

## 2. Contract Versioning

- **Major** (v1 → v2): Breaking schema changes.  Prior versions supported for 12 months.
- **Minor** (v1.1): Additive fields only.  Backwards compatible.
- **Patch** (v1.0.1): Bug fixes, documentation.  No schema changes.

Every response includes:
```json
{
  "api_version": "1.0.0",
  "schema_version": "1.0.0",
  "runtime_version": "3.1.0"
}
```

Backwards compatibility policy: **additive only in minor versions**.  No field removal, no type changes, no semantic changes to existing fields.

---

## 3. Job Model

### 3.1 Job Types

| Type                           | Purpose                                              |
|--------------------------------|------------------------------------------------------|
| `physics_vm_execution`         | Run a physics simulation (closed runtime)            |
| `rank_atlas_benchmark`         | Full Rank Atlas benchmark suite                      |
| `rank_atlas_diagnostic`        | Single-pack or single-dataset diagnostic             |
| `validation`                   | Validate an existing result bundle                   |
| `attestation`                  | Generate trust certificate from result bundle        |

### 3.2 Job States

```
queued → running → succeeded → validated → attested
                 ↘ failed
```

| State       | Terminal | Description                                    |
|-------------|----------|------------------------------------------------|
| `queued`    | No       | Accepted, awaiting execution                   |
| `running`   | No       | Executing on compute backend                   |
| `succeeded` | No       | Execution complete, result available            |
| `failed`    | Yes      | Execution failed (error code + message)        |
| `validated` | No       | Result has been validated against contract      |
| `attested`  | Yes      | Trust certificate generated and signed          |

### 3.3 Idempotency

`POST /v1/jobs` accepts an `idempotency_key` header.  Duplicate submissions with the same key return the existing job without re-executing.

### 3.4 Input Hashing

Every job computes a deterministic `input_manifest_hash` (SHA-256) from the canonical JSON representation of the input parameters.  This enables:
- Deduplication
- Cache hits
- Reproducibility verification

---

## 4. Artifact Envelope

Every job result is wrapped in a standardized envelope:

```json
{
  "envelope_version": "1.0.0",
  "job_id": "uuid",
  "job_type": "physics_vm_execution",
  "created_at": "ISO-8601",
  "input_manifest_hash": "sha256:...",
  "result": { ... },
  "validation": { ... },
  "certificate": { ... },
  "artifact_hashes": {
    "result": "sha256:...",
    "validation": "sha256:...",
    "certificate": "sha256:..."
  },
  "versions": {
    "api_version": "1.0.0",
    "schema_version": "1.0.0",
    "runtime_version": "3.1.0"
  }
}
```

### 4.1 Content Addressing

Every payload within an envelope is independently hashable.  Hash algorithm: SHA-256 over canonical JSON (sorted keys, no whitespace, UTF-8).

### 4.2 Numeric Representation

| Value         | Representation                | Notes                            |
|---------------|-------------------------------|----------------------------------|
| Normal float  | JSON number                   | Rounded to `field_precision`     |
| NaN           | `null` + `"_status": "nan"`   | Never string `"NaN"`            |
| ±Infinity     | `null` + `"_status": "inf"`   | Never string `"Infinity"`       |
| Missing       | `null` + `"_status": "missing"` | Explicit absence              |

---

## 5. Result Schema

### 5.1 Public Fields (always present)

| Field                        | Type          | Description                              |
|------------------------------|---------------|------------------------------------------|
| `job_id`                     | string (UUID) | Unique job identifier                    |
| `domain`                     | string        | Physics domain key                       |
| `domain_label`               | string        | Human-readable domain name               |
| `equation`                   | string        | Governing equation                       |
| `parameters`                 | object        | Physical parameters used                 |
| `grid.dimensions`            | integer       | Spatial dimensions                       |
| `grid.resolution`            | int[]         | Grid points per dimension                |
| `grid.domain_bounds`         | float[][]     | Physical domain bounds                   |
| `fields`                     | object        | Named field arrays (dense)               |
| `conservation.quantity`      | string        | Conserved quantity name                   |
| `conservation.initial_value` | float         | Value at t=0                             |
| `conservation.final_value`   | float         | Value at t_final                         |
| `conservation.relative_error`| float         | |(final−initial)/initial|                |
| `conservation.status`        | enum          | `conserved` or `drift`                   |
| `performance.wall_time_s`    | float         | Execution wall time                      |
| `performance.grid_points`    | integer       | Total grid points                        |
| `performance.time_steps`     | integer       | Steps executed                           |

### 5.2 Internal Fields (never exposed)

Bond dimensions, compression ratios, TT cores, singular values, scaling class, saturation rate, truncation counts, opcode lists, register counts.

---

## 6. Trust Certificate

```json
{
  "certificate_version": "1.0.0",
  "job_id": "uuid",
  "issued_at": "ISO-8601",
  "issuer": "hypertensor-runtime",
  "claims": [
    {
      "tag": "CONSERVATION",
      "claim": "total_mass preserved to 1e-6 relative error",
      "witness": { "initial": 1.0, "final": 0.999999, "rel_error": 1e-6 },
      "satisfied": true
    }
  ],
  "input_manifest_hash": "sha256:...",
  "result_hash": "sha256:...",
  "replay_metadata": {
    "runtime_version": "3.1.0",
    "config_hash": "sha256:...",
    "seed": 42,
    "device_class": "cuda"
  },
  "signature": "ed25519:..."
}
```

### 6.1 Claim Tags

| Tag             | Domain           | Description                              |
|-----------------|------------------|------------------------------------------|
| `CONSERVATION`  | All physics      | Invariant quantity preserved             |
| `BOUND`         | All              | Computed quantity within expected bound   |
| `STABILITY`     | All physics      | Solution remained numerically stable     |
| `CONVERGENCE`   | Benchmark        | Grid-independence criterion met          |
| `CONSISTENCY`   | Benchmark        | Cross-resolution consistency             |
| `DETERMINISM`   | All              | Replay produces identical result         |

---

## 7. API Endpoints

| Method | Path                              | Purpose                        | Auth |
|--------|-----------------------------------|--------------------------------|------|
| POST   | `/v1/jobs`                        | Submit a job                   | Yes  |
| GET    | `/v1/jobs/{job_id}`               | Get job status                 | Yes  |
| GET    | `/v1/jobs/{job_id}/result`        | Get result payload             | Yes  |
| GET    | `/v1/jobs/{job_id}/validation`    | Get validation report          | Yes  |
| GET    | `/v1/jobs/{job_id}/certificate`   | Get trust certificate          | Yes  |
| POST   | `/v1/validate`                    | Validate an artifact bundle    | Yes  |
| GET    | `/v1/contracts/{version}`         | Get contract schema            | No   |
| GET    | `/v1/capabilities`                | List domains + capabilities    | No   |
| GET    | `/v1/health`                      | Server health check            | No   |

---

## 8. Error Codes

| Code   | Meaning                            |
|--------|------------------------------------|
| `E001` | Invalid domain                     |
| `E002` | Parameter out of range             |
| `E003` | Resolution exceeds server limit    |
| `E004` | Job not found                      |
| `E005` | Job not in required state          |
| `E006` | Simulation diverged                |
| `E007` | Execution timeout                  |
| `E008` | Validation failed                  |
| `E009` | Invalid artifact bundle            |
| `E010` | Rate limit exceeded                |
| `E011` | Authentication failed              |
| `E012` | Internal error (opaque)            |

---

## 9. Security & Data Handling

### 9.1 Data Classes

| Class                | Handling                                     |
|----------------------|----------------------------------------------|
| Public benchmark     | Shareable, embeddable, citable               |
| Customer job input   | Encrypted at rest, deleted after retention   |
| Customer job output  | Encrypted at rest, customer-owned            |
| Runtime internals    | Never persisted, never transmitted            |

### 9.2 Retention

- Job metadata: 90 days
- Result payloads: 30 days (configurable)
- Certificates: Permanent (customer-owned artifact)
- Runtime logs: 7 days (internal only)

---

## 10. Licensing Matrix

| License          | Hosted API | Private VPC | On-Prem | Evaluation |
|------------------|:----------:|:-----------:|:-------:|:----------:|
| Execution        | ✓          | ✓           | ✓       | ✓ (limited)|
| Evidence bundles | ✓          | ✓           | ✓       | ✓          |
| Certificates     | ✓          | ✓           | ✓       | —          |
| Source code       | —          | —           | —       | —          |
| Custom adapters  | Contract   | Contract    | Contract| —          |
| SLA              | Standard   | Premium     | Premium | —          |
| Max resolution   | n_bits≤14  | Unlimited   | Unlimited| n_bits≤8  |
