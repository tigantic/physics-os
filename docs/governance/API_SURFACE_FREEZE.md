# API Surface Freeze — v4.0.0

**Baseline commit**: `569ff1da`
**Effective from**: `v4.0.0`
**Branch**: `release/v4.0.x`
**Policy**: No additions, removals, or behavioral changes unless a
launch gate failure requires it and the change is documented here.

---

## 1. Frozen Endpoints

| # | Method | Path                           | Auth | Status   |
|---|--------|--------------------------------|------|----------|
| 1 | POST   | `/v1/jobs`                     | Yes  | FROZEN   |
| 2 | GET    | `/v1/jobs/{job_id}`            | Yes  | FROZEN   |
| 3 | GET    | `/v1/jobs/{job_id}/result`     | Yes  | FROZEN   |
| 4 | GET    | `/v1/jobs/{job_id}/validation` | Yes  | FROZEN   |
| 5 | GET    | `/v1/jobs/{job_id}/certificate`| Yes  | FROZEN   |
| 6 | POST   | `/v1/validate`                 | No   | FROZEN   |
| 7 | GET    | `/v1/capabilities`             | No   | FROZEN   |
| 8 | GET    | `/v1/contracts/{version}`      | No   | FROZEN   |
| 9 | GET    | `/v1/health`                   | No   | FROZEN   |

**Rule**: No new endpoints may be added to the v1 prefix during this
hardening cycle.  Internal-only endpoints (metrics, debug) must use
a different prefix (e.g., `/_internal/`).

---

## 2. Frozen Request Schemas

### POST /v1/jobs — SubmitJobRequest

```json
{
  "job_type":           "string (enum, REQUIRED)",
  "domain":             "string | null",
  "n_bits":             "integer (4–14, default 8)",
  "n_steps":            "integer (1–10000, default 100)",
  "dt":                 "float | null (>0)",
  "max_rank":           "integer (2–128, default 64)",
  "parameters":         "object (default {})",
  "return_fields":      "boolean (default true)",
  "return_coordinates": "boolean (default true)",
  "artifact_bundle":    "object | null"
}
```

#### Frozen job_type enum values

| Value                    | Frozen |
|--------------------------|--------|
| `full_pipeline`          | ✓      |
| `physics_vm_execution`   | ✓      |
| `rank_atlas_benchmark`   | ✓      |
| `rank_atlas_diagnostic`  | ✓      |
| `validation`             | ✓      |
| `attestation`            | ✓      |

#### Frozen headers

| Header              | Purpose               |
|---------------------|------------------------|
| `Authorization`     | Bearer API key         |
| `X-Idempotency-Key` | Idempotent submission  |

### POST /v1/validate — Body

```json
{
  "(any)": "Accepts full envelope or raw result payload"
}
```

Detection rules:
- `"envelope_version"` field present → envelope mode
- `"grid"`, `"conservation"`, or `"fields"` present → raw result mode
- Neither → 400 with E009

---

## 3. Frozen Response Schemas

### Job Status Response (GET /v1/jobs/{id}, POST /v1/jobs)

```json
{
  "job_id":               "string (UUID)",
  "job_type":             "string (enum)",
  "state":                "string (enum)",
  "created_at":           "string (ISO-8601)",
  "completed_at":         "string | null (ISO-8601)",
  "input_manifest_hash":  "string (sha256:...)",
  "versions": {
    "api_version":        "string",
    "schema_version":     "string",
    "runtime_version":    "string"
  },
  "error":                "ErrorObject | null"
}
```

#### Frozen state enum values

| Value       | Terminal |
|-------------|----------|
| `queued`    | No       |
| `running`   | No       |
| `succeeded` | No       |
| `failed`    | Yes      |
| `validated` | No       |
| `attested`  | Yes      |

### Artifact Envelope (GET /v1/jobs/{id}/result)

```json
{
  "envelope_version":     "string",
  "job_id":               "string",
  "job_type":             "string",
  "status":               "string",
  "created_at":           "string",
  "completed_at":         "string | null",
  "idempotency_key":      "string | null",
  "input_manifest_hash":  "string",
  "result":               "ResultObject | null",
  "validation":           "ValidationObject | null",
  "certificate":          "CertificateObject | null",
  "artifact_hashes": {
    "input":              "string | null",
    "result":             "string | null",
    "validation":         "string | null",
    "certificate":        "string | null"
  },
  "versions":             "VersionsObject",
  "error":                "ErrorObject | null"
}
```

### Result Object (within envelope)

```json
{
  "domain":           "string",
  "domain_label":     "string",
  "equation":         "string",
  "parameters":       "object",
  "grid": {
    "dimensions":     "integer",
    "resolution":     "integer[]",
    "domain_bounds":  "float[][]",
    "coordinates":    "object (optional)"
  },
  "fields": {
    "<name>": {
      "name":   "string",
      "shape":  "integer[]",
      "values": "float[] | float[][]",
      "unit":   "string"
    }
  },
  "conservation": {
    "quantity":        "string",
    "initial_value":   "float",
    "final_value":     "float",
    "relative_error":  "float",
    "status":          "string (conserved|drift)"
  },
  "performance": {
    "wall_time_s":       "float",
    "grid_points":       "integer",
    "time_steps":        "integer",
    "throughput_gp_per_s": "float"
  }
}
```

### Validation Report (GET /v1/jobs/{id}/validation)

```json
{
  "valid":          "boolean",
  "checks": [
    {
      "name":       "string",
      "passed":     "boolean",
      "detail":     "string",
      "severity":   "string (error|info)"
    }
  ],
  "validated_at":   "string (ISO-8601)"
}
```

### Trust Certificate (GET /v1/jobs/{id}/certificate)

```json
{
  "certificate_version": "string",
  "job_id":              "string",
  "issued_at":           "string (ISO-8601)",
  "issuer":              "string",
  "claims": [
    {
      "tag":       "string",
      "claim":     "string",
      "witness":   "object",
      "satisfied": "boolean"
    }
  ],
  "input_manifest_hash": "string",
  "result_hash":         "string",
  "replay_metadata": {
    "runtime_version":   "string",
    "config_hash":       "string",
    "seed":              "integer | null",
    "device_class":      "string"
  },
  "signature":           "string (algo:hex)"
}
```

### Health Response (GET /v1/health)

```json
{
  "status":          "string",
  "uptime_seconds":  "float",
  "versions": {
    "api":           "string",
    "runtime":       "string",
    "schema":        "string"
  }
}
```

### Capabilities Response (GET /v1/capabilities)

```json
{
  "domain_count":    "integer",
  "domains":         "DomainSpec[]",
  "job_types":       "JobTypeSpec[]",
  "contract_version": "string"
}
```

---

## 4. Frozen Error Envelope

Every error response uses this shape:

```json
{
  "code":      "string (E001–E012)",
  "message":   "string",
  "retryable": "boolean"
}
```

Validation errors (422) add:

```json
{
  "code":      "E003",
  "message":   "string",
  "details":   "array (Pydantic validation errors)",
  "retryable": false
}
```

#### Frozen Error Code Table

| Code   | HTTP | Meaning                     | Retryable |
|--------|------|-----------------------------|-----------|
| `E001` | 400  | Invalid domain              | No        |
| `E002` | 400  | Parameter out of range      | No        |
| `E003` | 422  | Invalid request payload     | No        |
| `E004` | 404  | Job not found               | No        |
| `E005` | 409  | Job not in required state   | Yes       |
| `E006` | 200* | Simulation diverged         | No        |
| `E007` | 200* | Execution timeout           | Yes       |
| `E008` | 200* | Validation failed           | No        |
| `E009` | 400  | Invalid artifact bundle     | No        |
| `E010` | 429  | Rate limit exceeded         | Yes       |
| `E011` | 401  | Authentication failed       | No        |
| `E012` | 500  | Internal error (opaque)     | Yes       |

*E006–E008 appear in the job's error field, not as HTTP errors.

---

## 5. Frozen Certificate Envelope Format

| Field                  | Type    | Frozen |
|------------------------|---------|--------|
| `certificate_version`  | string  | ✓      |
| `job_id`               | string  | ✓      |
| `issued_at`            | string  | ✓      |
| `issuer`               | string  | ✓      |
| `claims`               | array   | ✓      |
| `input_manifest_hash`  | string  | ✓      |
| `result_hash`          | string  | ✓      |
| `replay_metadata`      | object  | ✓      |
| `signature`            | string  | ✓      |

---

## 6. What Is Returned Policy

### Always returned

- Physical observables (dense field arrays, rounded to `field_precision`)
- Conservation metrics (quantity, initial, final, relative_error, status)
- Validation status with named checks
- Certificate metadata with claims and signature
- Content hashes for tamper detection
- Version metadata

### Never returned

- TT cores or bond dimensions
- Singular value spectra
- Compression ratios
- Rank evolution traces
- Scaling classifications (A/B/C/D)
- IR opcodes or instruction counts
- Register counts
- Internal file paths or class names
- Stack traces or exception internals
- Server environment variables
- API keys in responses
- Raw internal telemetry

---

## 7. Freeze Violation Protocol

If a launch gate failure requires a schema change:

1. Document the violation in this file under §8 (Amendments)
2. Bump `schema_version` accordingly
3. Ensure backwards compatibility (additive only)
4. Update `contracts/v1/SPEC.md` and `envelope.schema.json`
5. Tag the change as a PATCH (non-breaking) or MINOR (additive)
6. Notify alpha users 48h in advance if applicable

---

## 8. Amendments

| Date | Version | Change | Reason |
|------|---------|--------|--------|
| —    | —       | —      | —      |

*No amendments recorded.  Baseline is clean.*
