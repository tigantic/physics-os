# Error Code Matrix

**Baseline**: v4.0.0
**Status**: FROZEN — no new codes without version bump

---

## Error Envelope Shape

Every error response uses the same envelope:

```json
{
  "code":      "E0XX",
  "message":   "Human-readable description",
  "retryable": true | false
}
```

Validation errors (HTTP 422) extend the envelope:

```json
{
  "code":      "E003",
  "message":   "Invalid request payload.",
  "details":   [ { "loc": [...], "msg": "...", "type": "..." } ],
  "retryable": false
}
```

---

## Full Error Matrix

| Code   | HTTP | Condition                        | Retryable | Client Guidance                                                         |
|--------|------|----------------------------------|-----------|-------------------------------------------------------------------------|
| `E001` | 400  | Invalid or unknown domain        | No        | Check `/v1/capabilities` for supported domains.                         |
| `E002` | 400  | Parameter out of range           | No        | Adjust parameter to within documented bounds.                           |
| `E003` | 422  | Malformed request payload        | No        | Fix request body to match schema.  See `details` array.                |
| `E004` | 404  | Job ID not found                 | No        | Verify job_id.  Jobs may expire after retention period.                 |
| `E005` | 409  | Job not in required state        | Yes       | Poll again — job may still be processing.                               |
| `E006` | —*   | Simulation diverged              | No        | Reduce `n_steps`, adjust parameters, or increase `max_rank`.           |
| `E007` | —*   | Execution timeout                | Yes       | Reduce `n_bits` or `n_steps`, or retry.  Server limit: configurable.   |
| `E008` | —*   | Validation failed                | No        | Result did not pass conservation or stability checks.                   |
| `E009` | 400  | Invalid artifact bundle          | No        | Ensure artifact has required keys (`result`, `domain`, etc.).           |
| `E010` | 429  | Rate limit exceeded              | Yes       | Wait and retry.  `Retry-After` header provides wait time.              |
| `E011` | 401  | Authentication failed            | No        | Provide valid `Authorization: Bearer <key>` header.                     |
| `E012` | 500  | Internal error (opaque)          | Yes       | Retry.  If persistent, contact operator with `job_id` and `request_id`.|

*\*E006–E008 are job-level errors.  They appear in the job's `error` field (via `GET /v1/jobs/{id}`), not as HTTP error responses.  The HTTP response for a failed job is still `200 OK` with `state: "failed"`.*

---

## Error Scenarios by Endpoint

### POST /v1/jobs

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Missing `job_type`                    | E003   | 422  | Pydantic validation error in `details`            |
| Unknown `job_type` value              | E003   | 422  | Enum validation error                             |
| Missing `domain` for execution job    | E001   | 400  | `"domain is required for execution jobs"`         |
| Unknown domain                        | E001   | 400  | `"Unknown domain: 'xyz'. Available: [...]"`       |
| `n_bits` > server max                 | E003   | 400  | `"n_bits=X exceeds limit Y"`                      |
| `n_bits` < 4                          | E003   | 422  | Pydantic ge constraint                            |
| `n_steps` < 1 or > 10000             | E003   | 422  | Pydantic constraint                               |
| Missing auth header                   | E011   | 401  | `"Missing Authorization header"`                  |
| Invalid API key                       | E011   | 401  | `"Invalid API key"`                               |
| Rate limit exceeded                   | E010   | 429  | `"Rate limit exceeded"` + `Retry-After: 60`       |
| Idempotency collision (same key, different payload) | — | 200  | Returns existing job (**not** an error)      |

### GET /v1/jobs/{job_id}

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Valid job_id                          | —      | 200  | Job status                                        |
| Unknown job_id                        | E004   | 404  | `"Job {id} not found"`                            |
| Missing auth                          | E011   | 401  | Auth error                                        |

### GET /v1/jobs/{job_id}/result

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Job completed                         | —      | 200  | Full envelope                                     |
| Job still running                     | E005   | 409  | `"Job is still running"` (retryable=true)         |
| Job queued                            | E005   | 409  | `"Job is still queued"` (retryable=true)          |
| Unknown job_id                        | E004   | 404  | Not found                                         |

### GET /v1/jobs/{job_id}/validation

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Validation available                  | —      | 200  | Validation report                                 |
| Not yet validated                     | E005   | 409  | `"Job not yet validated"` (retryable=true)        |
| Unknown job_id                        | E004   | 404  | Not found                                         |

### GET /v1/jobs/{job_id}/certificate

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Certificate available                 | —      | 200  | Trust certificate                                 |
| Not yet attested                      | E005   | 409  | `"Job not yet attested"` (retryable=true)         |
| Unknown job_id                        | E004   | 404  | Not found                                         |

### POST /v1/validate

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Valid envelope                        | —      | 200  | Validation result                                 |
| Valid raw result                      | —      | 200  | Validation result                                 |
| Unrecognized format                   | E009   | 400  | `"Unrecognized artifact format"`                  |

### GET /v1/capabilities, /v1/contracts, /v1/health

| Scenario                              | Code   | HTTP | Response                                          |
|---------------------------------------|--------|------|---------------------------------------------------|
| Normal                                | —      | 200  | Response payload                                  |
| Contract version not found            | —      | 404  | `"Contract version 'vX' not found"`               |
| Server error                          | E012   | 500  | Opaque internal error                             |

---

## Retry Guidance

| Retryable | Client Behavior                                                       |
|-----------|-----------------------------------------------------------------------|
| `true`    | Retry with exponential backoff.  Honor `Retry-After` if present.     |
| `false`   | Do not retry.  Fix the request or input parameters.                  |

### Recommended Retry Strategy

```
attempt 1: immediate
attempt 2: wait 1s
attempt 3: wait 2s
attempt 4: wait 4s
attempt 5: wait 8s
max attempts: 5
```

---

## Error Code Assignment Rules

- `E0XX` codes are frozen.  No new codes in PATCH versions.
- New codes in MINOR versions must be documented here first.
- Codes are never reused or reassigned.
- Every error response MUST include `code`, `message`, and `retryable`.
