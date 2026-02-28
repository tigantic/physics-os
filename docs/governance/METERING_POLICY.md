# Metering Policy

**Baseline**: v4.0.0
**Scope**: Private alpha — shadow metering only
**Status**: Active — defines how compute consumption is measured

---

## 1. Metering Unit

### 1.1 Definition

**Compute Unit (CU)**: The fundamental billing unit.

```
CU = wall_time_s × device_multiplier
```

| Device Class | Multiplier | Detection                                |
|--------------|------------|------------------------------------------|
| `cpu`        | 1.0        | `ONTIC_DEVICE != cuda`             |
| `cuda`       | 10.0       | `ONTIC_DEVICE == cuda` or auto-detected |

### 1.2 Measurement Point

CU is measured from the **sanitized result**, not from internal
telemetry.

```
CU source = sanitized_result["performance"]["wall_time_s"]
```

This is the wall-clock time of simulation execution only.  It
excludes:

- Request parsing
- Authentication
- Parameter validation
- Result sanitization
- Validation checks
- Certificate signing
- Response serialization

### 1.3 Precision

- `wall_time_s` is recorded to 4 decimal places
- CU is calculated to 4 decimal places
- Invoices round CU to 2 decimal places per line item
- Monthly totals round to 2 decimal places

---

## 2. Metering Rules

### 2.1 What Counts

| Scenario                                  | Metered?  | CU Recorded |
|-------------------------------------------|-----------|-------------|
| Job completes successfully (any state)    | Yes       | Based on actual wall time |
| `full_pipeline` → attested                | Yes       | Single CU record |
| `physics_vm_execution` → succeeded        | Yes       | Based on actual wall time |
| `validation` job                          | No        | 0 CU |
| `attestation` job                         | No        | 0 CU |

### 2.2 What Does NOT Count

| Scenario                                  | Metered?  | Reason                    |
|-------------------------------------------|-----------|---------------------------|
| Job fails during execution                | No        | E006, E007 failures       |
| Job fails during validation               | No        | E008 failure              |
| Idempotent replay (same key)             | No        | No new computation        |
| Read endpoints (GET /jobs/*)             | No        | No compute                |
| Health/capabilities/contracts checks      | No        | No compute                |
| Internal server error (E012)              | No        | Server bug                |
| Rate-limited request (E010)               | No        | Request rejected          |
| Auth failure (E011)                       | No        | Request rejected          |

### 2.3 Edge Cases

| Scenario                                  | Metered?  | Reason                    |
|-------------------------------------------|-----------|---------------------------|
| Very short job (< 0.01s)                  | Yes       | Minimum 0.01 CU           |
| Very long job (> 300s, timed out)         | No        | Failed (E007)             |
| Job succeeds but certificate fails        | Yes       | Compute was consumed      |
| Partial result (fields too large)         | Yes       | Compute was consumed      |

---

## 3. Metering Capture Point

### 3.1 Where CU Is Recorded

CU is captured in the job pipeline AFTER sanitization, at the point
where `performance.wall_time_s` is available:

```python
# Pseudocode — actual location in API router
sanitized = sanitize_result(execution_result, domain_key)
wall_time = sanitized["performance"]["wall_time_s"]
device_multiplier = 10.0 if device_class == "cuda" else 1.0
cu = wall_time * device_multiplier

# Record in metering log
meter_record = {
    "job_id": job.job_id,
    "api_key_suffix": job.api_key_suffix,
    "domain": domain_key,
    "device_class": device_class,
    "wall_time_s": wall_time,
    "compute_units": round(cu, 4),
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
```

### 3.2 Storage (Alpha)

During alpha, metering records are:

- Stored in-memory alongside the job
- Derivable from `job.result["performance"]["wall_time_s"]`
- Not persisted across restarts
- Not exported automatically

### 3.3 Storage (Post-Alpha)

- Dedicated metering table/collection
- Append-only (never modified)
- Daily aggregation for invoice generation
- 90-day retention minimum

---

## 4. API Key Association

### 4.1 Key → Usage Mapping

Every job records `api_key_suffix` (last 8 characters of the API key).
This associates compute consumption with a billing identity without
storing the full key.

### 4.2 Suffix Collision

If two API keys share the same 8-character suffix (unlikely with
`token_urlsafe(32)`), metering records may be indistinguishable.

**Alpha mitigation**: Operator ensures unique suffixes.
**Post-alpha mitigation**: Use full key hash for billing identity.

---

## 5. Invoice Generation

### 5.1 Shadow Invoice (Alpha)

Shadow invoices are NOT generated automatically in v4.0.0.
They can be produced manually by aggregating job results:

```bash
# Example: aggregate CU for a specific API key suffix
python3 -c "
import json, sys
# Parse job results from server or exported data
# Sum wall_time_s × device_multiplier for each successful job
# Output invoice JSON
"
```

### 5.2 Invoice Line Items

Each line item represents one metered job:

| Field            | Type    | Source                                 |
|------------------|---------|----------------------------------------|
| `job_id`         | string  | Job UUID                               |
| `date`           | string  | `job.created_at` (date portion)        |
| `domain`         | string  | Job input domain                       |
| `device_class`   | string  | `cpu` or `cuda`                        |
| `wall_time_s`    | float   | `result.performance.wall_time_s`       |
| `compute_units`  | float   | `wall_time_s × device_multiplier`      |
| `unit_price_usd` | float   | From PRICING_MODEL.md package rates    |

### 5.3 Monthly Summary

| Field            | Type    | Calculation                            |
|------------------|---------|----------------------------------------|
| `total_cu`       | float   | Sum of all line item CU                |
| `included_cu`    | float   | From package (Explorer = 100)          |
| `overage_cu`     | float   | `max(0, total_cu - included_cu)`       |
| `overage_usd`    | float   | `overage_cu × overage_rate`            |
| `total_usd`      | float   | `base_price + overage_usd`             |

---

## 6. Metering Integrity

### 6.1 Tamper Resistance

CU is derived from `wall_time_s`, which is part of the sanitized
result, which is hashed into the certificate's `result_hash`.

Certificate chain:

```
wall_time_s → sanitized_result → content_hash(result) → certificate.result_hash → signature
```

If an operator inflates `wall_time_s`, the `result_hash` changes,
which invalidates the certificate signature.  This provides
cryptographic binding between metering and compute proofs.

### 6.2 Client Verification

Clients can verify their metering by:

1. Downloading the certificate for each job
2. Verifying the signature
3. Re-computing CU from the certified `wall_time_s`
4. Comparing against the invoice

### 6.3 Dispute Resolution

If a client disputes a metering record:

1. Operator provides the certificate for the disputed job
2. Client verifies the certificate signature
3. Client extracts `wall_time_s` from the certified result
4. CU is recomputed from the certified value
5. If the certificate is valid and CU matches, the record stands
6. If the certificate fails verification, the charge is reversed

---

## 7. Quota Enforcement (Future)

Not implemented in v4.0.0.  Planned for post-alpha:

| Check                    | Action                               |
|--------------------------|--------------------------------------|
| CU budget exhausted      | Reject job with quota error          |
| CU budget at 80%         | Warn in response header              |
| CU budget at 100%        | Block + notify                       |
| Overage allowed          | Continue + bill overage rate         |
