# Determinism Envelope

**Baseline**: v4.0.0
**Status**: Active specification — defines acceptable variance for "same result"

---

## 1. Purpose

The Ontic Engine produces physics simulation results using Quantized Tensor
Train (QTT) decomposition.  Because the system runs on floating-point
hardware, exact bitwise reproducibility is not guaranteed.  This
document defines the **acceptable variance bands** within which two
executions of the same input are considered "the same result."

---

## 2. Determinism Tiers

### Tier 1: Bitwise Deterministic

| Component               | Guarantee                                            |
|--------------------------|------------------------------------------------------|
| `canonical_json()`       | Same input dict → identical bytes every call         |
| `content_hash()`         | Same bytes → identical SHA-256 hash                 |
| Certificate signature    | Same key + same payload → identical signature       |
| Idempotency key lookup   | Same key → same job ID                              |
| State machine            | Same transition sequence → same final state         |
| Sanitizer field names    | Same domain → same field keys and structure         |

### Tier 2: Numerically Reproducible (Same Platform)

| Component                | Guarantee                                            |
|--------------------------|------------------------------------------------------|
| QTT compilation          | Same compiler + same parameters → same IR           |
| QTT execution (CPU)      | Same IR + same seed → bitwise identical dense output |
| Conservation calculation  | Same field values → identical relative error        |
| Validation checks         | Same result → identical validation report           |

**Conditions**: Same hardware, same OS, same Python version, same
`numpy`/`scipy` versions, same thread count.

### Tier 3: Physically Equivalent (Cross-Platform)

| Component                | Guarantee                                            |
|--------------------------|------------------------------------------------------|
| QTT execution (GPU)      | Dense output within floating-point tolerance         |
| Cross-platform execution  | Dense output within floating-point tolerance        |
| Cross-version execution   | Dense output within documented tolerance            |

---

## 3. Acceptable Variance Bands

### 3.1 Field Values

For reconstructed dense field values (`fields.{name}.values`):

| Comparison              | Tolerance                 | Metric                         |
|--------------------------|---------------------------|---------------------------------|
| Same platform, same seed | 0 (bitwise identical)    | `max(|a - b|) == 0`           |
| Same platform, no seed   | `≤ 1e-12`               | `max(|a - b|) / max(|a|, 1)`  |
| Cross-platform (CPU)     | `≤ 1e-10`               | `max(|a - b|) / max(|a|, 1)`  |
| CPU vs GPU               | `≤ 1e-6`                | `max(|a - b|) / max(|a|, 1)`  |

### 3.2 Conservation Metrics

For `conservation.relative_error`:

| Comparison              | Tolerance                                                |
|--------------------------|----------------------------------------------------------|
| Same platform           | Identical (computed from same field values)              |
| Cross-platform          | Difference ≤ `1e-10` (propagated from field tolerance)  |

### 3.3 Performance Metrics

For `performance.wall_time_s` and `performance.throughput_pts_per_s`:

| Comparison              | Tolerance                                                |
|--------------------------|----------------------------------------------------------|
| Same hardware, same load | Within 20% of each other                               |
| Different hardware       | Not comparable (informational only)                     |

**Note**: Performance metrics are NOT part of the determinism guarantee.
They are informational and MUST NOT appear in certificate claims.

### 3.4 Content Hashes

For `input_manifest_hash`, `result_hash`, `config_hash`:

| Comparison              | Guarantee                                                |
|--------------------------|----------------------------------------------------------|
| Same input parameters   | Identical `input_manifest_hash`                         |
| Same field values (bitwise) | Identical `result_hash`                             |
| Different field values   | Different `result_hash` (collision-resistant)           |

---

## 4. Certificate Determinism

### 4.1 Same Input → Same Claims?

Given identical input parameters and identical execution results:

- **CONSERVATION claim**: Same `relative_error`, same `satisfied` — deterministic
- **STABILITY claim**: Same `wall_time_s` — deterministic on values, timing varies
- **BOUND claim**: Same `max_absolute_value` — deterministic

### 4.2 Same Input → Same Signature?

**No.**  Certificates include `issued_at` (timestamp) and `job_id` (UUID),
both of which change between executions.  Therefore:

- Same input → different `certificate_body` → different signature
- This is by design — certificates are one-time attestations, not cache keys

### 4.3 Signature Verification Is Deterministic

Given the same certificate JSON and the same public key:

- `verify_certificate()` always returns the same boolean
- This is the only determinism guarantee needed for trust verification

---

## 5. Replay Specification

### 5.1 Replay Metadata

Every certificate includes:

```json
{
  "replay_metadata": {
    "runtime_version": "3.1.0",
    "config_hash": "sha256:...",
    "seed": null,
    "device_class": "cpu"
  }
}
```

This metadata is sufficient to determine whether a replay attempt
should produce a matching result.

### 5.2 Replay Compatibility

| Field                   | Same → Replay Expected Identical? |
|-------------------------|-----------------------------------|
| `runtime_version`       | Yes (same code path)              |
| `config_hash`           | Yes (same parameter set)          |
| `seed`                  | Yes if both set, No if null       |
| `device_class`          | Within tolerance (see 3.1)        |

### 5.3 Replay Not Guaranteed When

- Runtime version differs (code changes may alter results)
- Config hash differs (different parameters)
- Seed is `null` (non-deterministic initialization)
- Device class differs by more than one tier

---

## 6. Testing Requirements

### 6.1 Tier 1 Tests (Bitwise)

- `canonical_json()` stability: serialize, deserialize, re-serialize → byte-identical
- `content_hash()` consistency: hash same content 1000 times → all identical
- Idempotency: same key → same job ID every time

### 6.2 Tier 2 Tests (Reproducibility)

- Execute Burgers domain twice with same parameters → compare dense arrays
- Expected: `max(|a - b|) == 0` on same platform
- Execute all 7 domains → verify conservation relative_error is identical across runs

### 6.3 Tier 3 Tests (Cross-Platform)

- Deferred to post-alpha (requires multi-platform CI)
- Documented tolerance bands apply

---

## 7. What Is NOT Deterministic

These values change between executions and MUST NOT be used for
equality comparisons:

| Field                    | Why It Changes                     |
|--------------------------|-------------------------------------|
| `job_id`                 | UUID generated per submission       |
| `created_at`             | Timestamp per submission            |
| `completed_at`           | Timestamp per completion            |
| `issued_at` (cert)       | Timestamp per attestation           |
| `signature` (cert)       | Derived from body including timestamps |
| `wall_time_s`            | System load dependent               |
| `throughput_pts_per_s`   | Derived from wall_time_s            |
