# Certificate Test Matrix

**Baseline**: v4.0.0
**Module under test**: `physics_os/core/certificates.py`
**Evidence module**: `physics_os/core/evidence.py`
**Status**: Required for Gate G6 (Certificate Integrity)

---

## Certificate Structure (Reference)

```json
{
  "certificate_version": "1.0.0",
  "job_id": "uuid",
  "issued_at": "ISO 8601",
  "issuer": "hypertensor-runtime",
  "claims": [
    {
      "tag": "CONSERVATION",
      "claim": "energy preserved to 1.23e-08 relative error",
      "witness": { "initial": 1.0, "final": 0.9999999877, "relative_error": 1.23e-08, "threshold": 1e-4 },
      "satisfied": true
    }
  ],
  "input_manifest_hash": "sha256:...",
  "result_hash": "sha256:...",
  "replay_metadata": {
    "runtime_version": "3.1.0",
    "config_hash": "sha256:...",
    "seed": null,
    "device_class": "cpu"
  },
  "signature": "ed25519:abcdef..."
}
```

---

## Test Scenarios

### T1: Happy Path — Round-Trip Sign + Verify

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Valid job result, Ed25519 key initialized     |
| **Action**      | `issue_certificate()` → `verify_certificate()` |
| **Expected**    | `verify_certificate()` returns `True`        |
| **Validates**   | Basic signing pipeline works                 |

### T2: Tampered Claim — Signature Invalidated

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Valid signed certificate                      |
| **Action**      | Modify `claims[0]["satisfied"]` from True to False |
| **Expected**    | `verify_certificate()` returns `False`       |
| **Validates**   | Any claim modification breaks the signature  |

### T3: Tampered Result Hash — Signature Invalidated

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Valid signed certificate                      |
| **Action**      | Replace `result_hash` with a different hash  |
| **Expected**    | `verify_certificate()` returns `False`       |
| **Validates**   | Result integrity is bound to the signature   |

### T4: Tampered Job ID — Signature Invalidated

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Valid signed certificate                      |
| **Action**      | Replace `job_id` with a different UUID       |
| **Expected**    | `verify_certificate()` returns `False`       |
| **Validates**   | Certificates cannot be transplanted between jobs |

### T5: Replay Metadata Tampering — Signature Invalidated

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Valid signed certificate                      |
| **Action**      | Change `replay_metadata.runtime_version`     |
| **Expected**    | `verify_certificate()` returns `False`       |
| **Validates**   | Version binding is part of the signed payload |

### T6: Missing Signature — Verification Fails

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Valid certificate body without `signature` key|
| **Action**      | `verify_certificate()` with no signature     |
| **Expected**    | Returns `False` (empty string signature)     |
| **Validates**   | Unsigned certificates are never accepted     |

### T7: Wrong Key — Cross-Key Verification Fails

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Certificate signed with key A                |
| **Action**      | Attempt `verify_certificate()` with key B loaded |
| **Expected**    | Returns `False`                              |
| **Validates**   | Key binding is enforced                      |

### T8: HMAC Fallback — Round-Trip

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| `cryptography` library NOT available (or forced HMAC mode) |
| **Action**      | `issue_certificate()` → `verify_certificate()` |
| **Expected**    | Returns `True`, signature prefix is `hmac-sha256:` |
| **Validates**   | Fallback signing mode works correctly        |

### T9: Canonical JSON Determinism

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Certificate body with unordered keys         |
| **Action**      | Call `canonical_json()` twice with different key order |
| **Expected**    | Both calls produce identical byte strings    |
| **Validates**   | Signing is order-independent                 |

### T10: Certificate Contains No Forbidden Fields

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Certificate issued from a real execution     |
| **Action**      | Inspect all keys recursively                 |
| **Expected**    | No bond dims, SVD values, TT cores, rank history, IR opcodes |
| **Validates**   | IP boundary holds through the certificate path |

### T11: Claim Tags Are From Allowlist

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Claims generated by `generate_claims()`      |
| **Action**      | Check each `claim.tag` against CLAIM_REGISTRY.md |
| **Expected**    | All tags are in `{CONSERVATION, STABILITY, BOUND}` |
| **Validates**   | No undocumented claim types leak through     |

### T12: Issued-At Timestamp Is UTC

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Precondition**| Certificate issued                           |
| **Action**      | Parse `issued_at`, check timezone            |
| **Expected**    | Ends with `+00:00` or `Z`                   |
| **Validates**   | Temporal consistency across deployments      |

---

## Coverage Matrix

| Test | Sign/Verify | Tamper Detection | IP Boundary | Key Management | Determinism |
|------|-------------|-----------------|-------------|----------------|-------------|
| T1   | ✅          |                 |             |                |             |
| T2   |             | ✅              |             |                |             |
| T3   |             | ✅              |             |                |             |
| T4   |             | ✅              |             |                |             |
| T5   |             | ✅              |             |                |             |
| T6   | ✅          |                 |             |                |             |
| T7   |             |                 |             | ✅             |             |
| T8   | ✅          |                 |             | ✅             |             |
| T9   |             |                 |             |                | ✅          |
| T10  |             |                 | ✅          |                |             |
| T11  |             |                 | ✅          |                |             |
| T12  |             |                 |             |                | ✅          |

---

## Gate G6 Pass Criteria

Gate G6 (Certificate Integrity) passes when:

1. All 12 test scenarios have passing implementations in `tests/`
2. Tamper scenarios (T2–T5) each test at least one field mutation
3. Certificate contains zero forbidden fields (T10)
4. All claim tags are from the registered allowlist (T11)
5. Round-trip works for both Ed25519 and HMAC modes (T1, T8)
6. Canonical JSON is deterministic across Python versions (T9)
