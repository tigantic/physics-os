# Launch Readiness — HyperTensor Runtime

**Document owner**: Brad / Tigantic Labs
**Baseline**: `v4.0.1` (`7c89bcd0`)
**Branch**: `main`
**Last updated**: 2026-02-27
**Target**: Paid private alpha (3–5 design partners)

> **Re-assessment needed.** This document was last fully audited on 2025-07-24 against `v4.0.0` (`569ff1da`). Since then the codebase has grown from ~1,157K to ~1,989K LOC, tests from 295 to 370+, gauntlet runners from 33 to 38, and attestation JSONs from ~60 to 125+. Gates G6 (Certificate Integrity), G7 (Observability), and G8 (Golden Benchmark Suite) have had significant implementation work. A full gate re-audit against the current baseline is pending.

---

## Gate Status Summary

| #  | Gate                       | Status       | Pass Criteria Met | Blocking |
|----|----------------------------|--------------|-------------------|----------|
| G1 | Scope Baseline             | ✅ PASS       | 9/9               | —        |
| G2 | Contract Stability         | ✅ PASS       | 5/5               | —        |
| G3 | Forbidden Outputs          | 🔶 PARTIAL   | 3/4               | Yes      |
| G4 | Reliability                | 🔶 PARTIAL   | 5/6               | Yes      |
| G5 | Security                   | 🔶 PARTIAL   | 5/7               | Yes      |
| G6 | Certificate Integrity      | 🔴 NOT STARTED| 0/6              | Yes      |
| G7 | Observability              | 🔶 PARTIAL   | 4/5               | Yes      |
| G8 | Golden Benchmark Suite     | 🔴 NOT STARTED| 0/4              | Yes      |
| G9 | Billing Shadow             | 🔶 PARTIAL   | 1/4               | No       |
| G10| Alpha Acceptance           | 🔴 NOT STARTED| 0/5              | Yes      |

**Overall verdict**: NOT READY (32/55 — 58%)

---

## G1 — Scope Baseline

**Purpose**: Confirm the exact system under test.

| Criterion                          | Status | Evidence                           |
|------------------------------------|--------|------------------------------------|
| Commit hash documented             | ✅ PASS | `569ff1da`                         |
| Tag applied                        | ✅ PASS | `v4.0.0`                           |
| API version documented             | ✅ PASS | 1.0.0                              |
| Supported job types enumerated     | ✅ PASS | 6 types in `RELEASE_NOTES`         |
| Supported domains enumerated       | ✅ PASS | 7 domains in registry              |
| Supported outputs enumerated       | ✅ PASS | `API_SURFACE_FREEZE.md`            |
| Release branch created             | ✅ PASS | `release/v4.0.x`                   |
| Versioning policy documented       | ✅ PASS | `VERSIONING_POLICY.md`             |
| API surface freeze documented      | ✅ PASS | `API_SURFACE_FREEZE.md`            |

**Gate**: ✅ PASS

---

## G2 — Contract Stability

**Purpose**: Ensure API schemas are locked and drift is detected.

| #    | Criterion                                          | Status       | Test                                        |
|------|----------------------------------------------------|--------------|---------------------------------------------|
| G2.1 | Request/response schemas frozen and documented     | ✅ PASS       | `API_SURFACE_FREEZE.md` §2–3               |
| G2.2 | Error envelope shape standardized                  | ✅ PASS       | `API_SURFACE_FREEZE.md` §4                 |
| G2.3 | Certificate envelope format frozen                 | ✅ PASS       | `API_SURFACE_FREEZE.md` §5                 |
| G2.4 | OpenAPI spec generated and checked in              | ✅ PASS       | `contracts/v1/openapi.json` (sha256:c18a49…) |
| G2.5 | Contract drift CI check implemented                | ✅ PASS       | `tools/scripts/check_contract_drift.py`            |

### G2 Pass Criteria

- All schemas documented ✓
- OpenAPI spec committed and hashed
- CI test fails on schema change without version bump
- Freeze violation protocol documented ✓

**Gate**: ✅ PASS

### G2 Remediation

| Item | Severity | Action                                           | Status     |
|------|----------|--------------------------------------------------|------------|
| G2.4 | Medium   | Generate OpenAPI JSON from running server, commit | ✅ Done     |
| G2.5 | Medium   | Add schema hash check script                     | ✅ Done     |

---

## G3 — Forbidden Outputs

**Purpose**: No internal mechanism details appear in any response, log,
certificate, or SDK exception.

| #    | Criterion                                          | Status       | Test                                        |
|------|----------------------------------------------------|--------------|---------------------------------------------|
| G3.1 | Forbidden fields registry created                  | ✅ PASS       | `FORBIDDEN_OUTPUTS.md`                      |
| G3.2 | Response sanitizer enforces allowlist              | ✅ PASS       | `sanitizer.py` strips all TT internals      |
| G3.3 | Log redaction verified                             | 🔴 FAIL       | No log redaction test                       |
| G3.4 | Certificate claim allowlist enforced               | ✅ PASS       | Only 3 approved claim tags                  |

### G3 Forbidden Fields List (minimum)

These must NEVER appear in any response, log, certificate, or error:

| Category             | Specific Fields                                          |
|----------------------|----------------------------------------------------------|
| TT internals         | cores, bond_dimensions, chi_max, chi_mean, chi_final     |
| Compression          | compression_ratio, rank_evolution, saturation_rate       |
| Spectra              | singular_values, svd_spectrum                            |
| Scaling              | scaling_class, scaling_classification                    |
| IR/compiler          | opcodes, instruction_count, register_count, ir_ops       |
| Server internals     | file paths, class names, module names, stack traces      |
| Secrets              | api_keys, signing_key, hmac_secret, environment vars     |
| Debug                | debug telemetry, profiling data, memory allocations      |

### G3 Pass Criteria

- Forbidden fields registry exists as a machine-readable list
- Automated test scans all response paths against the registry
- Log output verified clean of secrets and internal paths
- Certificate claims verified against allowlist

**Gate**: 🔶 PARTIAL (1 item remaining)

### G3 Remediation

| Item | Severity | Action                                              | Status     |
|------|----------|-----------------------------------------------------|------------|
| G3.1 | High     | Create `FORBIDDEN_OUTPUTS.md` with machine registry | ✅ Done     |
| G3.3 | Medium   | Add log redaction test                              | 🔴 Pending |

---

## G4 — Reliability

**Purpose**: Deterministic behavior under retries, timeouts,
cancellations, and worker faults.

| #    | Criterion                                          | Status         | Test                              |
|------|----------------------------------------------------|----------------|-----------------------------------|
| G4.1 | Idempotency keys prevent duplicate execution       | ✅ PASS         | Documented in `QUEUE_BEHAVIOR_SPEC.md` §3   |
| G4.2 | Duplicate submit returns existing job              | ✅ PASS         | Documented in `QUEUE_BEHAVIOR_SPEC.md` §3   |
| G4.3 | Timeout produces deterministic error               | ✅ PASS         | E007 + retryable=true, §2.3                 |
| G4.4 | Invalid transitions rejected by state machine      | ✅ PASS         | `InvalidTransition` exception, §1.4         |
| G4.5 | Concurrent burst does not corrupt state            | 🔴 NOT TESTED   | No concurrency test                         |
| G4.6 | Server restart behavior documented                 | ✅ PASS         | `QUEUE_BEHAVIOR_SPEC.md` §5                 |

### G4 Pass Criteria

- Idempotency verified under concurrent duplicate submission
- Timeout and cancellation produce correct error codes
- State machine rejects all invalid transitions
- Burst concurrency does not deadlock or corrupt
- Restart behavior explicitly documented (data loss is acceptable
  if disclosed)

**Gate**: � PARTIAL (1 item remaining)

### G4 Remediation

| Item | Severity | Action                                                    | Status     |
|------|----------|-----------------------------------------------------------|------------|
| G4.5 | High     | Write concurrent burst test (10+ simultaneous submissions)| 🔴 Pending |
| G4.6 | Medium   | Document restart behavior — all in-flight jobs are lost   | ✅ Done     |
| All  | High     | Create `QUEUE_BEHAVIOR_SPEC.md`                           | ✅ Done     |

---

## G5 — Security

**Purpose**: Auth, rate limiting, secret handling, and key rotation
are operational.

| #    | Criterion                                          | Status         | Test                              |
|------|----------------------------------------------------|----------------|-----------------------------------|
| G5.1 | Bearer auth enforced on protected endpoints        | ✅ PASS         | `auth.py`, tested                 |
| G5.2 | Rate limiting enforced (429)                       | ✅ PASS         | Token-bucket, per-key             |
| G5.3 | 401 on missing auth                                | ✅ PASS         | Returns E011                      |
| G5.4 | No secrets in logs                                 | 🔴 NOT TESTED   | Needs log scan                    |
| G5.5 | No private keys committed                          | 🔴 NOT TESTED   | Needs repo scan                   |
| G5.6 | Key rotation documented                            | ✅ PASS         | `SECURITY_OPERATIONS.md` §1.4     |
| G5.7 | Startup checks for missing secrets                 | ✅ PASS         | `SECURITY_OPERATIONS.md` §3       |

### G5 Pass Criteria

- All protected endpoints return 401 without valid key
- Rate limiter returns 429 after burst exhaustion
- `git log --all -p` contains no private keys or secrets
- Key rotation procedure documented
- Startup warns if using ephemeral keys in production

**Gate**: 🔶 PARTIAL (2 items remaining)

### G5 Remediation

| Item | Severity | Action                                                  | Status     |
|------|----------|---------------------------------------------------------|------------|
| G5.4 | High     | Verify no API keys appear in structured logs            | 🔴 Pending |
| G5.5 | Medium   | Run secret scan on full git history                     | 🔴 Pending |
| G5.6 | Medium   | Document key rotation in `SECURITY_OPERATIONS.md`       | ✅ Done     |
| G5.7 | Medium   | Add startup warning for ephemeral signing keys          | ✅ Done     |

---

## G6 — Certificate Integrity

**Purpose**: Certificates are tamper-evident and replay-resistant.

| #    | Criterion                                          | Status         | Test                              |
|------|----------------------------------------------------|----------------|-----------------------------------|
| G6.1 | Claim tampering detected                           | 🔴 NOT TESTED   | Needs adversarial test            |
| G6.2 | Envelope tampering detected                        | 🔴 NOT TESTED   | Needs adversarial test            |
| G6.3 | Signature byte corruption detected                 | 🔴 NOT TESTED   | Needs adversarial test            |
| G6.4 | Wrong verification key rejected                    | 🔴 NOT TESTED   | Needs adversarial test            |
| G6.5 | Replay against different payload detected          | 🔴 NOT TESTED   | Needs adversarial test            |
| G6.6 | Certificate from invalid validation rejected       | 🔴 NOT TESTED   | Needs adversarial test            |

### G6 Pass Criteria

- All 6 adversarial tests pass
- Tampered certificates fail verification deterministically
- Replayed certificates fail hash verification
- Wrong key produces clear rejection (not a crash)

**Gate**: 🔴 NOT STARTED

### G6 Remediation

| Item | Severity | Action                                              |
|------|----------|-----------------------------------------------------|
| All  | Critical | Create `CERTIFICATE_TEST_MATRIX.md` and test suite  |

---

## G7 — Observability

**Purpose**: Every request and job is diagnosable from logs and records.

| #    | Criterion                                          | Status         | Test                              |
|------|----------------------------------------------------|----------------|-----------------------------------|
| G7.1 | Every request has a request_id                     | 🔴 FAIL         | Not implemented                   |
| G7.2 | Every job has a job_id in logs                     | ✅ PASS         | Logger includes job_id            |
| G7.3 | API key identity (suffix) in logs                  | ✅ PASS         | `api_key_suffix` logged           |
| G7.4 | Structured log schema documented                   | ✅ PASS         | `OPERATIONS_RUNBOOK.md` §3        |
| G7.5 | Health and capabilities endpoints documented       | ✅ PASS         | `OPERATIONS_RUNBOOK.md` §5        |

### G7 Pass Criteria

- Request ID attached to every request/response cycle
- Job lifecycle transitions logged with job_id and duration
- Error codes logged on failure
- Structured log format documented
- Operator can trace any request end-to-end

**Gate**: 🔶 PARTIAL (1 item remaining)

### G7 Remediation

| Item | Severity | Action                                               | Status     |
|------|----------|------------------------------------------------------|------------|
| G7.1 | High     | Add request_id middleware to FastAPI app              | 🔴 Pending |
| G7.4 | Medium   | Document log schema in `OPERATIONS_RUNBOOK.md`       | ✅ Done     |
| G7.5 | Low      | Document health/capabilities in runbook              | ✅ Done     |

---

## G8 — Golden Benchmark Suite

**Purpose**: Canonical jobs with known-good results to regression-test
every release.

| #    | Criterion                                          | Status         | Test                              |
|------|----------------------------------------------------|----------------|-----------------------------------|
| G8.1 | At least 1 job per supported domain defined        | 🔴 NOT STARTED  | Need 7 canonical jobs             |
| G8.2 | Expected result tolerances documented              | 🔴 NOT STARTED  | Need tolerance bands              |
| G8.3 | Conservation invariant baselines recorded          | 🔴 NOT STARTED  | Need baseline measurements        |
| G8.4 | Automated regression test runnable                 | 🔴 NOT STARTED  | Need test script                  |

### G8 Pass Criteria

- 7 canonical jobs defined (one per domain)
- Each job has recorded baseline: conservation, wall_time band, field hash
- Regression test runs all 7 jobs and compares against baselines
- Test fails if any result exceeds tolerance band

**Gate**: 🔴 NOT STARTED

### G8 Remediation

| Item | Severity | Action                                                   |
|------|----------|----------------------------------------------------------|
| All  | High     | Run canonical jobs, record baselines, create test script |

---

## G9 — Billing Shadow

**Purpose**: Every job produces a hypothetical charge.  No actual
billing — shadow mode only.

| #    | Criterion                                          | Status         | Test                              |
|------|----------------------------------------------------|----------------|-----------------------------------|
| G9.1 | Billing calculator runs on every job               | 🔴 NOT STARTED  | No calculator exists              |
| G9.2 | Usage ledger records per-job cost fields           | 🔴 NOT STARTED  | No ledger exists                  |
| G9.3 | Pricing model documented                           | ✅ PASS         | `PRICING_MODEL.md` + `METERING_POLICY.md` |
| G9.4 | Invoice export available                           | 🔴 NOT STARTED  | Format documented, no code        |

### G9 Per-Job Fields (required in shadow ledger)

| Field                    | Purpose                              |
|--------------------------|--------------------------------------|
| `job_id`                 | Link to job record                   |
| `job_type`               | Pricing category                     |
| `domain`                 | Complexity class                     |
| `n_bits`                 | Resolution input                     |
| `n_steps`                | Step count input                     |
| `wall_time_s`            | Compute cost proxy                   |
| `grid_points`            | Resource input                       |
| `device_class`           | CPU/GPU tier                         |
| `proof_mode`             | compute | compute+proof              |
| `hypothetical_price_usd` | What would have been billed          |

### G9 Pass Criteria

- Every completed job has a shadow billing record
- Pricing model documented in `PRICING_MODEL.md`
- Metering unit defined in `METERING_POLICY.md`
- CSV export generates invoice-ready records

**Gate**: � PARTIAL (non-blocking for technical alpha,
blocking for paid alpha)

---

## G10 — Alpha Acceptance

**Purpose**: Minimum confidence thresholds before inviting users.

| #     | Criterion                                         | Status         | Threshold                         |
|-------|----------------------------------------------------|----------------|-----------------------------------|
| G10.1 | All blocking gates pass                            | 🔴 FAIL         | G2–G8 all green                   |
| G10.2 | Golden benchmark pass rate                         | 🔴 NOT TESTED   | ≥ 7/7 domains pass                |
| G10.3 | Error rate on valid payloads                       | 🔴 NOT TESTED   | < 1% failure on valid input       |
| G10.4 | Mean job completion time                           | 🔴 NOT TESTED   | < 30s for n_bits ≤ 10             |
| G10.5 | Certificate verification success rate              | 🔴 NOT TESTED   | 100% on server-issued certs       |

### G10 Pass Criteria

- All blocking gates (G1–G8) must pass
- Golden benchmark suite: 7/7 domains succeed
- Valid payload error rate: < 1%
- P95 job time for n_bits ≤ 10: < 30 seconds
- All server-issued certificates verify successfully

**Gate**: 🔴 NOT STARTED

---

## Remediation Priority (ordered by severity)

| Priority | Gate | Item  | Action                                         | Status     |
|----------|------|-------|-------------------------------------------------|------------|
| 1        | G6   | All   | Certificate adversarial test suite              | 🔴 Pending |
| 2        | G4   | G4.5  | Concurrent burst test                           | 🔴 Pending |
| 3        | G3   | G3.1  | Forbidden outputs registry                      | ✅ Done     |
| 4        | G7   | G7.1  | Request ID middleware                           | 🔴 Pending |
| 5        | G8   | All   | Golden benchmark suite                          | 🔴 Pending |
| 6        | G5   | G5.6  | Key rotation documentation                      | ✅ Done     |
| 7        | G5   | G5.7  | Startup secret validation                       | ✅ Done     |
| 8        | G2   | G2.4  | OpenAPI spec generation                         | ✅ Done     |
| 9        | G2   | G2.5  | Contract drift CI check                         | ✅ Done     |
| 10       | G9   | All   | Billing shadow mode                             | 🔶 Partial |

---

## Appendix A — Scope Exclusions

The following are explicitly out of scope during this cycle:

- Public self-serve signup
- Billing automation portal
- Dashboard UI
- Multi-tenant isolation beyond alpha
- New physics domains
- Additional SDK languages
- GPU backend expansion
- Features not required by a launch gate failure

## Appendix B — Decision Log

| Date       | Decision                                              | Rationale                          |
|------------|--------------------------------------------------------|------------------------------------|
| 2026-02-24 | v4.0.0 frozen as baseline                             | All surfaces verified end-to-end   |
| 2026-02-24 | Release branch `release/v4.0.x` created               | Hardening track separated          |
| 2026-02-24 | G9 (Billing) marked non-blocking for technical alpha  | Can test without charging          |
| 2025-07-24 | Phase 3–8 documentation delivered                     | All spec docs created              |
| 2025-07-24 | G2 Contract Stability gate PASSED                     | OpenAPI frozen + drift check       |
| 2025-07-24 | ERROR_CODE_MATRIX.md created                          | 12 error codes, all endpoints      |
| 2025-07-24 | FORBIDDEN_OUTPUTS.md created                          | IP boundary registry               |
| 2025-07-24 | SECURITY_OPERATIONS.md created                        | Key mgmt, rotation, startup checks |
| 2025-07-24 | CERTIFICATE_TEST_MATRIX.md created                    | 12 adversarial test scenarios      |
| 2025-07-24 | CLAIM_REGISTRY.md created                             | 3 claim tags frozen                |
| 2025-07-24 | QUEUE_BEHAVIOR_SPEC.md created                        | State machine + concurrency spec   |
| 2025-07-24 | DETERMINISM_ENVELOPE.md created                       | Variance bands by tier             |
| 2025-07-24 | OPERATIONS_RUNBOOK.md created                         | Log schema, monitoring, ops        |
| 2025-07-24 | PRICING_MODEL.md created                              | 3 tiers, CU definition, shadow     |
| 2025-07-24 | METERING_POLICY.md created                            | Metering unit, capture point       |

## Appendix C — Gate Completion Tracking

```
G1  ████████████████████ 100%  (9/9)
G2  ████████████████████ 100%  (5/5)  ← PASSED
G3  ███████████████░░░░░  75%  (3/4)
G4  ████████████████░░░░  83%  (5/6)
G5  ██████████████░░░░░░  71%  (5/7)
G6  ░░░░░░░░░░░░░░░░░░░░   0%  (0/6)
G7  ████████████████░░░░  80%  (4/5)
G8  ░░░░░░░░░░░░░░░░░░░░   0%  (0/4)
G9  █████░░░░░░░░░░░░░░░  25%  (1/4)
G10 ░░░░░░░░░░░░░░░░░░░░   0%  (0/5)
────────────────────────────────────
Overall: 32/55 criteria met (58%)
```
