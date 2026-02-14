# Audit Remediation Framework

## 1. Purpose

This framework governs how audit findings are triaged, assigned, remediated,
and verified. It ensures all CRITICAL and HIGH findings are resolved before
production deployment.

## 2. Severity Classification

| Severity | Definition | Response SLA | Deployment Gate |
|----------|-----------|-------------|-----------------|
| **CRITICAL** | Exploitable soundness break, proof forgery, fund theft | 24h triage, 7-day fix | BLOCKS deployment |
| **HIGH** | Non-trivial vulnerability, DoS, significant information leak | 48h triage, 14-day fix | BLOCKS deployment |
| **MEDIUM** | Limited impact vulnerability, gas inefficiency, edge case | 7-day triage, 30-day fix | Warning, not blocking |
| **LOW** | Code quality, minor inefficiency, style | 14-day triage, next release | Informational |
| **INFO** | Best practice suggestion, documentation | Backlog | Informational |

## 3. Triage Process

### 3.1 Finding Reception

1. Auditor delivers findings via secure channel (encrypted email or portal)
2. Engineering lead assigns a **triage owner** within SLA
3. Triage owner validates finding reproducibility
4. Triage owner assigns severity (may differ from auditor's assessment)
5. Finding entered into tracking system with:
   - Unique ID: `AUDIT-{YEAR}-{SEQ}` (e.g., `AUDIT-2025-001`)
   - Severity
   - Affected component (circuit / contract / infra)
   - Assigned engineer
   - Target resolution date

### 3.2 Disagreement Protocol

If Tigantic disagrees with auditor severity:
1. Document rationale in writing
2. Schedule call with auditor to discuss
3. Final severity decided jointly
4. If unresolved: auditor's classification prevails for deployment gates

## 4. Remediation Workflow

```
Finding Received
       │
       ▼
   ┌────────┐
   │ Triage │────► Severity assigned, engineer assigned
   └────┬───┘
        │
        ▼
   ┌─────────┐
   │  Fix PR  │────► Code change + test case for finding
   └────┬────┘
        │
        ▼
   ┌────────────┐
   │ Code Review │────► Internal review (2 approvers for CRITICAL)
   └────┬───────┘
        │
        ▼
   ┌──────────────┐
   │ Auditor Verify │────► Auditor confirms fix resolves finding
   └────┬─────────┘
        │
        ▼
   ┌────────┐
   │ Closed │
   └────────┘
```

### 4.1 Fix Requirements

Each remediation PR must include:

1. **Reference:** `AUDIT-{YEAR}-{SEQ}` in PR title and commit message
2. **Test Case:** Unit or integration test that would have caught the finding
3. **Regression Gate:** Test added to CI to prevent reintroduction
4. **Documentation:** Update to relevant doc if API/behavior changes
5. **Two Approvals:** Minimum for CRITICAL/HIGH (one from different team)

### 4.2 Fix Verification

For CRITICAL and HIGH findings:
- Auditor must explicitly sign off on remediation
- Sign-off documented in audit tracking system
- Re-verification scope: original finding + related components

## 5. CI Integration

### 5.1 Pre-Merge Gates

All PRs during remediation phase must pass:

```yaml
# .github/workflows/audit-gates.yml
audit_gates:
  checks:
    - cargo test --all-features --workspace
    - cargo clippy --all-features -- -D warnings
    - cargo audit                      # Known CVE check
    - slither contracts/               # Solidity static analysis
    - foundry test                     # Solidity unit tests
    - benchmark regression check       # ≤10% regression threshold
```

### 5.2 Post-Audit Regression Tests

Each finding produces a test case tagged `#[cfg(test)]` with a comment:

```rust
/// Regression test for AUDIT-2025-001: under-constrained witness in Euler3D
/// conservation check. Previously, the prover could construct a valid proof
/// for a state that violated conservation by up to 2^-8.
#[test]
fn test_audit_2025_001_conservation_constraint() {
    // Test that under-conserving state is rejected...
}
```

### 5.3 Continuous Monitoring

Post-deployment:
- Weekly `cargo audit` runs for dependency CVEs
- Monthly re-run of all regression tests in production-equivalent environment
- Quarterly review of audit findings vs current codebase

## 6. Deployment Authorization

### 6.1 Checklist

Before mainnet deployment, ALL of the following must be true:

- [ ] All CRITICAL findings: remediated AND auditor-verified
- [ ] All HIGH findings: remediated AND auditor-verified
- [ ] All MEDIUM findings: remediated OR documented risk acceptance
- [ ] Final audit report received (signed by lead auditor)
- [ ] CI pipeline green on audited commit
- [ ] Benchmark regression check passed
- [ ] VK hash matches audited version
- [ ] Multi-sig deployment keys prepared
- [ ] Monitoring and alerting operational
- [ ] Incident response runbook reviewed

### 6.2 Sign-Off

Deployment requires written sign-off from:
1. Engineering Lead
2. Security Lead (or audit liaison)
3. CEO / CTO

## 7. Risk Acceptance

For findings classified MEDIUM or below that are not remediated:

1. Document the finding and rationale for deferral
2. Assign a risk owner (named individual)
3. Set a review date (maximum 90 days)
4. Log in risk register

Template:

```
Finding: AUDIT-2025-XXX
Severity: MEDIUM
Description: [brief]
Risk Acceptance Rationale: [why not fixing now]
Mitigating Controls: [what reduces risk]
Risk Owner: [name]
Review Date: [YYYY-MM-DD]
```

## 8. Post-Audit Maintenance

- **30 days post-deploy:** Review all LOW/INFO findings for next sprint
- **90 days post-deploy:** Re-engage auditor for delta review if significant changes
- **Annual:** Full re-audit of ZK circuits if constraint system changes
- **On major upgrade:** Mandatory delta audit for changed components
