<div align="center">

# The Physics OS — Roadmap

**Release v4.0.1** · **February 2026** · **Powered by The Ontic Engine**

</div>

---

## Current Release — v4.0.1 (Hardened Baseline)

The v4.0.1 release is the **infrastructure-hardened baseline** for the paid private alpha. No new features, domains, or surface area are added during the v4.0.x hardening cycle.

### Completed in v4.0.x

| Deliverable | Status |
|-------------|:------:|
| 35-finding audit remediation | ✅ |
| 11 CI/CD workflows (including OIDC trusted publishing) | ✅ |
| Observability stack (Prometheus + Grafana + 8 alert rules) | ✅ |
| Dependabot (3 ecosystems: pip, cargo, GitHub Actions) | ✅ |
| CODEOWNERS (278 domain-expert review mappings) | ✅ |
| Version integrity enforcement (7-checkpoint sync) | ✅ |
| Pre-commit hooks (ruff, secrets, conventional commits) | ✅ |
| PEP 561 type stubs shipped | ✅ |
| Platform Specification (2,052 lines, fully validated) | ✅ |
| Log security hardening (CI-enforced) | ✅ |

---

## Milestones

### M1 — Private Alpha (Q1 2026) `← CURRENT`

| Goal | Status |
|------|:------:|
| Runtime Access Layer (REST API, SDK, CLI, MCP) | ✅ |
| Ed25519 trust certificates | ✅ |
| 7 physics domains compiled to QTT IR | ✅ |
| CU metering pipeline | ✅ |
| IP sanitization boundary (25 forbidden categories) | ✅ |
| Documentation site (MkDocs Material) | ✅ |
| Hardened CI/CD (11 workflows) | ✅ |

### M2 — Closed Beta (Q2 2026)

| Goal | Status |
|------|:------:|
| Multi-tenant job isolation | 🔲 |
| Persistent job store (PostgreSQL + WAL) | 🔲 |
| Horizontal prover pool scaling | 🔲 |
| Rate limiting per API key tier | 🔲 |
| Dashboard for job monitoring | 🔲 |
| Performance regression CI gate | 🔲 |

### M3 — Public Beta (Q3 2026)

| Goal | Status |
|------|:------:|
| Self-service API key provisioning | 🔲 |
| Webhook notifications on job completion | 🔲 |
| Batch job submission API | 🔲 |
| GPU cloud deployment (NVIDIA A100/H100) | 🔲 |
| Additional physics domains (10+) | 🔲 |
| SLA monitoring and reporting | 🔲 |

### M4 — General Availability (Q4 2026)

| Goal | Status |
|------|:------:|
| Production SLA (99.9% uptime) | 🔲 |
| Enterprise SSO (SAML/OIDC) | 🔲 |
| On-premises deployment option | 🔲 |
| Marketplace integrations | 🔲 |
| Certified compliance documentation | 🔲 |

---

## Research Frontiers

These are active research threads, not committed deliverables:

| Frontier | Description |
|----------|-------------|
| **Exascale QTT** | Push beyond 16,384³ (4.4 × 10¹² DOF) — target 10¹⁵ |
| **QTT-Native GPU** | Full TT-core operations on CUDA without host roundtrips |
| **Gevulot Mainnet** | Decentralized ZK proof verification on Gevulot network |
| **Cross-Primitive Fusion** | OT → SGW → RKHS → PH → GA end-to-end pipeline optimization |
| **Lean 4 Automation** | Computer-generated formal proofs for new domain compilers |

---

## Detailed Strategic Roadmap

For the comprehensive strategic roadmap with honest status assessment and validation evidence:

**[`docs/roadmaps/ROADMAP.md`](docs/roadmaps/ROADMAP.md)** — 632 lines, validated milestones

See also: **[`docs/audits/ROADMAP_AUDIT.md`](docs/audits/ROADMAP_AUDIT.md)** — integrity verification of all roadmap claims
