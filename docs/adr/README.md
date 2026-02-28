# Architecture Decision Records (ADR)

This directory contains platform-wide Architecture Decision Records for The Physics OS-VM.

## What is an ADR?

An Architecture Decision Record captures an architecturally significant decision along with its context and consequences. ADRs are numbered sequentially and are **immutable once accepted** — if a decision is superseded, a new ADR is created that references the old one.

## Format

Each ADR follows the naming convention: `YYYYMMDD-NNNN-short-title.md`

### Template

```markdown
# ADR-NNNN: Title

| Field | Value |
|-------|-------|
| **Status** | proposed / accepted / deprecated / superseded by ADR-XXXX |
| **Date** | YYYY-MM-DD |
| **Author** | @handle |
| **Supersedes** | (if applicable) |

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?
```

## Scope

These ADRs cover **platform-level** decisions that affect:

- Core substrate APIs and data model
- Plugin architecture and domain pack interfaces
- V&V harness design
- CI/CD pipeline architecture
- QTT/TN acceleration strategy
- Packaging, versioning, and release policy
- Cross-pack interface contracts

Pack-specific decisions should be documented in the pack's own `docs/` directory unless they have platform-wide implications.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](ADR-0001-capability-ledger-as-source-of-truth.md) | Capability Ledger as Source of Truth | accepted | 2026-02-08 |
| [0002](ADR-0002-v-state-gates-not-dates.md) | V-State Gates Not Dates | accepted | 2026-02-08 |
| [0003](ADR-0003-domain-packs-as-plugins.md) | Domain Packs as Plugins | accepted | 2026-02-08 |
| [0004](ADR-0004-canonical-platform-protocols.md) | Canonical Platform Protocols | accepted | 2026-02-08 |
| [0005](ADR-0005-vertical-slices-as-exit-gates.md) | Vertical Slices as Phase Exit Gates | accepted | 2026-02-08 |
| [0006](ADR-0006-vv-harness-architecture.md) | V&V Harness Architecture | accepted | 2026-02-08 |
| [0007](ADR-0007-domain-pack-framework-and-anchors.md) | Domain Pack Framework and Anchors | accepted | 2026-02-08 |
| [0008](ADR-0008-phase4-full-taxonomy-coverage.md) | Phase 4: Full Taxonomy Coverage | accepted | 2026-02-08 |
| [0009](ADR-0009-phase5-qtt-acceleration.md) | Phase 5: QTT Acceleration | accepted | 2026-02-08 |
| [0010](ADR-0010-phase6-coupled-inverse-uq-optimization.md) | Phase 6: Coupled Inverse UQ Optimization | accepted | 2026-02-08 |
| [0011](ADR-0011-phase7-productization.md) | Phase 7: Productization | accepted | 2026-02-08 |
| [0012](ADR-0012-never-go-dense.md) | Never Go Dense — All Operations in TT/QTT | accepted | 2026-02-25 |
| [0013](ADR-0013-ed25519-trust-certificates.md) | Ed25519 for Trust Certificates | accepted | 2026-02-25 |
| [0014](ADR-0014-synchronous-job-pipeline.md) | Synchronous Job Pipeline (Not Async Queue) | accepted | 2026-02-25 |
| [0015](ADR-0015-whitelist-only-ip-sanitization.md) | Whitelist-Only IP Sanitization | accepted | 2026-02-25 |
| [0016](ADR-0016-shadow-billing-during-alpha.md) | Shadow Billing During Alpha | accepted | 2026-02-25 |
| [0017](ADR-0017-q16-16-fixed-point-zk.md) | Q16.16 Fixed-Point for ZK Arithmetic | accepted | 2026-02-25 |
| [0018](ADR-0018-lean4-formal-proofs.md) | Lean 4 for Formal Proofs | accepted | 2026-02-25 |
| [0019](ADR-0019-halo2-zk-circuits.md) | Halo2 for ZK Circuits | accepted | 2026-02-25 |
| [0020](ADR-0020-in-memory-job-store-v1.md) | In-Memory Job Store for v1 | accepted | 2026-02-25 |
| [0021](ADR-0021-python-monolith-rust-crates.md) | Python Monolith + Rust Crates Architecture | accepted | 2026-02-25 |
| [0022](ADR-0022-mcp-server-ai-agents.md) | MCP Server for AI Agent Integration | accepted | 2026-02-25 |
| [0023](ADR-0023-domain-pack-taxonomy.md) | Domain Pack Taxonomy (168 Nodes, 20 Verticals) | accepted | 2026-02-25 |
