# Architecture Decision Records (ADR)

This directory contains platform-wide Architecture Decision Records for HyperTensor-VM.

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
| [0001](20260208-0001-capability-ledger-as-source-of-truth.md) | Capability Ledger as Source of Truth | accepted | 2026-02-08 |
| [0002](20260208-0002-v-state-gates-not-dates.md) | V-State Gates Not Dates | accepted | 2026-02-08 |
| [0003](20260208-0003-domain-packs-as-plugins.md) | Domain Packs as Plugins | accepted | 2026-02-08 |
| [0004](20260208-0004-canonical-platform-protocols.md) | Canonical Platform Protocols | accepted | 2026-02-08 |
| [0005](20260208-0005-vertical-slices-as-exit-gates.md) | Vertical Slices as Phase Exit Gates | accepted | 2026-02-08 |
