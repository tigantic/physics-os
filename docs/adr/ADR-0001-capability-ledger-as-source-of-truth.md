# ADR-0001: Capability Ledger as Source of Truth

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-08 |
| **Author** | @core-team |

## Context

HyperTensor-VM spans 140 taxonomy nodes across 20 domain packs. Multiple documents track different aspects of the project:

- `PLATFORM_SPECIFICATION.md` tracks LOC inventory and code existence (822K LOC, 140/140 coverage).
- `Commercial_Execution.md` defines the V-state maturity model and execution plan.
- Individual test files, benchmarks, and docs are scattered across the repository.

Without a single authoritative registry, it is impossible to answer: "What is the maturity state of node PHY-II.1?" with a machine-readable, verifiable answer. Teams will make conflicting claims about readiness, and there is no gate enforcement.

## Decision

The **capability ledger** (`ledger/`) is the single source of truth for the maturity state of every taxonomy node. Specifically:

1. Each node has a `ledger/nodes/PHY-{PACK}.{NODE}.yaml` file conforming to `ledger/schema.yaml`.
2. `ledger/index.yaml` is an auto-generated aggregate.
3. The `state` field in each node YAML is the **only** authoritative V-state for that node.
4. V-state promotions require evidence (test results, benchmark outputs) and are gated by CI.
5. `PLATFORM_SPECIFICATION.md` remains the source of truth for inventory/LOC metrics but does **not** confer maturity status.

The ledger is regenerated from the generator script (`ledger/generate_ledger.py`) for the initial bootstrap and then maintained via individual node file edits thereafter.

## Consequences

- **Easier:** Automated dashboards, CI gating, cross-team visibility into maturity.
- **Easier:** External stakeholders can query readiness without reading code.
- **Harder:** Every V-state promotion requires explicit evidence and a PR to the ledger.
- **Risk:** Ledger drift if CI validation is not enforced. Mitigated by the schema validation workflow.
