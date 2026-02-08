# ADR-0002: V-State Gates Not Dates

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-08 |
| **Author** | @core-team |

## Context

Traditional project planning assigns calendar deadlines to milestones. For a platform with 140 solver nodes spanning classical mechanics to quantum information, fixed dates create perverse incentives:

- Teams rush to "check the box" without proper verification.
- Validated benchmarks are skipped in favor of shipping code.
- Maturity claims become aspirational rather than evidence-based.

## Decision

All progress is measured by **version-state gates** (V0.0 through V1.0), not by calendar dates.

- Each V-state has explicit, verifiable exit criteria defined in `Commercial_Execution.md` §2.
- A node cannot advance from state Vx to V(x+1) without meeting all exit criteria for V(x+1).
- Phase transitions (Phase 0→1→2→…→7) are gated by the aggregate V-state of relevant nodes, not by dates.
- The capability ledger records current state; promotions are PR-gated.

There are no fixed deadlines in the execution plan. The roadmap specifies **what must be true** for each phase transition, not **when** it must happen.

## Consequences

- **Easier:** Teams can work at the pace quality demands without artificial pressure.
- **Easier:** External stakeholders get honest maturity assessments.
- **Harder:** Predicting delivery timelines for commercial commitments (mitigated by tier-based prioritization: Tier A nodes are fast-tracked).
- **Harder:** Program management must track gate readiness rather than Gantt charts.
