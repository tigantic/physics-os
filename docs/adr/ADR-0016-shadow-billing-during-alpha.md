# ADR-0016: Shadow Billing During Alpha

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

HyperTensor-VM's pricing model (`PRICING_MODEL.md`) defines compute-unit metering based on mesh size, solver iterations, QTT rank, and GPU time. Before enforcing billing in production, the metering infrastructure must be validated against real workloads to ensure:

1. Meters are accurate (no drift, no double-counting).
2. Pricing tiers are competitive against ANSYS/COMSOL/OpenFOAM cloud.
3. Edge cases (solver divergence, early termination, cached results) are handled correctly.
4. Customer feedback confirms perceived value aligns with cost.

Launching with untested billing risks overcharging (customer churn) or undercharging (unsustainable economics).

## Decision

**All metering runs in shadow mode during the alpha phase.** Specifically:

1. Every job records full metering telemetry to `artifacts/billing/shadow/`.
2. The metering pipeline computes what *would* be charged but does not enforce payment.
3. Shadow invoices are generated weekly for internal review and select alpha partners.
4. The `METERING_POLICY.md` governs the transition from shadow to live billing.
5. Alpha users see a "Metered (shadow)" badge in the API response headers.
6. Transition to live billing requires: (a) 30 days of shadow data, (b) <2% drift vs. manual audit, (c) sign-off from finance and engineering leads.

## Consequences

- **Easier:** Iterate on pricing without customer-facing billing errors.
- **Easier:** Collect real-world cost distribution data before committing to public pricing.
- **Harder:** Engineering must maintain two code paths (shadow vs. live) until cutover.
- **Risk:** Shadow mode extends indefinitely, delaying revenue. Mitigated by the `LAUNCH_GATE_MATRIX.json` hard gate requiring shadow-to-live transition before GA.
