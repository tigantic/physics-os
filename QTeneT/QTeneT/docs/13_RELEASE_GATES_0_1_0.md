# Release Gates — 0.1.0-dev (Phase 1)

**Classification:** Proprietary & Confidential

This document translates the Phase 0 external audit recommendations into **falsifiable gates** for the first functional milestone.

---

## Objective
Ship **QTeneT 0.1.0-dev** as the first version that:
- wires a minimal set of core primitives to real upstream implementations,
- enforces rank-control and never-go-dense invariants,
- introduces meaningful tests (golden + property),
- upgrades CI from “green but meaningless” to “green means something.”

---

## Gate A — Core Wiring (must be real, not stubs)

### A1. Core tensor type
- [ ] `qtenet.core.QTTTensor` exists (even if thin wrapper) and carries:
  - dims
  - ranks
  - dtype
  - layout metadata

### A2. Decomposition
- [ ] `qtenet.core.decomposition.tt_svd(...)` (or equivalent) is wired to a canonical upstream implementation.

### A3. Rounding
- [ ] `qtenet.core.round(...)` (or equivalent) wired + documented.

### A4. Point evaluation
- [ ] `qtenet.core.point_eval(qtt, index)` wired.

**Definition of done:** none of the above raise `NotImplementedError`.

---

## Gate B — Tests (minimum meaningful suite)

### B1. Golden tests (minimum 5)
- [ ] Known analytic function → build QTT → verify point values at fixed indices.
- [ ] Known low-rank tensor → decompose → reconstruct point checks.
- [ ] Shift operator preserves known pattern.

### B2. Property tests (minimum 3)
- [ ] Rounding idempotence: `round(round(x)) ≈ round(x)` within tolerance.
- [ ] Error bound sanity: `||x - round(x)|| <= eps * ||x||` (where measurable).
- [ ] No-dense guard: accidental densification is blocked unless explicitly allowed.

### B3. CI enforcement
- [ ] CI runs tests and fails on regression.

---

## Gate C — CLI Contract (minimum viable)

### C1. `qtenet doctor`
- [ ] Returns 0/3 with accurate environment diagnostics.

### C2. `qtenet inventory`
- [ ] Emits inventory path.

### C3. `qtenet inspect` (scaffold ok)
- [ ] Parses arguments and outputs a JSON stub with correct schema.

---

## Gate D — Taxonomy Triage (reduce “other” risk)

- [ ] Produce a triage report: which “other” artifacts are in-scope for QTeneT vs out-of-scope.
- [ ] Reclassify at least the top 50 high-signal “other” artifacts.

---

## Gate E — Provenance Manifest (first implementation)

- [ ] Define manifest schema (JSON).
- [ ] At least one CLI command emits a manifest artifact.

---

## Gate F — Security Baseline (pre-private distribution)

- [ ] `qtenet reconstruct` requires `--allow-dense` and enforces a hard cap on output size.
- [ ] Confirm no use of `pickle` / `eval` in QTeneT surface.
- [ ] Verify `SECURITY.md` contact is operational.

---

## Exit Criteria
When A–F are satisfied, tag **0.1.0-dev** and treat it as:
- internal demo-ready,
- investor-diligence compatible,
- baseline for Phase 2 wiring.
