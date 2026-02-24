# Release Notes — χ-Regularity Evidence Bundle v2.3.1

**Date:** 2026-02-24  
**Commit:** `799102423d88619a0c649cf4ab5c9a34b204269c` (branch `main`)  
**Previous Version:** v2.3 (initial 20-pack campaign results)

---

## Summary

v2.3.1 is a documentation-quality release that converts the evidence bundle
from a working hypothesis note into a claim-addressable, reproducible evidence
release. No measurements were re-run; all 514 data points (352 + 162) are
unchanged from v2.3.

---

## Changes

### Hypothesis Document (`chi_regularity_hypothesis.md`)

1. **Status reclassified.** "EMPIRICALLY CONFIRMED" → "Empirically Supported
   (20-pack pilot)" to reflect pilot scale and avoid over-claiming.

2. **Header split.** Single `Status` field replaced with two fields:
   - `Scientific Status:` conjecture evaluation against acceptance criteria.
   - `Infrastructure Status:` operational pipeline validation (514
     measurements, 0 failures).

3. **Division-of-claims statement** added to abstract block, explicitly
   separating the scientific conjecture from the platform-capability claim.

4. **Pack VI caveat.** q = 2.22 now marked `borderline` under strict
   B-threshold (q > 2.0). Non-divergent behavior noted.

5. **Pack III α-fit.** α ≈ 0.035 (R² = 0.062) reclassified as
   non-diagnostic; V0.2 complexity proxy does not vary, so power-law fit
   is a single-point artifact, not evidence of rank growth.

6. **Grid independence metrics** split into configuration-level (36/44 =
   82 %) and pack-level (16/20 strict, ≥ 18/20 lenient).

7. **Section cross-reference** fixed: Abstract cites § 3.5 (campaign
   results), not the non-existent § 3.6.

8. **Affiliation** changed from "Tigantic Holdings LLC" to "Independent
   Research — HyperTensor Project."

9. **Exact commit hash** added to provenance header.

10. **§ 6.4 — Measurement Validity and Solver-Induced Bias.** New subsection
    documenting the reference-solver limitation for V0.2 packs and its
    bounded impact on rank measurements.

11. **Appendix C — Claim Ledger.** Nine indexed claims (C-001 through
    C-009) with status, evidence pointers, and acceptance criteria.

12. **Immutable provenance block** added to document header.

### Evidence Manifest (`evidence_manifest.json`)

1. **Status vocabulary.** Six allowed status values defined with
   machine-readable descriptions: `supported`, `supported_with_caveat`,
   `partially_supported`, `borderline`, `demonstrated`,
   `assumed_not_verified`.

2. **Missing SHA-256 hashes populated.** `rank_atlas_campaign.py` and
   `chi_regularity_hypothesis.md` now have artifact hashes.

3. **C-007 counting rule.** Explicit algorithmic definition added:
   "Pack passes lenient if majority of its (pack, ξ) configurations pass
   |b|/a < 0.05 and no configuration exhibits monotonic rank divergence
   across all tested n_bits values."

4. **Bundle-level integrity.** `integrity` block added pointing to
   `SHA256SUMS.txt` with verification command.

### New Files

| File                                    | Purpose                                      |
|-----------------------------------------|----------------------------------------------|
| `docs/research/SHA256SUMS.txt`          | All artifact SHA-256 checksums               |
| `docs/research/REPRODUCE.md`           | Step-by-step reproduction instructions       |
| `docs/research/RELEASE_NOTES_v2.3.1.md`| This file                                    |

---

## Verification

```bash
# From repo root
sha256sum -c docs/research/SHA256SUMS.txt
```

---

## What v2.3.1 Does NOT Change

- No campaign measurements re-run.
- No solver code modified.
- No acceptance thresholds adjusted.
- Data files (`rank_atlas_20pack.json`, `rank_atlas_deep_III_VI.json`, etc.)
  are byte-identical to v2.3.
