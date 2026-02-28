# Decision Log

## NS-Millennium Architectural and Strategic Decisions

All significant decisions documented with rationale for future reference.

---

## Decision Format

```
### [DECISION-XXX] Title

**Date:** YYYY-MM-DD  
**Category:** Architecture | Strategy | Methodology | Collaboration  
**Status:** DECIDED | PENDING | REVISITED  

**Context:**  
What situation required a decision?

**Options Considered:**  
1. Option A — pros/cons  
2. Option B — pros/cons  

**Decision:**  
What was decided and why?

**Consequences:**  
What changes as a result?

**Review Trigger:**  
Under what conditions should this be revisited?
```

---

## Decisions

### [DECISION-001] Separate Repository for NS Research

**Date:** 2025-12-22  
**Category:** Strategy  
**Status:** DECIDED  

**Context:**  
The χ-regularity research is exploratory and high-risk. The Physics OS is a production CFD framework with existing proofs and stability guarantees.

**Options Considered:**  
1. Develop in The Physics OS main — keeps code together but mixes stable/experimental
2. Create branch in The Physics OS — still pollutes history
3. Create separate repository — clean separation of concerns

**Decision:**  
Create `tigantic/NS-Millennium` as a fork of The Physics OS. This provides:
- Full The Ontic Engine infrastructure (QTT, proofs, dense guard)
- Clean separation for experimental work
- Independent versioning and releases
- No risk to The Physics OS stability

**Consequences:**  
- Maintain two repositories
- Sync upstream The Ontic Engine improvements as needed
- NS-specific code lives only in NS-Millennium

**Review Trigger:**  
If research produces stable tools useful for general CFD, consider backporting to The Physics OS.

---

### [DECISION-002] No Timeline Commitments

**Date:** 2025-12-22  
**Category:** Strategy  
**Status:** DECIDED  

**Context:**  
Research outcomes are uncertain. Artificial deadlines create pressure to overclaim or cut corners.

**Options Considered:**  
1. Set aggressive timeline — motivating but potentially harmful
2. Set loose timeline — still creates expectations
3. No timeline — milestone-based progress only

**Decision:**  
No calendar-based timelines. Progress measured by milestone achievement and quality of results, not speed.

**Consequences:**  
- Milestones tagged when complete, not when scheduled
- No pressure to publish prematurely
- Must maintain discipline to keep momentum without deadlines

**Review Trigger:**  
If external factors (funding, collaboration) require timeline, revisit with explicit acknowledgment.

---

### [DECISION-003] Success Ladder Framework

**Date:** 2025-12-22  
**Category:** Strategy  
**Status:** DECIDED  

**Context:**  
Millennium Problem research has high failure rate. Need framework to recognize partial successes.

**Options Considered:**  
1. Binary success/failure — demoralizing if prize not achieved
2. Multiple tiers — allows meaningful contribution at every level

**Decision:**  
Adopt 5-level Success Ladder:
- Level 5: Millennium Prize
- Level 4: Major Theorem
- Level 3: Computational Discovery
- Level 2: Tool Contribution
- Level 1: Novel Approach

**Consequences:**  
- All outcomes except complete silence are publishable
- Negative results (χ doesn't predict regularity) is Level 1-2 success
- Team morale protected by recognizing incremental wins

**Review Trigger:**  
If levels prove too coarse or too fine, adjust.

---

### [DECISION-004] Open Science Commitment

**Date:** 2025-12-22  
**Category:** Methodology  
**Status:** DECIDED  

**Context:**  
Credibility in mathematics requires reproducibility. Closed research invites skepticism.

**Options Considered:**  
1. Keep proprietary until publication — protects priority but limits review
2. Fully open from start — maximum transparency, risk of scoop

**Decision:**  
Fully open development:
- All code on GitHub public
- All data archived
- Real-time progress visible

Rationale: The χ-regularity angle is novel enough that scooping is unlikely. Transparency builds trust.

**Consequences:**  
- Anyone can reproduce our results
- Community can identify errors early
- Must maintain clean, documented code at all times

**Review Trigger:**  
If significant competitive pressure emerges, may restrict pre-publication access.

---

### [DECISION-005] Projection Method for Incompressibility

**Date:** 2025-12-22  
**Category:** Architecture  
**Status:** DECIDED  

**Context:**  
The 3D incompressible Navier-Stokes equations require enforcing ∇·u = 0. Three approaches were analyzed:
1. Projection Method (Chorin-Temam) — solve Poisson for pressure, project to div-free
2. Vorticity-Streamfunction — work with ω = ∇×u, Biot-Savart recovery
3. Penalty/Artificial Compressibility — add -λ∇(∇·u) term, approximate

**Options Considered:**  

1. **Penalty Method**
   - Pros: Fast to implement, explicit timestepping, no Poisson solve
   - Cons: Only approximately incompressible (error ~1/λ), artificial dissipation, stiff at high λ
   - CRITICAL: At extreme Re, cannot distinguish χ growth from physical singularity vs numerical artifact
   - Phase 3/4 impact: Ambiguous results, would need to redo with Projection anyway

2. **Vorticity-Streamfunction**
   - Pros: Automatic incompressibility, natural for vortex dynamics
   - Cons: Biot-Savart integral is nonlocal O(N⁶), breaks QTT structure
   - Risk: HIGH — uncharted territory for QTT

3. **Projection Method**
   - Pros: Exact incompressibility, pressure available, matches literature
   - Cons: Requires TT Poisson solver (iterative)
   - Phase 3/4 impact: Clean χ(t) signal, publishable without asterisks

**Decision:**  
Use **Projection Method** for incompressibility enforcement.

Rationale:
- Phase 3 (extreme Re exploration) requires unambiguous χ(t) interpretation
- Penalty would contaminate signal with artificial effects
- We already have TT linear algebra infrastructure (ALS/DMRG)
- Poisson solver is tractable; penalty savings are false economy
- Results directly comparable to literature; publication-ready

**Consequences:**  
- Phase 1 requires working TT Poisson solver
- Add Phase 1a (2D stepping stone) to de-risk Poisson implementation
- Slower initial progress, but cleaner final results
- Pressure field available for validation benchmarks
- Exact mass conservation (to truncation tolerance)

**Review Trigger:**  
If TT Poisson proves intractable after extensive effort, revisit with Penalty as fallback (with documented limitations).

**VERIFICATION (2025-12-22):**  
Decision validated by Phase 1a implementation:
- FFT Poisson achieves exact solve (machine precision)
- Projection reduces divergence by 10¹² (to ~10⁻¹²)
- Taylor-Green benchmark: 0.02% decay error, div < 10⁻¹⁴
- No penalty artifacts — clean χ(t) tracking confirmed
- See: `proofs/proof_phase_1a_result.json`

---

## Pending Decisions

| ID | Topic | Status | Blocking |
|----|-------|--------|----------|
| 006 | Phase 1b: 3D extension strategy | PENDING | Phase 1a complete |

