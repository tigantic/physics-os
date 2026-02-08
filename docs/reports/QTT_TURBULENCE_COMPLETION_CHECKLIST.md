# QTT Turbulence Solver: Completion Checklist

**Authority:** Principal Investigator  
**Prerequisite:** Phases 1-6 COMPLETE (Workflow_Development.md)  
**Objective:** Defensible, publishable claim — QTT turbulence with O(log N) scaling

---

## STEP 1: SPECTRAL POISSON SOLVER

**Why:** Jacobi diverges. `poisson_iterations=0` is a workaround. 5.8% dissipation at 128³ traces directly to skipping velocity reconstruction. This is the blocker.

- [ ] 1.1 — Review existing FFT infrastructure in codebase (`tensornet/` search for fft, spectral, fourier)
- [ ] 1.2 — Design QTT-compressed spectral Poisson solver (`poisson_spectral.py`)
  - Solve ∇²ψ = f in Fourier space: ψ̂(k) = f̂(k) / |k|²
  - QTT-native FFT or hybrid FFT approach
- [ ] 1.3 — Implement `poisson_spectral.py` with full error handling and type hints
- [ ] 1.4 — Validate against analytical Poisson solutions (known RHS → known solution)
  - Gate: error < 1e-6 on 32³, 64³, 128³
- [ ] 1.5 — Integrate into `ns3d_turbo.py` replacing `poisson_iterations=0` path
- [ ] 1.6 — Re-run Phase 5 dissipation analysis with spectral Poisson active
  - Gate: dissipation < 1% at 128³ (down from 5.8%)
- [ ] 1.7 — Re-run `prove_qtt_turbulence.py` — all 5 proofs must still pass
- [ ] 1.8 — Generate attestation: `STEP1_SPECTRAL_POISSON_ATTESTATION.json`
- [ ] 1.9 — Update `Workflow_Development.md` with results
- [ ] 1.10 — Git commit: `fix(poisson): spectral solver replaces Jacobi`

**HOLD — Do not proceed to Step 2 until 1.6 gate is met.**

If dissipation does NOT drop below 1%, the remaining source is advection truncation → proceed to Step 2 regardless but flag the gap.

---

## STEP 2: ADVECTION TRUNCATION CONTROL

**Why:** Hadamard products in advection term lose high-frequency energy on truncation. Second source of numerical dissipation after Poisson.

- [ ] 2.1 — Profile energy loss per operation in `_compute_rhs()`
  - Isolate: how much energy does each Hadamard + truncate lose?
  - Isolate: how much does curl truncation lose?
- [ ] 2.2 — Implement component-specific truncation tolerances
  - Tighter tolerance in advection path (preserve energy)
  - Standard tolerance elsewhere (preserve speed)
- [ ] 2.3 — Alternative: selective rank elevation in advection
  - Run advection at rank 24, everything else at rank 16
  - Measure memory impact — must still fit 256³ in 8GB
- [ ] 2.4 — Evaluate: conservative truncation scheme (energy-preserving by construction)
  - Symmetric truncation that guarantees ‖u_truncated‖ = ‖u_original‖
- [ ] 2.5 — Select best approach from 2.2/2.3/2.4 based on dissipation reduction vs cost
- [ ] 2.6 — Implement selected approach
- [ ] 2.7 — Re-run Phase 5 dissipation analysis
  - Gate: improvement over Step 1 result at 128³
- [ ] 2.8 — Re-run `prove_qtt_turbulence.py` — all 5 proofs must still pass
- [ ] 2.9 — Re-run memory validation — 256³ must still fit in 8GB
- [ ] 2.10 — Generate attestation: `STEP2_ADVECTION_TRUNCATION_ATTESTATION.json`
- [ ] 2.11 — Git commit: `opt(advection): truncation control for energy preservation`

---

## STEP 3: DECAYING HOMOGENEOUS ISOTROPIC TURBULENCE (DHIT)

**Why:** Taylor-Green is a symmetric, well-behaved test case. DHIT is the standard benchmark the field uses. Reviewers expect it.

- [ ] 3.1 — Implement DHIT initialization
  - von Kármán-Pao energy spectrum: E(k) = A·k⁴·exp(-2(k/k_p)²)
  - Random phases with prescribed spectrum
  - Divergence-free projection
- [ ] 3.2 — Implement energy spectrum measurement
  - Shell-averaged E(k) computation from QTT fields
  - Time-series output at configurable intervals
- [ ] 3.3 — Run DHIT at 64³, Re_λ ~ 50-100
  - Evolve through initial transient to developed state
  - Record energy spectrum at multiple time snapshots
- [ ] 3.4 — Validate K41 scaling
  - Gate: E(k) ~ k^(-5/3) slope in inertial range within 20%
  - Compensated spectrum plot: k^(5/3)·E(k) should be flat
- [ ] 3.5 — Measure dissipation rate ε = -dE/dt
  - Compare against ε = 2ν·Ω (enstrophy-based)
  - Gate: two estimates agree within 10%
- [ ] 3.6 — Run DHIT at 128³ — same validation
- [ ] 3.7 — Create `dhit_benchmark.py` with full attestation
- [ ] 3.8 — Generate spectrum plots (save as PNG + data as JSON)
- [ ] 3.9 — Generate attestation: `STEP3_DHIT_ATTESTATION.json`
- [ ] 3.10 — Git commit: `feat(dhit): decaying homogeneous isotropic turbulence benchmark`

---

## STEP 4: HEAD-TO-HEAD DNS COMPARISON

**Why:** The claim requires comparison against established methods. Without this, it's a solver talking to itself.

- [ ] 4.1 — Select reference DNS code
  - Option A: hit3d (open-source pseudospectral, well-documented)
  - Option B: Dedalus (Python-based spectral, accessible)
  - Option C: Published DHIT statistics from literature (Ishihara et al., Kaneda et al.)
  - Select based on accessibility and reproducibility
- [ ] 4.2 — Run identical DHIT case in reference code
  - Same initial conditions (or statistically equivalent)
  - Same Reynolds number, same grid resolution equivalent
- [ ] 4.3 — Compare energy spectra at matched time snapshots
  - Overlay plots: QTT vs reference
  - Quantify deviation in inertial range
- [ ] 4.4 — Compare dissipation rate evolution
  - ε(t) curves: QTT vs reference
- [ ] 4.5 — Compare velocity PDFs and structure functions (if accessible)
- [ ] 4.6 — Create comparison table

  | Metric | Reference DNS | QTT Turbo | Deviation |
  |--------|--------------|-----------|-----------|
  | K41 slope | | | |
  | Dissipation rate | | | |
  | Memory | | | |
  | Time/step | | | |
  | Scaling | O(N³) or O(N log N) | O(log N) | |

- [ ] 4.7 — Honest assessment: where does QTT match, where does it deviate, why?
- [ ] 4.8 — Generate attestation: `STEP4_DNS_COMPARISON_ATTESTATION.json`
- [ ] 4.9 — Git commit: `val(dns): head-to-head comparison with reference DNS`

---

## STEP 5: REYNOLDS NUMBER SWEEP

**Why:** χ ~ Re^0.035 is THE thesis. Prove it or disprove it across a range. Either result is publishable.

- [ ] 5.1 — Design sweep: Re_λ = {50, 100, 200, 500, 1000}
  - Adjust ν to control Reynolds at fixed grid
  - Or adjust grid with fixed ν — document choice and rationale
- [ ] 5.2 — For each Re: run DHIT, measure peak bond dimension χ
  - Record χ at steady state (after transient)
  - Record χ evolution over time
- [ ] 5.3 — Fit χ vs Re
  - Log-log plot
  - Compute exponent: χ ~ Re^α
  - Gate: α < 0.1 supports thesis (near-constant χ)
  - If α > 0.5: thesis fails at high Re — document the crossover
- [ ] 5.4 — Record memory and timing at each Re
  - Does O(log N) hold across Reynolds range?
- [ ] 5.5 — Create χ vs Re plot (THE figure)
- [ ] 5.6 — If thesis holds: document with confidence
- [ ] 5.7 — If thesis breaks: identify crossover Re and characterize failure mode
- [ ] 5.8 — Generate attestation: `STEP5_REYNOLDS_SWEEP_ATTESTATION.json`
- [ ] 5.9 — Git commit: `sci(reynolds): bond dimension scaling with Reynolds number`

---

## STEP 6: ARXIV PAPER

**Why:** The receipts need a frame. The paper is the frame.

### 6A: Structure

- [ ] 6A.1 — Title and abstract (last — write after everything else)
- [ ] 6A.2 — Introduction: QTT for turbulence, why it should work, what we prove
- [ ] 6A.3 — Method: QTT representation, vorticity formulation, O(log N) operations
- [ ] 6A.4 — Results: Taylor-Green, DHIT, DNS comparison, Reynolds sweep
- [ ] 6A.5 — Discussion: limitations (dissipation, Poisson approximation), future work
- [ ] 6A.6 — Conclusion: what's proven, what's not, what it means

### 6B: Figures

- [ ] 6B.1 — O(log N) scaling plot (from prove_qtt_turbulence.py)
- [ ] 6B.2 — Compression ratio vs grid size
- [ ] 6B.3 — Energy spectrum: QTT vs reference DNS
- [ ] 6B.4 — χ vs Re plot (THE figure)
- [ ] 6B.5 — Dissipation rate comparison
- [ ] 6B.6 — Memory comparison: QTT vs dense at scale

### 6C: Supplementary

- [ ] 6C.1 — All attestation JSON hashes
- [ ] 6C.2 — Git commit history for reproducibility
- [ ] 6C.3 — Configuration files for all benchmarks
- [ ] 6C.4 — Link to repository (when ready for public)

### 6D: Submission

- [ ] 6D.1 — Internal review pass (read it cold, find the holes)
- [ ] 6D.2 — LaTeX formatting per arXiv standards
- [ ] 6D.3 — Submit to arXiv (cs.NA or physics.comp-ph)
- [ ] 6D.4 — Generate final attestation: `STEP6_ARXIV_SUBMISSION_ATTESTATION.json`
- [ ] 6D.5 — Git tag: `v1.0.0-arxiv`

---

## CONSTITUTIONAL COMPLIANCE

All steps must satisfy:

```
☐ No shortcuts, mocks, placeholders, or stubs
☐ Complete error handling and type hints
☐ Attestation JSON with SHA256 hash at each step
☐ Git commit with descriptive message at each step
☐ This checklist updated with results at each step
☐ prove_qtt_turbulence.py passes after every change
☐ 256³ still fits in 8GB after every change
```

---

## DECISION GATES

```
After Step 1:
  IF dissipation < 1% at 128³ → Step 2 is optimization (nice to have)
  IF dissipation > 1% at 128³ → Step 2 is mandatory

After Step 3:
  IF K41 validated in DHIT → Step 4 is comparison (strengthens claim)
  IF K41 NOT validated → STOP. Diagnose before proceeding.

After Step 5:
  IF χ ~ Re^α with α < 0.1 → Thesis validated. Paper is strong.
  IF α > 0.1 but < 0.5 → Thesis partially holds. Paper frames the limit.
  IF α > 0.5 → Thesis fails. Paper reframes as "QTT for moderate Re."
  ALL THREE ARE PUBLISHABLE. Honesty is the claim.
```

---

*This checklist extends Workflow_Development.md Phases 1-6. All work governed by CONSTITUTION.md.*
