# Risks & Gaps Audit

**Created**: 2025-12-17  
**Last Updated**: 2025-12-21  
**Status**: Living Document  

---

## Risks & Gaps (Ranked by Priority)

### 1. 🚩 Algorithmic Correctness Gaps in New Domains (High)

**Status**: ✅ RESOLVED (2025-12-21) — DMRG fixed, CFD pending validation

~~DMRG sometimes not fully converged~~ — Fixed! Root cause was incorrect index ordering in environment contractions (`_contract_left_env`, `_contract_right_env`) and `matvec` function. All 3 xfail tests in `test_dmrg_physics.py` now pass.

The 1D CFD solver (Euler1D) is implemented and tested for basic sanity (shock tube initial condition), but more complex scenarios (shock capturing accuracy, boundary condition handling, etc.) aren't fully validated.

**Mitigation**: Add "proof" tests for CFD (conservation laws, exact solutions).

**DMRG Fix Details**:
- Fixed `_contract_left_env`: Changed einsum from `'awa,asb->wasb'` to `'awc,asb->wcsub'` to maintain separate ket/bra virtual indices
- Fixed `_contract_right_env`: Changed einsum to correctly contract `(a,s,n,d)` indices  
- Fixed `matvec` in `_two_site_eigensolve`: Changed W1 contraction from `'wcsub,wtum->ctubm'` to `'wcsub,wtsm->ctubm'` and W2 from `'ctubm,mvsn->ctvbn'` to `'ctubm,mvun->ctvbn'`
- Fixed `MPO.expectation`: Replaced broken 5-index Greek einsum with correct step-by-step contraction

---

### 2. 🚩 Incomplete Feature Implementations (Medium-High)

**Status**: ⚠️ OPEN — Expected per roadmap

Many modules are present but essentially stubs (marked with `NotImplementedError`). Examples: `tensornet.cfd.les.py`, `adaptive/`, `autonomy/` are mostly placeholders per the Execution Tracker.

**Mitigation**: Clearly communicate which modules are experimental. Add notes in README or module docstrings (like "Phase 10 – not yet implemented").

---

### 3. 🚩 Performance Optimization Not Yet Proven at Scale (Medium)

**Status**: ⚠️ OPEN

Python-level loops in CFD stepping could become slow for large grids. Memory scaling of MPS for 2D may hit limits.

**Mitigation**: Profile existing benchmarks, integrate C++/CUDA optimization for critical loops, test on moderate grid sizes.

---

### 4. 🚩 Complexity of Contribution (Medium)

**Status**: ⚠️ OPEN

High standards (Constitution, strict linting, proofs for new algorithms) could intimidate new contributors.

**Mitigation**: Provide CONTRIBUTING.md with TL;DR of Constitution's key points. Label "good first issue" items.

---

### 5. 🚩 Documentation Lag for New Features (Medium)

**Status**: ⚠️ OPEN

Documentation may fall behind as features are added rapidly.

**Mitigation**: Make updating docs part of definition of done for each phase/feature. Integrate documentation builds in CI.

---

### 6. 🚩 CI Gaps & Flaky Integration (Low-Medium)

**Status**: ✅ RESOLVED (2025-12-21)

~~Two issues: (a) CI not running benchmarks due to misconfigured module path. (b) Integration tests masked with `|| true`.~~

**Resolution**:
- ✅ Benchmark path fixed — `benchmarks/` now correctly referenced
- ✅ Integration tests run properly in CI
- ✅ Test naming refactored to Constitutional format (181 tests renamed per Article III.3.2)

---

### 7. 🚩 External Dependency Risks (Low)

**Status**: ⚠️ OPEN — Acceptable

Project depends on PyTorch 2.x and targets Python 3.11+. Bleeding-edge versions may have breaking changes.

**Mitigation**: Pin versions in requirements-lock.txt. CI already tests Python 3.12.

---

### 8. 🚩 Future Integration Challenges (Low)

**Status**: ⚠️ OPEN — Future concern

Integrating into actual flight systems may surface issues (deterministic memory, C/C++ interfaces).

**Mitigation**: Roadmap includes Phase 11 deployment, Phase 12 simulation. TensorRT export stub already in place.

---

### 9. 🚩 Lack of User Feedback Loop (Low)

**Status**: ⚠️ OPEN

No public issues or discussions visible. Project may be building features users don't need.

**Mitigation**: Consider PyPI release or engaging with academics in fluids/quantum for feedback.

---

### 10. 🚩 Documentation Overload (Low)

**Status**: ⚠️ OPEN

New developers might be overwhelmed by volume (Constitution 400+ lines, Execution Tracker 1500+ lines).

**Mitigation**: Provide summary sections. Maintain short "Project Status" in README.

---

## Summary

| Priority | Risk | Status |
|----------|------|--------|
| High | Algorithmic correctness (DMRG, CFD) | ✅ DMRG FIXED / CFD pending |
| Medium-High | Incomplete features | ⚠️ Expected |
| Medium | Performance at scale | ⚠️ OPEN |
| Medium | Contribution complexity | ⚠️ OPEN |
| Medium | Documentation lag | ⚠️ OPEN |
| Low-Medium | CI gaps | ✅ RESOLVED |
| Low | External dependencies | ⚠️ Acceptable |
| Low | Future integration | ⚠️ Future |
| Low | User feedback loop | ⚠️ OPEN |
| Low | Documentation overload | ⚠️ OPEN |

The highest priority items are:
1. ~~Shore up testing and correctness in DMRG~~ ✅ DONE
2. Add proof tests for CFD (conservation laws, exact solutions)
3. Keep performance on the radar as complexity grows

None are fundamental flaws; they are areas to watch in an otherwise well-run project.

---

## Recommended Next Steps & Roadmap (30–60–90 Days)

### Next 30 Days (Immediate) — Stabilize and Fix Known Issues

- [x] **Resolve DMRG Convergence Issues**: ~~Investigate the 3 `xfail` tests~~ — DONE (2025-12-21)
  - Fixed `_contract_left_env` and `_contract_right_env` index ordering
  - Fixed `matvec` in `_two_site_eigensolve` einsum contractions
  - Fixed `MPO.expectation` method
  - All 3 tests now pass with machine-precision accuracy

- [x] **Fix CI Benchmark Command**: ~~Correct the module path in the benchmark job~~ — DONE

- [x] **Enable Integration Tests in CI**: ~~Remove `|| true` masking~~ — DONE

- [x] **Test Naming Refactor**: ~~Rename tests to Constitutional format~~ — DONE (181 tests renamed)

- [x] **Document Phase 2 Progress**: ~~Update README to mention Euler1D, shock tube ICs, Riemann solvers~~ — DONE (2025-12-21)

- [x] **Extend CFD Testing**: ~~Verify Riemann solver functions respect physical bounds~~ — DONE (2025-12-21)
  - Created `tests/integration/test_cfd_physics.py` with 15 physics tests
  - Tests: Sod, Lax, double-rarefaction shock tubes
  - Verifies ρ > 0, p > 0 for exact_riemann
  - Tests HLL, HLLC, Roe flux validity and consistency
  - Verifies primitive↔conserved roundtrip conversion
  - Tests Rankine-Hugoniot mass conservation at shocks

- [ ] **Community Visibility**: Consider open-sourcing or announcing to broader audience.

---

### Next 60 Days (Mid-term) — Enhance Features and Usability

- [ ] **Finalize Phase 2 – 1D CFD**:
  - Implement missing pieces in `limiters.py` (TVD limiters: Minmod, Superbee)
  - Test boundary conditions (reflective, outflow) with simulation scenarios
  - Postpone adjoint/optimization until forward solver is fully validated

- [ ] **Phase 3 – 2D Solver Integration**:
  - Debug and test Euler2D with small 2D test cases
  - Supersonic flow over wedge (geometry.py)
  - Strang dimensional splitting validation

- [ ] **Performance Profiling**:
  - Use PyTorch profiler to identify slow Python loops
  - Increase bond dimension/grid size in benchmarks to find bottlenecks
  - Integrate MemoryPool for GPU runs

- [ ] **Packaging**: Dry-run of wheel build, consider TestPyPI publish

---

### Next 90 Days (Longer-term) — Expand Capability and Prepare for Wider Use

- [ ] **Phase 3 Completion**: 2D CFD feature-complete with demo (Mach reflection, shock interaction)

- [ ] **User Documentation & Outreach**:
  - Complete auto-generated API Reference
  - Write tutorial article ("MPS for spin chain physics" or "Compressing CFD simulation")
  - Consider Zenodo DOI for citation

- [ ] **Community Engagement**:
  - Set up GitHub Discussions
  - Label "good first issue" items
  - Collaborate with TeNPy or CFD library authors

- [ ] **Technical Debt Cleanup**:
  - Remove outdated code references
  - Re-evaluate module plan, drop unneeded stubs

- [ ] **Performance & Scaling Tests**:
  - 2D solver scaling experiments
  - Prototype distributed DMRG
  - GPU acceleration for CFD

- [ ] **Compliance & Quality**:
  - Lightweight requirements mapping to tests
  - Archive proof results with releases

---

*By following this staged plan, in 90 days HyperTensor should have robust 1D/2D capability, a growing user base, and polished presentation — moving from "alpha" to "beta" stage while laying groundwork for advanced phases (3D, control, deployment).*
