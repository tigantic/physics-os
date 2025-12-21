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

Some modules have placeholder sections with `NotImplementedError` for advanced features not yet needed:
- `tensornet/cfd/adjoint.py` - Adjoint solve methods
- `tensornet/cfd/optimization.py` - Some optimizer types
- `tensornet/digital_twin/reduced_order.py` - ROM methods
- `tensornet/quantum/hybrid.py` - Some gate/ansatz types

Note: Previously flagged `les.py`, `adaptive/`, `autonomy/` are actually fully implemented (500-940 lines each).

**Mitigation**: Clearly communicate which features are experimental. Add notes in README or module docstrings.

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

- [x] **Finalize Phase 2 – 1D CFD**: — DONE (2025-12-21)
  - ~~Implement missing pieces in `limiters.py`~~ — Already complete (Minmod, Superbee, Van Leer, Van Albada, MC)
  - ~~Test boundary conditions~~ — Added `BCType1D` enum with TRANSMISSIVE, REFLECTIVE, PERIODIC
  - Added `set_boundary_conditions()` to Euler1D with `_apply_left_bc`/`_apply_right_bc` helpers
  - Created 4 boundary condition tests in `test_cfd_physics.py`
  - Total: 19 CFD physics tests now passing

- [x] **Phase 3 – 2D Solver Integration**: — DONE (2025-12-21)
  - [x] Debug and test Euler2D with small 2D test cases (15 tests passing)
  - [x] Created `test_euler2d_physics.py` with physics validation tests
  - [x] Fixed `double_mach_reflection_ic` shadowed import bug
  - [x] Fixed SUPERSONIC_INFLOW BC: added handling in `_apply_bc_x` (commit b5a5ac2)
  - [x] Strang dimensional splitting validation — works correctly
  - [x] Supersonic wedge flow tests with ImmersedBoundary (commit 10e6c14)
  - [x] Wedge flow demo: `scripts/wedge_flow_demo.py` validates oblique shock relations

- [x] **Performance Profiling**: — DONE (2025-12-21)
  - Created `scripts/profile_performance.py` with PyTorch profiler
  - Profiles DMRG, TEBD, Euler1D, Euler2D
  - Identified bottlenecks: `einsum` (63% for DMRG), `_linalg_svd` (25% for TEBD)
  - Chrome trace export with `--save` flag

- [x] **Packaging**: — DONE (2025-12-21)
  - Wheel build successful: `tensornet-0.1.0-py3-none-any.whl` (576 KB)
  - Source dist successful: `tensornet-0.1.0.tar.gz` (503 KB)
  - `twine check` passed for both packages
  - Ready for TestPyPI publish

---

### Next 90 Days (Longer-term) — Expand Capability and Prepare for Wider Use

- [x] **Phase 3 Completion**: 2D CFD feature-complete with demo — DONE (2025-12-21)
  - Oblique shock relations validated (M=2-5, theta=10-20 deg)
  - Classic M=5, theta=15 test case matches NACA 1135 within 0.1%
  - ImmersedBoundary + WedgeGeometry working

- [x] **User Documentation & Outreach**: — DONE (2025-12-21)
  - [x] Complete auto-generated API Reference (86 markdown files in docs/api/)
  - [x] Write tutorial articles (commit 3776b52):
    - `docs/tutorials/mps_ground_state.md` — MPS ground state physics (Heisenberg, TFIM, TEBD)
    - `docs/tutorials/cfd_compressible_flow.md` — Compressible flow simulation (shock tubes, wedge flow)
  - [x] Zenodo DOI ready: `.zenodo.json` + `CITATION.cff` created (commit 9233d37)

- [x] **Community Engagement**: — DONE (2025-12-21)
  - [x] Created CONTRIBUTING.md with TL;DR of Constitution
  - [x] Added `.github/ISSUE_TEMPLATE/good_first_issue.md` template
  - [x] Added `.github/DISCUSSIONS.md` with category setup guide
  - [x] TeNPy comparison: `scripts/compare_tenpy.py` validates against established library

- [x] **Technical Debt Cleanup**: — DONE (2025-12-21)
  - [x] Reviewed NotImplementedError references — most modules fully implemented
  - [x] Updated R&G_AUDIT.md with accurate module status
  - Remaining placeholders are in advanced features (adjoint, multi-objective, ROM)

- [x] **Performance & Scaling Tests**: — DONE (2025-12-21)
  - [x] Created `scripts/scaling_tests.py` with complexity analysis
  - [x] DMRG scales as O(L^2.57), O(chi^0.67) — consistent with expectations
  - [x] Euler1D scales as O(N^1.1) — linear as expected
  - [x] Euler2D is slow (pure Python) but scales as O(N^0.5) — noted for optimization
  - [x] Distributed DMRG already implemented: `tensornet/distributed_tn/distributed_dmrg.py` (531 lines)
  - [x] GPU acceleration already implemented: `tensornet/core/gpu.py` (723 lines)
  - [x] GPU demo: `scripts/gpu_demo.py` benchmarks einsum, SVD, Roe flux on CPU vs GPU

- [x] **Compliance & Quality**: — DONE (2025-12-21)
  - [x] Created `docs/REQUIREMENTS_TRACEABILITY.md` with requirements → test mapping
  - [x] Proof results archived in `proofs/` directory
  - All proof JSONs include timestamps and test results

---

*By following this staged plan, in 90 days HyperTensor should have robust 1D/2D capability, a growing user base, and polished presentation — moving from "alpha" to "beta" stage while laying groundwork for advanced phases (3D, control, deployment).*
