# Requirements Traceability Matrix

This document maps high-level requirements to implementing code and tests.

## Core MPS/MPO Requirements

| ID | Requirement | Implementation | Tests |
|----|-------------|----------------|-------|
| MPS-01 | MPS can represent arbitrary quantum states | `tensornet/core/mps.py` | `test_mps_random_creation_*` |
| MPS-02 | MPS normalization preserves quantum norm | `MPS.normalize()` | `test_mps_normalization_*` |
| MPS-03 | MPO encodes local Hamiltonians | `tensornet/core/mpo.py` | `test_mpo_heisenberg_*` |
| MPS-04 | MPO-MPS contraction computes expectation | `MPO.expectation()` | `test_mpo_expectation_*` |

## Algorithm Requirements

| ID | Requirement | Implementation | Tests |
|----|-------------|----------------|-------|
| ALG-01 | DMRG converges to ground state | `tensornet/algorithms/dmrg.py` | `test_dmrg_heisenberg_ground_state_*` |
| ALG-02 | TEBD performs unitary time evolution | `tensornet/algorithms/tebd.py` | `test_tebd_*` |
| ALG-03 | TDVP preserves MPS manifold | `tensornet/algorithms/tdvp.py` | `test_tdvp_*` |
| ALG-04 | Lanczos finds lowest eigenvalue | `tensornet/algorithms/lanczos.py` | `test_lanczos_*` |

## Physics Validation Requirements

| ID | Requirement | Implementation | Tests |
|----|-------------|----------------|-------|
| PHY-01 | Heisenberg chain: E/L → -0.4432 (exact) | `heisenberg_mpo()` | `test_dmrg_heisenberg_energy_*` |
| PHY-02 | TFIM: correct phase transition at g=1 | `tfim_mpo()` | `test_tfim_phase_transition_*` |
| PHY-03 | Bethe ansatz comparison | - | `Physics/benchmarks/heisenberg_ground_state.py` |

## CFD Requirements

| ID | Requirement | Implementation | Tests |
|----|-------------|----------------|-------|
| CFD-01 | Euler1D conserves mass/momentum/energy | `tensornet/cfd/euler_1d.py` | `test_euler1d_conservation_*` |
| CFD-02 | Shock tube matches exact Riemann | `tensornet/cfd/godunov.py` | `test_cfd_sod_shock_tube_*` |
| CFD-03 | Euler2D uses Strang splitting | `tensornet/cfd/euler_2d.py` | `test_euler2d_*` |
| CFD-04 | Oblique shock matches NACA 1135 | `oblique_shock_exact()` | `scripts/wedge_flow_demo.py` |
| CFD-05 | Boundary conditions: all types | `BCType`, `BCType1D` | `test_cfd_bc_*` |

## Quality Requirements

| ID | Requirement | Implementation | Tests |
|----|-------------|----------------|-------|
| QUA-01 | Test names follow Constitutional format | All tests | CI naming check |
| QUA-02 | Proof tests archive results | `proofs/` | `tests/test_proofs.py` |
| QUA-03 | API documentation generated | `tensornet/docs/api_reference.py` | 86 markdown files |
| QUA-04 | Code passes ruff + black | CI workflow | `.github/workflows/ci.yml` |

## Proof Test Archive

Proof results are archived in `proofs/` with each release:
- `proof_algorithms_result.json` — DMRG, TEBD convergence proofs
- `proof_decompositions_result.json` — SVD, QR stability proofs  
- `proof_mps_result.json` — MPS norm, orthogonality proofs

## Test Coverage Summary

| Module | Unit Tests | Integration Tests | Physics Tests |
|--------|------------|-------------------|---------------|
| `core/` | 45+ | 10+ | - |
| `algorithms/` | 30+ | 15+ | 5+ |
| `cfd/` | 19+ | 15+ | 5+ |
| `mps/` | 10+ | 5+ | 3+ |

Total: ~180 tests, all passing.

---

*Last updated: 2025-12-21*
