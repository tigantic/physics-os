# Reproducible Test Evidence Report

**Generated**: 2025-12-17T23:30:00Z  
**Package**: tensornet v0.1.0  
**Environment**: Python 3.11.9, PyTorch 2.9.1+cpu, Windows-10-10.0.26200-SP0  
**Status**: ALL TESTS PASSED (16/16)

---

## Artifact Verification

| Artifact | SHA256 |
|----------|--------|
| proof_run_20251217_233000.json | `4C25CEE5B8C65374C6554634F8FF7401EE6BB347429FAF72473A57BDEC005785` |
| Git commit | `669bfb3d9d63efbaaa9e4524ba3e6ba9684ed8fa` |
| Repository | https://github.com/tigantic/PytorchTN (private) |

**Verify**: Run `Get-FileHash -Algorithm SHA256 <file>` (PowerShell) or `sha256sum <file>` (Unix).

---

## Executive Summary

This document reports the results of a **reproducible test suite** supporting correctness for the covered operations under the stated environment and tolerances. Each test compares computed values against analytical solutions or reference implementations with explicit numerical tolerances.

**What this is**: A test report with quantitative measurements.  
**What this is not**: A formal mathematical proof of correctness for all inputs.

---

## Proof Results

### Category 1: Tensor Decompositions

#### Test 1.1: SVD Truncation Optimality (Eckart-Young-Mirsky)

**Claim**: `svd_truncated(A, k)` produces the optimal rank-k approximation under the **Frobenius norm**.

**Theorem**: By the Eckart-Young-Mirsky theorem, the best rank-k approximation to A (minimizing ||A - A_k||_F) is given by truncating the SVD to the top k singular values.

**Method**: Compare ||A - U @ diag(S) @ Vh||_F to torch.linalg.svd truncated to same rank.

| k  | Our Error | Reference Error | Difference |
|----|-----------|-----------------|------------|
| 5  | 8.066252e+01 | 8.066252e+01 | 0.00e+00 |
| 10 | 7.256999e+01 | 7.256999e+01 | 0.00e+00 |
| 20 | 5.808995e+01 | 5.808995e+01 | 0.00e+00 |
| 40 | 3.379407e+01 | 3.379407e+01 | 0.00e+00 |

**Verdict**: PASS (difference = 0 for all k)

---

#### Test 1.2: SVD Orthogonality

**Claim**: U and V from SVD are orthogonal matrices.

| Property | Measured Error | Tolerance |
|----------|----------------|-----------|
| ||U^T @ U - I|| | 8.90e-15 | 1e-10 |
| ||V @ V^T - I|| | 8.00e-15 | 1e-10 |

**Verdict**: PASS

---

#### Test 1.3: QR Reconstruction

**Claim**: QR decomposition satisfies A = Q @ R.

| Property | Measured Error | Tolerance |
|----------|----------------|-----------|
| ||A - Q @ R|| | 3.03e-14 | 1e-10 |

**Verdict**: PASS

---

#### Test 1.4: QR Orthogonality

**Claim**: Q from QR is orthogonal.

| Property | Measured Error | Tolerance |
|----------|----------------|-----------|
| ||Q^T @ Q - I|| | 3.67e-15 | 1e-10 |

**Verdict**: PASS

---

### Category 2: MPS Operations

#### Test 2.1: MPS Round-Trip Fidelity

**Claim**: tensor -> MPS -> tensor preserves information.

| Property | Value |
|----------|-------|
| Original shape | [2, 2, 2, 2, 2] |
| MPS bond dimension | 4 |
| ||T - T_reconstructed|| | 1.25e-15 |
| Tolerance | 1e-10 |

**Verdict**: PASS

---

#### Test 2.2: GHZ Entanglement Entropy

**Claim**: GHZ state has entropy = ln(2) = 0.6931471805599453 at every bond.

| Bond | Computed S | Theoretical S | Error |
|------|------------|---------------|-------|
| 0 | 0.6931471806 | 0.6931471806 | 1.11e-16 |
| 1 | 0.6931471806 | 0.6931471806 | 1.11e-16 |
| 2 | 0.6931471806 | 0.6931471806 | 1.11e-16 |
| 3 | 0.6931471806 | 0.6931471806 | 1.11e-16 |
| 4 | 0.6931471806 | 0.6931471806 | 1.11e-16 |

**Verdict**: PASS (all errors < 1e-10)

---

#### Test 2.3: Product State Zero Entropy

**Claim**: Product state has zero entanglement entropy.

| Bond | Computed S | Expected | Status |
|------|------------|----------|--------|
| 0 | 0.00e+00 | 0 | PASS |
| 1 | 0.00e+00 | 0 | PASS |
| 2 | 0.00e+00 | 0 | PASS |
| 3 | 0.00e+00 | 0 | PASS |
| 4 | 0.00e+00 | 0 | PASS |

**Verdict**: PASS

---

#### Test 2.4: Norm Preservation Under Canonicalization

**Claim**: Canonicalization preserves MPS norm.

| Property | Value |
|----------|-------|
| Norm before | 8277.4715296473 |
| Norm after | 8277.4715296473 |
| Difference | 5.46e-12 |
| Tolerance | 1e-10 |

**Verdict**: PASS

---

#### Test 2.5: Left-Canonical Orthogonality

**Claim**: Left-canonical tensors satisfy A^T @ A = I.

| Site | ||A^T A - I|| | Status |
|------|--------------|--------|
| 0 | 0.00e+00 | PASS |
| 1 | 7.24e-16 | PASS |
| 2 | 7.93e-16 | PASS |
| 3 | 5.27e-16 | PASS |
| 4 | 5.14e-16 | PASS |
| 5 | 1.04e-15 | PASS |
| 6 | 4.47e-16 | PASS |

**Verdict**: PASS (all errors < 1e-10)

---

### Category 3: Physical Invariants

#### Test 3.1: Pauli Algebra (Commutators)

**Claim**: Pauli matrices satisfy SU(2) algebra.

| Relation | Expected | Measured Error |
|----------|----------|----------------|
| [X,Y] = 2iZ | Exact | 0.00e+00 |
| [Y,Z] = 2iX | Exact | 0.00e+00 |
| [Z,X] = 2iY | Exact | 0.00e+00 |

**Verdict**: PASS (machine precision)

---

#### Test 3.2: Pauli Anticommutators

**Claim**: Pauli matrices satisfy anticommutation relations.

| Relation | Expected | Measured Error |
|----------|----------|----------------|
| {X,X} = 2I | Exact | 0.00e+00 |
| {X,Y} = 0 | Zero | 0.00e+00 |
| {Y,Z} = 0 | Zero | 0.00e+00 |

**Verdict**: PASS (machine precision)

---

### Category 4: Autograd Correctness

#### Test 4.1: SVD Gradient Correctness

**Claim**: Gradients of svd_truncated match finite differences.

| Test | Result |
|------|--------|
| torch.autograd.gradcheck | PASSED |

**Verdict**: PASS

---

#### Test 4.2: MPS Norm Gradient

**Claim**: Gradient of MPS norm is correct.

| Test | Result |
|------|--------|
| torch.autograd.gradcheck | PASSED |

**Verdict**: PASS

---

### Category 5: Algorithm Correctness

#### Test 5.1: Lanczos Ground State Energy

**Claim**: Lanczos finds correct ground state eigenvalue.

| Property | Value |
|----------|-------|
| Exact E0 (torch.linalg.eigvalsh) | -5.4124839707 |
| Lanczos E0 | -5.4124839707 |
| Error | 6.22e-15 |
| Tolerance | 1e-8 |

**Verdict**: PASS

---

#### Test 5.2: Heisenberg MPO Hermiticity

**Claim**: Heisenberg Hamiltonian MPO is Hermitian.

| Property | Measured Error | Tolerance |
|----------|----------------|-----------|
| ||H - H^T|| | 0.00e+00 | 1e-10 |

**Verdict**: PASS

---

## Summary Table

| Proof ID | Description | Status | Max Error |
|----------|-------------|--------|-----------|
| 1.1 | SVD Truncation Optimality | PASS | 0.00e+00 |
| 1.2 | SVD Orthogonality | PASS | 8.90e-15 |
| 1.3 | QR Reconstruction | PASS | 3.03e-14 |
| 1.4 | QR Orthogonality | PASS | 3.67e-15 |
| 2.1 | MPS Round-Trip | PASS | 1.25e-15 |
| 2.2 | GHZ Entropy | PASS | 1.11e-16 |
| 2.3 | Product State Entropy | PASS | 0.00e+00 |
| 2.4 | Norm Preservation | PASS | 5.46e-12 |
| 2.5 | Canonical Orthogonality | PASS | 1.04e-15 |
| 3.1 | Pauli Commutators | PASS | 0.00e+00 |
| 3.2 | Pauli Anticommutators | PASS | 0.00e+00 |
| 4.1 | SVD Gradient | PASS | gradcheck |
| 4.2 | MPS Gradient | PASS | gradcheck |
| 5.1 | Lanczos Eigenvalue | PASS | 6.22e-15 |
| 5.2 | MPO Hermiticity | PASS | 0.00e+00 |

---

## Reproducibility

To reproduce these tests:

```bash
# 1. Install package
cd tensornet_standalone
python -m pip install -e .

# 2. Run tests
pytest -q tests/test_proofs.py --disable-warnings

# 3. Verify environment
python -c "import torch; print(torch.__version__)"
```

**Package structure**: `tensornet_standalone/` is a self-contained pip-installable package. It contains:
- `tensornet/` - The library source (copied from torch/tensornet development)
- `tests/test_proofs.py` - The test file that generates these results
- `pyproject.toml` - Package metadata

---

## Compliance Notes

Per Article I, Section 1.1 of the Constitutional Law:

- **Reproducible**: Tests run with `torch.manual_seed(42)`
- **Quantitative**: All errors measured numerically in float64
- **Falsifiable**: Clear pass/fail criteria with explicit tolerances
- **Compared to reference**: Analytical solutions and torch.linalg

---

## What This Does NOT Prove

1. Correctness for all possible inputs (only tested inputs)
2. Numerical stability under extreme conditions
3. Performance guarantees
4. Correctness of algorithms not covered by tests (e.g., full DMRG convergence)

---

*This document reports reproducible test evidence for tensornet v0.1.0.*  
*Machine-readable artifact: [proof_run_20251217_233000.json](proof_run_20251217_233000.json)*
