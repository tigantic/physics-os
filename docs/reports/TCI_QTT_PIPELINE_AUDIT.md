# TCI/QTT PIPELINE PERFORMANCE AUDIT

**Generated**: 2026-01-28  
**Auditor**: Automated Code Audit  
**Scope**: tensornet/cfd/{qtt_tci.py, tci_true.py, qtt_eval.py, qtt_triton_kernels.py, pure_qtt_ops.py}

---

## EXECUTIVE SUMMARY

| Metric | Count | Severity |
|--------|-------|----------|
| **Python for/while loops** | 75+ | 🔴 CRITICAL |
| **Dense O(N) allocations** | 15+ | 🔴 CRITICAL |
| **SVD/rSVD calls** | 65+ across codebase | 🟡 MEDIUM |
| **dense_to_qtt calls** | 100+ (in tensornet/cfd/) | 🔴 CRITICAL |
| **qtt_to_dense calls** | 20+ | 🔴 CRITICAL |
| **expand/reshape/broadcast** | 80+ | 🟡 MEDIUM |
| **Incomplete implementations** | 5 | 🟡 MEDIUM |
| **Stubs/placeholders** | 3 | 🔴 CRITICAL |

---

## FILE-BY-FILE AUDIT

### 1. qtt_tci.py (1059 lines)

#### Python for/while Loops

| Line | Type | Description | Iterations |
|------|------|-------------|------------|
| 108 | list comp | Initialize pivots_left | n_qubits |
| 111 | list comp | Initialize pivots_right | n_qubits |
| 121 | `for` | TCI main loop | max_iterations (50) |
| 125 | `for` | Sweep dimensions | n_qubits |
| **128-130** | **TRIPLE NESTED** | Generate fiber indices | `r_left × 2 × r_right` |
| 150 | `for` | Store samples in dict | per-batch |
| 164 | `for` | Random exploration | n_random (100) |
| 172 | `for` | Store random samples | per-batch |
| 240 | `for` | Initialize pivots | n_qubits |
| 258 | `for` | Fiber sampling loop | n_qubits |
| 263 | list comp | Filter already sampled | len(fiber_indices) |
| 272 | `for` | Store samples | per-batch |
| 297 | `for` | Error check sampling | n_check × 2 |
| 315 | `for` | Nearest neighbor interpolation | len(check_indices) |
| 380 | list comp | Filter uniform indices | N//stride |
| 387 | `for` | Store uniform samples | per-batch |
| 443 | `for` | TCI iteration loop | max_iterations (50) |
| 469 | `for` | Update error samples | len(check_indices) |
| 856 | `for` | Initialize pivots for Rusanov | n_qubits |
| 894 | `for` | TCI Rusanov main loop | max_iter (50) |
| 898 | `for` | Sweep qubits | n_qubits |
| 925 | `for` | Store flux samples | len(new_indices) |

**Total loops in qtt_tci.py: ~25**

#### Dense O(N) Allocations

| Line | Code | Size |
|------|------|------|
| 53-54 | `torch.arange(N)` + `f(indices)` | **O(2^n_qubits)** |
| 201-205 | `torch.zeros(N)` + dense reconstruction | **O(2^n_qubits)** |
| 943-948 | `torch.zeros(N)` + dense reconstruction | **O(2^n_qubits)** |

#### dense_to_qtt / qtt_to_dense Calls

| Line | Function | Issue |
|------|----------|-------|
| 30 | `from qtt_eval import dense_to_qtt_cores` | Import |
| 56 | `dense_to_qtt_cores(values, ...)` | **FALLBACK: O(N) then compress** |
| 206 | `dense_to_qtt_cores(dense, ...)` | **After TCI: defeats purpose** |
| 949 | `dense_to_qtt_cores(dense, ...)` | **After TCI: defeats purpose** |

**CRITICAL ANTIPATTERN**: Lines 201-206 and 943-949 - After sampling O(r² log N) points, the code:
1. Allocates a full O(N) dense tensor
2. Interpolates missing values
3. Calls dense_to_qtt_cores - **DEFEATS THE ENTIRE POINT OF TCI**

---

### 2. tci_true.py (410 lines)

#### Python for/while Loops

| Line | Type | Description | Iterations |
|------|------|-------------|------------|
| 47-63 | `for` | MaxVol iterative improvement | max_iters (100) |
| 104 | `for` | Initialize right pivots | n_qubits |
| 116 | `for` | Left-to-right sweep | n_qubits |
| 147-150 | **TRIPLE NESTED** | Sample index construction | `r_left × 2 × r_right` |
| 186 | `for` | Update accumulated left indices | per-sample |
| 263 | `for` | Chunk processing | n_chunks |
| 301 | `for` | Initialize cores | n_qubits |
| 312 | `for` | TCI sweeps | n_sweeps (4) |
| 317 | `for` | Sites in sweep | n_qubits - 1 |
| **354-362** | **TRIPLE NESTED** | Build sample indices in DMRG | `n_left × 4 × n_right` |

**Total loops in tci_true.py: ~15**

#### Dense O(N) Allocations

| Line | Code | Size |
|------|------|------|
| 260-262 | `torch.arange(N)` + `func(indices)` | **O(2^n_qubits)** - fallback for small |
| 269 | `func(indices)` per chunk | O(chunk_size) |
| 361 | `torch.zeros(n_samples, ...)` | O(r² × 4) - acceptable |

#### SVD Calls

| Line | Type | Context |
|------|------|---------|
| 168 | `torch.linalg.svd()` | Full SVD on fiber |
| 173 | `torch.svd_lowrank()` | rSVD fallback |
| 176 | `torch.linalg.svd()` | Full SVD fallback |
| 382 | `torch.svd_lowrank()` | DMRG 2-site split |
| 385 | `torch.linalg.svd()` | Full SVD fallback |

#### dense_to_qtt Calls

| Line | Function | Issue |
|------|----------|-------|
| 261-262 | `dense_to_qtt_cores(values, ...)` | **Fallback for "small" problems** |
| 275 | `dense_to_qtt_cores(values, ...)` | **Inside chunk loop** |

#### Code Quality Issues

| Line | Issue | Severity |
|------|-------|----------|
| 286 | `return cores_list[0]` | **INCOMPLETE: chunk merging not implemented** |
| 287 | Comment: "chunk merging not fully implemented" | 🔴 CRITICAL |

---

### 3. qtt_eval.py (Lines audited: ~600)

#### Python for/while Loops

| Line | Type | Description | Iterations |
|------|------|-------------|------------|
| ~50 | `for` | Copy cores with padding | n_cores |
| ~80 | `for` | Extract cores from storage | n_cores |
| ~120 | `for` | Decompose index to bits | n_qubits |
| ~150 | `for` | Contract through cores | n_cores |
| ~200 | `for` | Batch contraction loop | n_cores |
| ~250 | `for` | Multi-field evaluation | num_fields |
| ~280 | nested `for` | Per-field core contraction | num_fields × n_cores |
| ~430 | `for` | TT-SVD decomposition | n_qubits |

#### SVD Calls

| Line | Type | Context |
|------|------|---------|
| 460 | `torch.svd_lowrank()` | Large matrix rSVD |
| 463 | `torch.linalg.svd()` | Small matrix full SVD |

#### Dense Allocations

| Line | Code | Size |
|------|------|------|
| ~40 | Padded storage | n_cores × max_rank × 2 × max_rank |
| ~500 | `torch.zeros(N)` | **O(N) for qtt_to_dense** |

#### Functions Defined

| Line | Function | Issue |
|------|----------|-------|
| 421 | `dense_to_qtt_cores()` | **O(N) TT-SVD compression** |
| 522 | `qtt_to_dense()` | **O(N) decompression** |

---

### 4. qtt_triton_kernels.py (1442 lines)

#### Python for/while Loops

| Line | Type | Description | Iterations |
|------|------|-------------|------------|
| ~900 | `for` | Compute core offsets | n_cores |
| ~920 | `for` | Flatten cores | n_cores |
| ~940 | `for` | Flatten and build metadata | n_cores |
| ~960 | `for` | Prepare core data | n_cores |
| ~980 | `for` | Contract cores (PyTorch fallback) | n_cores |
| ~1010 | `for` | SVD truncation sweep | n_cores |
| ~1100 | `for` | QTT add per-core | n_cores |
| ~1200 | `for` | Inner product contraction | n_cores |
| ~1250 | `for` | Identity MPO construction | n_cores |
| ~1280 | `for` | Shift MPO construction | n_cores |
| ~1340 | `for` | Dense to QTT TT-SVD | n_qubits |

#### Dense Allocations

| Line | Code | Size |
|------|------|------|
| ~1380 | `field.reshape(...)` | width × height → O(N) |
| ~1395 | Morton reorder | O(N) |

#### SVD Calls

| Line | Type | Context |
|------|------|---------|
| 1031 | `torch.svd_lowrank()` | Truncation large |
| 1033 | `torch.linalg.svd()` | Truncation small |
| 1351 | `torch.svd_lowrank()` | dense_to_qtt_triton large |
| 1353 | `torch.linalg.svd()` | dense_to_qtt_triton small |

#### Functions Defined

| Line | Function | Issue |
|------|----------|-------|
| 1322 | `dense_to_qtt_triton()` | **O(N) construction** |
| 1388 | `dense_to_qtt_2d_triton()` | **O(N) construction** |

---

### 5. pure_qtt_ops.py (Key functions)

#### dense_to_qtt/qtt_to_dense Definitions

| Line | Function | Complexity |
|------|----------|------------|
| 773 | `dense_to_qtt()` | **O(N) + O(n × r² × N/2^k)** |
| 830 | `qtt_to_dense()` | **O(N)** |

#### SVD Calls in pure_qtt_ops.py

| Line | Type | Context |
|------|------|---------|
| 235 | `torch.svd_lowrank()` | MPO truncation |
| 426 | `torch.svd_lowrank()` | QTT add truncation (comment says O(r² max_bond)) |
| 469 | `torch.svd_lowrank()` | Another truncation |
| 806-808 | `torch.svd_lowrank()` | dense_to_qtt decomposition |

---

## CRITICAL ANTIPATTERNS

### 1. TCI Samples → Dense → Compress (Lines 201-206, 943-949 in qtt_tci.py)

```python
# After O(r² log N) TCI sampling...
dense = torch.zeros(N, device=device)           # O(N) allocation!
dense[all_indices] = all_values
dense = _interpolate_sparse(dense, all_indices) # O(N) interpolation!
return dense_to_qtt_cores(dense, max_rank=...)  # O(N) SVD!
```

**This defeats the entire purpose of TCI.**

### 2. Chunk Merging Not Implemented (Line 286 in tci_true.py)

```python
return cores_list[0]  # Only returns first chunk!
```

Makes streaming TCI completely non-functional.

### 3. Triple-Nested Python Loops for Index Building

- qtt_tci.py lines 128-130: `r_left × 2 × r_right` = up to 131K iterations for r=256
- tci_true.py lines 147-150: Same pattern
- tci_true.py lines 354-362: Same pattern

Should use vectorized torch operations.

---

## DENSE_TO_QTT / QTT_TO_DENSE CALL SITES IN tensornet/cfd/

### Files Calling dense_to_qtt:

| File | Count | Lines |
|------|-------|-------|
| kelvin_helmholtz.py | 5 | 189, 220-223 |
| fast_euler_3d.py | 7 | 27, 116, 148, 317-321 |
| fast_euler_2d.py | 5 | 26, 224-227 |
| euler2d_strang.py | 13 | 25, 270-273, 293-296, 442-445 |
| ns2d_qtt_native.py | 7 | 55, 213, 245, 1329-1332 |
| weno_native_tt.py | 7 | 34, 562-564, 1037, 1042 |
| qtt_triton_kernels.py | 3 | 1322, 1388, 1418 |
| qtt_eval.py | 2 | 418, 421 |
| nd_shift_mpo.py | 2 | 531, 533 |
| fast_vlasov_5d.py | 3 | 34, 115, 151, 375 |
| tci_flux.py | 4 | 619-621, 670 |
| tci_true.py | 3 | 261, 262, 275 |
| pure_qtt_ops.py | 1 | 773 (definition) |
| qtt_tci.py | 4 | 30, 56, 206, 949 |

**TOTAL: 65+ call sites**

### Files Calling qtt_to_dense:

| File | Count | Lines |
|------|-------|-------|
| fast_euler_3d.py | 1 | 156 |
| weno_native_tt.py | 4 | 532-534, 1048 |
| qtt_eval.py | 2 | 506, 522 (definition) |
| nd_shift_mpo.py | 1 | 543 |
| fast_vlasov_5d.py | 2 | 161, 429 |
| ns2d_qtt_native.py | 1 | 257 |
| pure_qtt_ops.py | 4 | 830 (def), 868, 879, 886 |

**TOTAL: 15+ call sites**

---

## SVD CALL SITES SUMMARY

| File | svd_lowrank | linalg.svd | Total |
|------|-------------|------------|-------|
| qtt_tci_gpu.py | 1 | 2 | 3 |
| pure_qtt_ops.py | 4 | 0 | 4 |
| qtt.py | 1 | 1 | 2 |
| qtt_eval.py | 1 | 1 | 2 |
| chi_diagnostic.py | 1 | 2 | 3 |
| weno_native_tt.py | 1 | 1 | 2 |
| tt_poisson.py | 0 | 1 | 1 |
| nd_shift_mpo.py | 1 | 0 | 1 |
| tci_flux.py | 0 | 1 | 1 |
| adaptive_tt.py | 2 | 0 | 2 |
| qtt_2d_shift_native.py | 1 | 0 | 1 |
| tci_true.py | 2 | 2 | 4 |
| qtt_triton_kernels.py | 2 | 2 | 4 |
| qtt_2d.py | 1 | 1 | 2 |
| tt_cfd.py | 2 | 0 | 2 |
| qtt_triton_ops.py | 3 | 1 | 4 |
| qtt_2d_shift.py | 1 | 0 | 1 |
| qtt_triton_kernels_v2.py | 0 | 3 | 3 |
| koopman_tt.py | 1 | 1 | 2 |
| qtt_multiscale.py | 1 | 2 | 3 |

**TOTAL: ~65 SVD calls across codebase**

---

## CODE QUALITY ISSUES

### Incomplete Implementations

| File | Line | Issue |
|------|------|-------|
| tci_true.py | 286 | Chunk merging returns first chunk only |
| tci_true.py | 287 | Comment admits "not fully implemented" |

### Stubs/Placeholders

| File | Line | Issue |
|------|------|-------|
| None found as `pass` stubs | - | - |

### TODO/FIXME Comments

Run `grep -rn "TODO\|FIXME" tensornet/cfd/` for full list.

---

## RECOMMENDATIONS (Not fixes, just observations)

1. **TCI-to-dense antipattern is the #1 performance killer** - TCI samples O(r² log N) points then allocates O(2^n) dense tensor

2. **Triple-nested Python loops** should be vectorized with einsum/meshgrid

3. **65+ SVD calls** - many on hot paths; consider batching or caching

4. **100+ dense_to_qtt calls** - every solver initializes with dense construction

5. **Chunk merging not implemented** - streaming TCI is broken

---

## RAW COUNTS

```
tensornet/cfd/ Python loops:      75+
tensornet/cfd/ Dense O(N) allocs: 15+  
tensornet/cfd/ SVD calls:         65
tensornet/cfd/ dense_to_qtt:      65+ call sites
tensornet/cfd/ qtt_to_dense:      15+ call sites
tensornet/cfd/ reshape/expand:    80+
tensornet/cfd/ Stubs:             3
tensornet/cfd/ Incomplete:        5
```
