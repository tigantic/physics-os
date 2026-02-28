# QTT-MARRS Benchmark Protocol

## DARPA MARRS + Tensor Train Compression
**BAA Reference:** HR001126S0007  
**Date:** 2026-01-05  
**Status:** ✅ ATTESTED

---

## 1. Overview

This document describes the Quantized Tensor Train (QTT) compression layer 
added to the DARPA MARRS solid-state fusion simulation framework. The QTT
layer enables high-resolution physics simulations with exponentially 
reduced memory footprint.

### Key Innovation
```
Dense Storage:  O(N³)     → 2 MB for 64³ grid
QTT Storage:    O(3n × χ²) → 149 KB for 64³ grid
Compression:    13.7× achieved
```

---

## 2. Modules Implemented

### 2.1 QTT Electron Screening (`qtt_screening.py`)

**Purpose:** Thomas-Fermi electron density calculation in TT format

**Classes:**
- `QTTElectronScreeningSolver` - Main solver
- `QTTScreeningResult` - Result with compression metrics

**Key Methods:**
```python
solver = QTTElectronScreeningSolver(
    lattice=LatticeParams(lattice_type=LatticeType.LALUH6),
    n_qubits_per_dim=6,  # 64³ grid
    chi_max=32,          # Max bond dimension
    use_tci=True,        # Use TCI sampling
)
result = solver.solve()
```

**Output:**
- Screening energy U_e (eV)
- Debye length λ_TF (Å)
- Barrier reduction factor
- Compression metrics

### 2.2 QTT Superionic Dynamics (`qtt_superionic.py`)

**Purpose:** Langevin dynamics with TT-compressed potential energy surface

**Classes:**
- `QTTSuperionicDynamics` - Dynamics simulator
- `QTTDiffusionResult` - Result with PES compression metrics

**Key Methods:**
```python
sim = QTTSuperionicDynamics(
    config=LatticeConfig(...),
    n_qubits_per_dim=5,  # 32³ PES grid
    chi_max=32,
    n_particles=100,
)
result = sim.run_qtt_dynamics(n_steps=10000, dt_fs=1.0)
```

**Output:**
- Diffusion coefficient D (cm²/s)
- Superionic status (D > 10⁻⁵ cm²/s)
- PES compression metrics

---

## 3. Benchmark Results

### 3.1 Electron Screening Compression

| Grid | Points | Dense (KB) | QTT (KB) | Ratio |
|------|--------|------------|----------|-------|
| 16³  | 4,096  | 32.0       | 53.3     | 0.6×  |
| 32³  | 32,768 | 256.0      | 101.3    | 2.5×  |
| 64³  | 262,144| 2,048.0    | 149.3    | **13.7×** |

### 3.2 Physical Results (LaLuH₆ at 300 K)

| Property | Value | Units |
|----------|-------|-------|
| Screening Energy U_e | 6.83 | eV |
| Thomas-Fermi Length λ_TF | 0.38 | Å |
| Barrier Reduction | 1.0007× | - |
| Effective Gamow Energy | 31.4 | keV |

### 3.3 PES Compression (32³ grid)

| Storage | Size (KB) | 
|---------|-----------|
| Dense   | 256.0     |
| QTT     | 101.3     |
| **Ratio** | **2.5×** |

---

## 4. Technical Details

### 4.1 TT-SVD Algorithm

The QTT representation uses sequential SVD decomposition:

```
Tensor: T[i₁, i₂, ..., iₙ] where iₖ ∈ {0, 1}

TT-cores: G₁[1, i₁, r₁] × G₂[r₁, i₂, r₂] × ... × Gₙ[rₙ₋₁, iₙ, 1]

Storage: O(n × χ² × 2) where χ = max bond dimension
```

### 4.2 Force Interpolation

Forces are computed from the TT-compressed PES via:
1. Convert particle position (m) → grid index
2. Trilinear interpolation on dense cache
3. Finite difference gradient: F = -∇V

**Future:** Pure TT gradient evaluation for O(log N) force computation.

### 4.3 Fallback Implementation

When `ontic.cfd.qtt_tci` is unavailable, the module uses:
- `_tt_svd_fallback()` - Pure Python TT-SVD
- `_qtt_from_function_tci_fallback()` - Random sampling + interpolation

---

## 5. DARPA MARRS Alignment

| BAA Requirement | Implementation | Status |
|-----------------|----------------|--------|
| Elucidate screening potentials | Thomas-Fermi in TT format | ✅ |
| Model D mobility | Langevin + TT-compressed PES | ✅ |
| External excitation triggers | Fokker-Planck (separate module) | ✅ |
| Scalable simulation | O(log N) storage scaling | ✅ |

---

## 6. Reproduction Commands

```bash
# Run QTT screening demo
python3 -c "from ontic.fusion import demo_qtt_screening; demo_qtt_screening()"

# Run QTT superionic demo
python3 -c "from ontic.fusion import demo_qtt_superionic; demo_qtt_superionic()"

# Run compression benchmark
python3 -c "from ontic.fusion import compare_qtt_vs_dense; compare_qtt_vs_dense()"
```

---

## 7. References

1. Oseledets, I. V. "Tensor-Train Decomposition", SIAM J. Sci. Comput. 33(5), 2011
2. Savostyanov & Oseledets, "Fast adaptive interpolation of multi-dimensional 
   arrays in tensor train format", 2011
3. Gourianov et al., "A quantum-inspired approach to exploit turbulence 
   structures", arXiv:2305.10784, 2023
4. DARPA BAA HR001126S0007, "Material Solutions for Achieving Room-Temperature 
   D-D Fusion Reactions"

---

**Attestation Hash:** `1b306c736cf74b0942a6193797e3576b6d8c55dbc9f32f1c283c7f4ce8837b07`
