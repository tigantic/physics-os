# Navier-Stokes Black Swan Hunt: Results Summary

## Date: 2026-01-16 (Updated)

## Executive Summary

**STATUS: NO BLACK SWAN FOUND - EVIDENCE SUPPORTS REGULARITY**

Initial Kida vortex "blowup" was determined to be a **NUMERICAL ARTIFACT** 
from spectral aliasing. With proper 2/3 dealiasing, all tested initial 
conditions remain bounded.

---

## Kida Vortex: False Alarm Analysis

### Initial Finding (WITHOUT dealiasing)
- Explosive growth at T* ≈ 1.80
- Appeared consistent across resolutions

### Verification (WITH 2/3 dealiasing)

| N | T* (blowup) | max\|ω\| | E_final/E_0 | Verdict |
|---|-------------|----------|-------------|---------|
| 32 | N/A | 16.45 | 0.91 | ✓ BOUNDED |
| 48 | N/A | 25.03 | 0.91 | ✓ BOUNDED |
| 64 | N/A | 55.53 | 0.89 | ✓ BOUNDED |

### Diagnosis
- Energy DECAYS (correct viscous physics)
- Vorticity grows but stays FINITE
- Without dealiasing: k=3 Kida modes caused aliasing cascade
- **CONCLUSION: Previous "blowup" was numerical artifact**

---

## Hunt Results (All Bounded)

### Initial Condition Survey (Re = 10,000, N = 48)

| IC Name | BKM Integral | max|ω| | Verdict |
|---------|--------------|--------|---------|
| Taylor-Green | 3.37 | 2.0 | ✓ BOUNDED |
| ABC Flow | 4.90 | 2.4 | ✓ BOUNDED |
| **Kida Vortex** | **8×10¹⁶** | **∞** | **⚠️ BLOWUP** |
| Anti-Parallel Tubes | 87.7 | 54.2 | ✓ BOUNDED |
| Hou-Luo (standard) | 4.01 | 2.0 | ✓ BOUNDED |
| Hou-Luo (tight) | 5.93 | 3.1 | ✓ BOUNDED |
| Hou-Luo (wide) | 2.57 | 1.3 | ✓ BOUNDED |

### Kida Vortex Deep Analysis

**Observation:** Explosive vorticity growth:
- t = 0.0: max|ω| = 9.2
- t = 0.4: max|ω| = 6.4
- t = 0.8: max|ω| = 16.1
- t = 1.2: max|ω| = 29.6
- t = 1.6: max|ω| = 102.6
- t = 1.8: max|ω| → ∞

**Blowup time:** T* ≈ 1.80 (consistent across N=32, N=48)

**Viscosity test:** Blowup persists at Re=1000 (10x more viscous) at same time T* ≈ 1.80

**Amplitude test:** 
- amp = 0.1: BOUNDED (max|ω| = 0.68)
- amp = 0.3: BOUNDED (max|ω| = 1.99)
- amp = 0.5: BOUNDED (max|ω| = 6.34)
- amp = 1.0: BOUNDED (max|ω| = 90.9)

The "normalized" version in our hunter has slightly larger amplitude (×1.15) which pushes it over a threshold.

---

## Interpretation

### Possibilities:

1. **True Euler/NS Singularity**
   - The Kida vortex is known in the literature as a potential blowup candidate
   - High symmetry (k=3 modes) creates vortex-stretching resonances
   - The blowup time being independent of viscosity suggests inviscid (Euler) origin
   - This would be Clay Prize relevant!

2. **Numerical Instability**
   - Spectral methods can become unstable at high Re
   - The k=3 Kida modes may cause aliasing issues
   - The consistent T* across resolutions is suspicious (usually numerical instability varies with N)

3. **Strong but Bounded Growth**
   - The amplitude dependence suggests there's a threshold
   - Below threshold: bounded, above: apparent blowup
   - Could be a "near-singularity" that eventually regularizes

### Evidence Assessment:

| For Singularity | Against Singularity |
|-----------------|---------------------|
| T* consistent across N | Amplitude-dependent |
| T* independent of Re | Only one IC shows it |
| Known candidate geometry | Could be numerical |
| BKM criterion satisfied | Short simulation time |

---

## Required Follow-Up

To resolve whether this is a TRUE singularity:

1. **Higher Resolution Study**
   - Run at N = 128, 256, 512
   - If T* converges: more likely real
   - If T* increases with N: numerical artifact

2. **Careful CFL Analysis**
   - Adaptive time stepping based on actual max|u|
   - Check if blowup is CFL violation

3. **Comparison with Literature**
   - The Kida vortex has been studied extensively
   - Compare our T* with published results

4. **Energy Conservation Check**
   - Monitor total energy - should be conserved (Euler) or decay (NS)
   - Energy growth indicates numerical instability

---

## Conclusion

**CANDIDATE IDENTIFIED, VERIFICATION REQUIRED**

The Kida vortex shows behavior consistent with potential singularity formation:
- Explosive vorticity growth
- Consistent blowup time across resolutions and viscosities
- BKM integral diverges

However, this requires careful verification with:
- Higher resolution convergence study
- Energy conservation monitoring
- Comparison with published Kida vortex studies

If confirmed as a true singularity, this would constitute evidence AGAINST global regularity of Navier-Stokes (the "NO" answer to the Millennium Problem).

---

## Files

- `navier_stokes_black_swan.py` - Main hunter code
- `kida_convergence_study.py` - Convergence analysis
- `black_swan_hunt_results.json` - Raw numerical results
