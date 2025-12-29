# Experiment Log

## NS-Millennium Research

Chronological record of all experiments, observations, and results.

---

## Log Format

Each entry follows the structure:

```
### [EXPERIMENT-XXX] Title
**Date:** YYYY-MM-DD  
**Phase:** [PHASE-N]  
**Investigator:** Name  

**Objective:**  
**Configuration:**  
**Results:**  
**Conclusions:**  
**Artifacts:**  
```

---

## Experiments

### [EXPERIMENT-001] 2D Incompressible NS Solver Development
**Date:** 2025-12-22  
**Phase:** [PHASE-1A]  
**Investigator:** HyperTensor Team  

**Objective:**  
Build and validate 2D incompressible Navier-Stokes solver using spectral methods with projection (Chorin-Temam) for exact incompressibility.

**Configuration:**  
- Domain: [0, 2π]² with periodic BC
- Discretization: Spectral (FFT-based)
- Time integration: Forward Euler + projection
- Test case: Taylor-Green vortex, ν=0.1, t∈[0,1]
- Resolution: N=64
- CFL: 0.2 (for stability)

**Results:**  
1. **Poisson Solver**
   - Self-consistency error: ~10⁻¹⁵ (machine precision)
   - Convergence order: 4.24× per doubling (expected 4.0 for O(dx²))
   
2. **Projection**
   - Divergence reduction: 10¹² (from ~6 to ~10⁻¹²)
   - Machine-precision incompressibility achieved

3. **Taylor-Green Benchmark**
   - Energy decay rate error: 0.02% (gate: <5%)
   - Velocity error: 7.89×10⁻⁵
   - Final divergence: 8.77×10⁻¹⁵

**Conclusions:**  
Phase 1a gate criteria satisfied:
- ✓ Poisson: O(dx²) convergence, self-consistency
- ✓ Projection: divergence < 10⁻⁶
- ✓ Taylor-Green: decay error < 5%

The spectral projection method provides EXACT incompressibility, eliminating RISK-R8 (divergence contamination of χ signal).

**Artifacts:**  
- `tensornet/cfd/tt_poisson.py` - Poisson solver and spectral operators
- `tensornet/cfd/ns_2d.py` - 2D NS solver
- `proofs/proof_phase_1a.py` - Comprehensive Phase 1a proofs
- `proofs/proof_phase_1a_result.json` - Proof results

---

## Quick Reference

| ID | Date | Phase | Summary | Outcome |
|----|------|-------|---------|---------|
| 001 | 2025-12-22 | 1a | 2D NS solver with Taylor-Green | ✓ PASS |

