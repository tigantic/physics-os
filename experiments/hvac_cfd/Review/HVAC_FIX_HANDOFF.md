# HVAC CFD Nielsen Benchmark Fix — Handoff to Opus

## Executive Summary

The Nielsen benchmark is failing (27% RMS error vs <10% target) because **first-order upwind advection is killing the jet**.

**Root Cause:** Numerical viscosity from upwind is 535× larger than physical viscosity.

**Fix:** Replace upwind with central differences (or skew-symmetric form).

---

## The Math

At Re=5000 with grid dx = 0.035 m:

| Property | Value |
|----------|-------|
| Physical viscosity | ν = 1.5×10⁻⁵ m²/s |
| Upwind numerical viscosity | ν_num ≈ U·Δx/2 ≈ 8×10⁻³ m²/s |
| **Ratio** | **535×** |

The jet can't survive 535× more diffusion than physics dictates.

---

## Why HyperTensor's Other Solvers Work

The working `ns_2d.py` and `tt_poisson.py` solvers use **spectral (FFT) derivatives**:

```python
# From tt_poisson.py - compute_gradient_2d()
phi_hat = torch.fft.fft2(phi)
dphi_dx = torch.fft.ifft2(1j * KX * phi_hat).real
dphi_dy = torch.fft.ifft2(1j * KY * phi_hat).real
```

Spectral methods have **zero numerical diffusion**. That's why Taylor-Green vortex and other benchmarks pass.

---

## Why Spectral Won't Work for Nielsen

Spectral requires **periodic boundary conditions**. Nielsen has:
- Walls (no-slip)
- Inlet (Dirichlet)
- Outlet (Neumann/convective)

Can't use FFT. Need finite differences.

---

## The Fix

Replace upwind advection in `projection_solver.py`:

### OLD (line ~85, compute_advection method):
```python
# First-order upwind — HIGH numerical diffusion
dudx = torch.where(u_int > 0, dudx_backward, dudx_forward)
```

### NEW:
```python
# Central differences — ZERO numerical diffusion
du_dx = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dx)
```

Or better, use **skew-symmetric form** (energy-conserving):
```python
# Skew-symmetric = 0.5 * (convective + conservative)
# Conserves kinetic energy exactly, prevents artificial growth
adv_u = 0.5 * (conv_u + cons_u)  # See projection_solver_fixed.py
```

---

## Stability Consideration

| Scheme | Stability | Diffusion |
|--------|-----------|-----------|
| Upwind | Unconditional | HIGH (bad) |
| Central | CFL < 1 | Zero (good) |
| Skew-symmetric | CFL < 1 | Zero + energy conserving |

Central/skew-symmetric need smaller timestep (already handled in fixed solver with `dt_safety=0.25`).

---

## Files

1. **`/home/claude/projection_solver_fixed.py`** — Complete fixed solver
   - Central difference advection
   - Skew-symmetric option (recommended)
   - Ready to drop into tensornet/hvac/

2. **`/home/claude/advection_schemes.py`** — Comparison of schemes
   - upwind, central, skew_symmetric, hybrid, QUICK
   - Numerical viscosity calculations

---

## Expected Results After Fix

| Metric | Before (upwind) | After (central/skew) |
|--------|-----------------|----------------------|
| RMS at x/H=1.0 | 32% | <10% |
| RMS at x/H=2.0 | 20% | <10% |
| Jet behavior | Dies immediately | Persists along ceiling |
| Convergence | Diverging | Converging |

---

## Test Command

After copying fixed solver:
```bash
cd /path/to/hypertensor
python -m tensornet.hvac.nielsen --re 5000 --nx 256 --ny 128
```

---

## References

1. **Morinishi et al. (1998)** "Fully conservative higher order finite difference schemes for incompressible flow" — JCP 143:90-124 — Skew-symmetric formulation

2. **HyperTensor tt_poisson.py** — Working spectral implementation with projection method

3. **Nielsen benchmark** — Aalborg University IEA Annex 20

---

## Summary for Opus

**Do this:**
1. Replace `compute_advection()` in projection_solver.py with central or skew-symmetric
2. Lower `dt_safety` to 0.25
3. Re-run Nielsen benchmark
4. Expect <10% RMS error

**Don't do this:**
- Add turbulence modeling (not needed for this fix)
- Increase grid resolution (not the problem)
- Change pressure solver (not the problem)

The physics is fine. The numerics were wrong. Central differences fix it.
