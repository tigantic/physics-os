# NS2D QTT — Scheduled V&V Scenarios

**Version**: 1.0.0
**Date**: March 3, 2026
**Classification**: PROPRIETARY — Tigantic Holdings LLC
**Framework alignment**: ONTIC_VV_FRAMEWORK.md v1.6.0

---

## Overview

Two formal benchmark scenarios that extend the existing 4-panel evidence
package beyond ultra-short-horizon / same-solver-only evidence, providing:

1. **Long-horizon analytic accuracy** on the production 512² grid
2. **Dense FFT cross-check** against an independent reference solver

Together they close the remaining evidentiary gaps for a commercial-grade
claim: the QTT MG-DC Navier-Stokes 2D solver is accurate, stable, and
reproducible over nontrivial time horizons, and its results agree with an
independent dense discretization of the same PDE to tight tolerances.

---

## Scenario 1 — Long-Horizon Taylor-Green @ 512²

| Field | Value |
|-------|-------|
| **Run ID** | `TG_LH_512_QTT_PROD` |
| **V&V category** | `@benchmark` + `@stress` |
| **Tier** | Tier 1 (analytical solution) |
| **Schedule** | Weekly (release gate candidate) |
| **Satisfies** | SV-1 (convergence), SV-3 (conservation), SV-4 (stability), CV-4 (deterministic) |

### Purpose

Prove the solver stays correct and stable beyond the trivial early-time
window, with meaningful viscous decay (~7.6% enstrophy drop at t=0.05),
while keeping the production Poisson tolerance (1e-3) and the production
MG-DC config.  Forces any truncation-instability to reveal itself over
1049 time steps.

### Configuration (exact — locked)

| Parameter | Value |
|-----------|-------|
| Grid | 512×512 (`n_bits = 9`) |
| Domain | Periodic [0,1]² |
| Viscosity | ν = 0.01 |
| Formulation | Vorticity-streamfunction |
| Time integrator | Explicit Euler |
| dt | 4.76837158203125e-05 (stable CFL dt) |
| t_final | 0.05 |
| n_steps | 1049 (`round(0.05 / dt)`) |
| Poisson | MG-DC, tol=1e-3, 3+3 smooth, 7 levels (9→3 bits), 5 coarse sweeps |
| Rank | 64 |
| Truncation tolerance | 1e-10 |
| Seeds | Run **twice**: seed=0 and seed=42 |

### Initial Condition (exact)

```
ω₀(x, y) = 2·sin(2πx)·sin(2πy)
```

Mean mode set to 0 (zero-mean / null-space handled per existing convention).

### Analytical QoIs

The Taylor-Green vortex with single-mode IC is an exact solution to the
full incompressible Navier-Stokes equations (advection vanishes identically).
The vorticity decays purely by diffusion:

```
ω(x,y,t) = 2·sin(2πx)·sin(2πy)·exp(-8π²νt)
```

**Enstrophy**:
$$E(t) = \tfrac{1}{2}\exp(-16\pi^2 \nu t)$$

**Omega L2 norm**:
$$\|\omega\|_2 = \exp(-8\pi^2 \nu t)$$

**Reference values** (ν = 0.01):

| Time | Enstrophy | ‖ω‖₂ |
|------|-----------|-------|
| t = 0.01 | 0.4921663314346322 | 0.9921354055113971 |
| t = 0.02 | 0.4844553955956484 | 0.9843326628692644 |
| t = 0.05 | 0.46203990564820613 | 0.9612907007229459 |

### Logging Requirements

- **Every step** (or every 10 steps): enstrophy, omega_l2, circulation, div_l2_norm, div_relative_to_vel
- **Poisson per step**: residual, CG iters, warm-start hit/miss
- **Snapshots** saved at: t ∈ {0.01, 0.02, 0.05}

### Pass Criteria

| Check | Metric | Threshold | Severity |
|-------|--------|-----------|----------|
| **QoI accuracy** | `enstrophy_error_rel(t_final)` | ≤ 1e-4 | error |
| **QoI accuracy** | `omega_l2_error_rel(t_final)` | ≤ 1e-4 | error |
| **Constraint** | `div_relative_to_vel(t_final)` | ≤ 1e-3 | error |
| **Stability** | No catastrophic residual growth | monotone or bounded | error |
| **Reproducibility** | `|ΔE|/E` between seeds at t_final | ≤ 1e-6 | error |
| **Reproducibility** | `|Δ‖ω‖₂|/‖ω‖₂` between seeds at t_final | ≤ 1e-6 | error |

### Output Schema

```json
{
  "case_id": "TG_LH_512_QTT_PROD",
  "commit": "<git SHA>",
  "timestamp": "<ISO 8601>",
  "device": "<GPU name>",
  "config": { "n_bits": 9, "dt": 4.76837158203125e-05, "n_steps": 1049, "nu": 0.01, "poisson": "MG-DC", "rank": 64 },
  "seeds": {
    "0": {
      "wall_time_s": 0.0,
      "snapshots": {
        "0.01": { "enstrophy": 0.0, "enstrophy_analytical": 0.0, "enstrophy_error_rel": 0.0, "omega_l2": 0.0, "omega_l2_analytical": 0.0, "omega_l2_error_rel": 0.0 },
        "0.02": { "..." : "..." },
        "0.05": { "..." : "..." }
      },
      "final": {
        "enstrophy": 0.0, "enstrophy_error_rel": 0.0,
        "omega_l2": 0.0, "omega_l2_error_rel": 0.0,
        "div_relative_to_vel": 0.0, "circulation_error": 0.0
      },
      "poisson": { "max_residual": 0.0, "mean_iters": 0.0, "max_iters": 0, "cold_iters": 0 },
      "series": { "enstrophy": [], "omega_l2": [], "poisson_residual": [], "poisson_iters": [] }
    },
    "42": { "..." : "..." }
  },
  "reproducibility": {
    "delta_enstrophy_rel": 0.0,
    "delta_omega_l2_rel": 0.0
  },
  "validation": {
    "passed": true,
    "checks": []
  },
  "claims": [
    { "tag": "CONSERVATION", "claim": "..." },
    { "tag": "STABILITY", "claim": "..." },
    { "tag": "CONVERGENCE", "claim": "..." },
    { "tag": "REPRODUCIBILITY", "claim": "..." }
  ]
}
```

### Run Command

```bash
python3 scripts/run_tg_long_horizon.py
```


---

## Scenario 2 — Dense FFT Cross-Check @ 256²

| Field | Value |
|-------|-------|
| **Run ID** | `TG_XCHECK_256_QTT_vs_DENSEFDFFT` |
| **V&V category** | `@benchmark` (reference comparison) |
| **Tier** | Tier 1 (discrete-exact) |
| **Schedule** | Nightly |
| **Satisfies** | SV-1 (convergence), SV-5 (error estimation) |

### Purpose

Prove the QTT simulation tracks a high-accuracy dense reference for the
same discrete PDE (same grid, same Laplacian stencil), isolating TT
truncation and QTT Poisson floor effects.  Dense reference uses FFT
diagonalization of the 5-point periodic Laplacian — a true "dense gold"
solve that is not "another QTT," avoiding operator mismatch.

### Configuration (exact — locked)

| Parameter | Value |
|-----------|-------|
| Grid | 256×256 (`n_bits = 8`) |
| Domain | Periodic [0,1]² |
| Viscosity | ν = 0.01 |
| Time integrator | Explicit Euler (both QTT and dense) |
| dt | 1.9073486328125e-04 (4× the 512² dt, safe for 256²) |
| t_final | 0.05 |
| n_steps | 262 (`round(0.05 / dt)`) |

**QTT run**: Normal QTT pipeline, Poisson tol=1e-3, rank=64, MG-DC config as above.

**Dense reference**: Dense float64 arrays, same discrete operators, Poisson
solved via FFT diagonalization (exact for the discrete Laplacian).

### Initial Condition (exact)

Same Taylor-Green:
```
ω₀(x, y) = 2·sin(2πx)·sin(2πy)
```

### Dense FFT Poisson (exact specification)

Standard 5-point periodic Laplacian:

$$(Δ_h ψ)_{i,j} = \frac{ψ_{i+1,j} + ψ_{i-1,j} + ψ_{i,j+1} + ψ_{i,j-1} - 4ψ_{i,j}}{h^2}$$

In Fourier, the eigenvalue is:

$$λ(k_x, k_y) = \frac{2\cos(2πk_x/N) + 2\cos(2πk_y/N) - 4}{h^2}$$

Solve procedure:
1. Compute $\hat{ω}(k)$ via FFT
2. Set $\hat{ψ}(0,0) = 0$ (zero-mean)
3. For other modes: $\hat{ψ}(k) = \hat{ω}(k) / λ(k)$
4. Inverse FFT to get ψ

This is the exact discrete Poisson inverse for the same 5-point stencil
the QTT solver approximates.

### Comparison Metrics

At t_final (and optionally at intermediate checkpoints):

| Metric | Formula |
|--------|---------|
| **Field error** | `rel_L2(ω) = ‖ω_qtt − ω_dense‖₂ / ‖ω_dense‖₂` |
| **Enstrophy diff** | `|E_qtt − E_dense| / E_dense` |
| **Omega L2 diff** | `|‖ω‖₂_qtt − ‖ω‖₂_dense| / ‖ω‖₂_dense` |
| **Divergence** | QTT's div_relative_to_vel (dense is ~machine ε) |

### Pass Criteria

| Check | Metric | Threshold | Severity |
|-------|--------|-----------|----------|
| **Field agreement** | `rel_L2(omega)(t_final)` | ≤ 5e-3 | error |
| **Enstrophy match** | `|E_qtt − E_dense| / E_dense` | ≤ 1e-4 | error |
| **Omega L2 match** | `‖ω‖₂` relative difference | ≤ 1e-4 | error |

### Output Schema

```json
{
  "case_id": "TG_XCHECK_256_QTT_vs_DENSEFDFFT",
  "commit": "<git SHA>",
  "timestamp": "<ISO 8601>",
  "device": "<GPU name>",
  "config": { "n_bits": 8, "dt": 1.9073486328125e-04, "n_steps": 262, "nu": 0.01 },
  "qtt": {
    "wall_time_s": 0.0,
    "enstrophy": 0.0, "omega_l2": 0.0, "div_relative_to_vel": 0.0,
    "poisson": { "max_residual": 0.0, "mean_iters": 0.0, "max_iters": 0 }
  },
  "dense": {
    "wall_time_s": 0.0,
    "enstrophy": 0.0, "omega_l2": 0.0
  },
  "comparison": {
    "omega_rel_l2": 0.0,
    "enstrophy_rel_diff": 0.0,
    "omega_l2_rel_diff": 0.0
  },
  "analytical": {
    "enstrophy": 0.0, "omega_l2": 0.0,
    "qtt_enstrophy_error_rel": 0.0,
    "dense_enstrophy_error_rel": 0.0
  },
  "validation": {
    "passed": true,
    "checks": []
  },
  "claims": [
    { "tag": "CONVERGENCE", "claim": "..." }
  ]
}
```

### Run Command

```bash
python3 scripts/run_tg_fft_crosscheck.py
```


---

## CI/CD Integration

### Schedule Mapping

| Scenario | Gate | Trigger | Expected runtime |
|----------|------|---------|-----------------|
| TG_XCHECK_256 (**Scenario 2**) | Nightly | Cron / regression | ~5-10 min |
| TG_LH_512 (**Scenario 1**) | Weekly | Cron / release gate | ~60-90 min |

### Regression Thresholds (for `detect_vv_regression.py`)

| Case ID | Metric | Threshold | Direction |
|---------|--------|-----------|-----------|
| `TG_LH_512_QTT_PROD` | `enstrophy_error_rel` | 1e-4 | must decrease or stay |
| `TG_LH_512_QTT_PROD` | `omega_l2_error_rel` | 1e-4 | must decrease or stay |
| `TG_LH_512_QTT_PROD` | `wall_time_s` | +20% of rolling baseline | performance |
| `TG_XCHECK_256_QTT_vs_DENSEFDFFT` | `omega_rel_l2` | 5e-3 | must not increase |
| `TG_XCHECK_256_QTT_vs_DENSEFDFFT` | `enstrophy_rel_diff` | 1e-4 | must not increase |

### Claim Tags

These scenarios emit the following provenance claim tags:

| Tag | Source |
|-----|--------|
| `CONSERVATION` | Circulation / divergence checks |
| `STABILITY` | No blowup / bounded residual drift |
| `CONVERGENCE` | Analytic QoI accuracy, field agreement |
| `REPRODUCIBILITY` | Seed-to-seed QoI delta |
| `PERFORMANCE` | Wall time within budget |

### V&V Harness Integration

Both scenarios are registered in `ontic/sim/validation/ns2d_evidence.py`
as formal `VVTest` cases in the NS2D evidence plan:

```python
from ontic.sim.validation import build_ns2d_evidence_plan, run_vv_plan

plan = build_ns2d_evidence_plan()
# Plan includes panels 1-4 (short-horizon) + panels 5-6 (long-horizon, cross-check)
all_passed, report = run_vv_plan(plan, output_path="ns2d_vv_report.md")
```

---

## Existing MG Tuning Artifacts

The `vv_coldstart_*_9_10_11.json` files in the repo root serve as
**Performance / Solver Health** lineage.  They are treated as performance
regression cases with thresholds on time-to-cold-converge, residual floor,
and variance.  These complement the two scenarios above — the MG tuning
files track solver internals; the scenarios track physics correctness.

---

## Commercial Narrative

If both scenarios pass:

> "Analytic benchmark accuracy holds over a nontrivial horizon on 512²."
>
> "Dense cross-check at 256² matches the same discrete PDE within tight tolerances."
>
> "Poisson residual floor exists, but it does not degrade QoIs in the production tolerance regime."

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-03 | Tigantic Holdings LLC | Initial scenario specification |
