# Challenge IV · Phase 3 — Reactor Design Optimization

**Date:** 2026-02-28 08:05 UTC
**Configurations:** 10,000
**Feasible:** 3,936
**Stable:** 3,875
**Q > 10:** 405
**Wall time:** 0.1 s

## Exit Criteria

- Configs ≥ 10K: **PASS** (10,000)
- Optimal Q > 10: **PASS** (Q=26.22)
- Material filter: **PASS** (3,936/10,000 feasible)
- Stability map: **PASS** (3,875/3,936 stable)
- QTT ≥ 2×: **PASS** (204.8×)

## Optimal Design

| Parameter | Value |
|-----------|-------|
| Major radius R | 3.431 m |
| Minor radius a | 1.144 m |
| Aspect ratio | 3.00 |
| Toroidal field B_T | 12.06 T |
| Elongation κ | 2.195 |
| Triangularity δ | 0.581 |
| Plasma current I_p | 16.83 MA |
| **Q factor** | **26.22** |
| Fusion power | 1311.2 MW |
| Stored energy | 291.9 MJ |
| τ_E (IPB98) | 0.981 s |
| q₉₅ | 27.13 |
| β_N | 1.417 |
| Wall heat flux | 1.29 MW/m² |
| Neutron wall load | 3.97 MW/m² |
| Coil stress | 397 MPa |
| Cost index | 0.938 |

## Pareto Front (Q vs Cost) — Top 10

| ID | R (m) | a (m) | B (T) | Q | P_fus (MW) | Cost |
|:--:|:-----:|:-----:|:-----:|:--:|:----------:|:----:|
| 1758 | 3.43 | 1.14 | 12.1 | 26.2 | 1311 | 0.938 |
| 7995 | 3.44 | 1.15 | 10.7 | 23.6 | 1182 | 0.752 |
| 3432 | 3.33 | 1.15 | 10.1 | 22.9 | 1143 | 0.640 |
| 7202 | 3.29 | 1.09 | 10.2 | 22.3 | 1117 | 0.583 |
| 7996 | 3.28 | 0.99 | 10.7 | 19.5 | 974 | 0.533 |
| 6699 | 2.58 | 1.17 | 10.1 | 18.5 | 923 | 0.518 |
| 8189 | 3.39 | 0.85 | 10.7 | 18.0 | 900 | 0.405 |
| 1937 | 2.92 | 0.88 | 10.9 | 16.9 | 847 | 0.378 |
| 9922 | 3.46 | 0.79 | 10.7 | 14.1 | 704 | 0.356 |
| 8489 | 2.89 | 0.76 | 10.1 | 13.6 | 678 | 0.248 |

**QTT compression:** 204.8×
