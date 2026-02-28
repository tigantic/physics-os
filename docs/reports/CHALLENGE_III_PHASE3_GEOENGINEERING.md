# Challenge III · Phase 3 — Geoengineering Intervention Modeling

**Date:** 2026-02-28 08:02 UTC
**Configs:** 120 SAI scenarios
**Ensemble:** 50 members per config
**Regions:** 8
**Wall time:** 0.1 s

## Exit Criteria

- Configs ≥ 100: **PASS** (120)
- Regions ≥ 6: **PASS** (8)
- QTT ≥ 2×: **PASS** (24.1×)

## Best Configuration

- **Config ID:** 5
- **Site:** Equatorial Pacific (0.0°N, -160.0°E)
- **Particle:** SO2_fine (r=0.3 µm)
- **Rate:** 20.0 Tg/yr
- **ΔT global:** -4.624 K
- **RF:** -4.75 W/m²
- **Residence:** 9.4 months

## Top 10 Scenarios by Cooling

| ID | Site | Particle | Rate | ΔT (K) | σ(ΔT) | RF (W/m²) |
|:--:|------|----------|:----:|:------:|:-----:|:---------:|
| 5 | Equatorial Pacific | SO2_fine | 20.0 | -4.624 | 1.180 | -4.75 |
| 77 | Arctic | SO2_fine | 20.0 | -4.568 | 1.173 | -4.85 |
| 101 | Tropical Atlantic | SO2_fine | 20.0 | -4.439 | 1.073 | -4.92 |
| 53 | North Atlantic | SO2_fine | 20.0 | -4.422 | 1.127 | -4.75 |
| 29 | Southern Indian Ocean | SO2_fine | 20.0 | -4.400 | 1.340 | -4.71 |
| 28 | Southern Indian Ocean | SO2_fine | 12.0 | -2.852 | 0.648 | -2.93 |
| 52 | North Atlantic | SO2_fine | 12.0 | -2.721 | 0.605 | -2.81 |
| 4 | Equatorial Pacific | SO2_fine | 12.0 | -2.702 | 0.699 | -2.83 |
| 76 | Arctic | SO2_fine | 12.0 | -2.617 | 0.724 | -2.92 |
| 100 | Tropical Atlantic | SO2_fine | 12.0 | -2.603 | 0.693 | -2.77 |

## Regional Impacts (Best Config)

| Region | ΔT (K) | ΔP (%) | Crop (%) | UV (%) | O₃ (%) |
|--------|:------:|:------:|:--------:|:------:|:------:|
| North America | -6.147 | 6.17 | -10.17 | 5.24 | 2.62 |
| Europe | -6.400 | 8.98 | -7.20 | 5.43 | 2.72 |
| South Asia | -5.278 | 10.54 | -5.34 | 4.93 | 2.47 |
| Sub-Saharan Africa | -4.400 | 13.66 | -2.88 | 4.78 | 2.39 |
| South America | -4.991 | 11.03 | -3.82 | 4.97 | 2.49 |
| East Asia | -5.696 | 10.44 | -7.21 | 5.20 | 2.60 |
| Australia | -5.477 | 10.07 | -7.13 | 5.05 | 2.52 |
| Arctic | -7.344 | 10.20 | -9.94 | 5.82 | 2.91 |

**QTT compression:** 24.1×
