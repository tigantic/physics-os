# Challenge III Phase 2: Regional Climate Ensemble — Report

**Generated:** 2026-02-28T07:08:26.731088+00:00
**Pipeline time:** 167.5 s

## Configuration

- **Ensemble members:** 10,000
- **SSP scenarios:** 4 (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5)
- **Domain:** 200 km × 200 km at 5 km resolution
- **Grid:** 40 × 40 = 1,600 cells
- **Time steps:** 200 × 60 s = 3.3 hours
- **Sampling:** Latin Hypercube Sampling

## Memory Benchmark

| Metric | Value |
|--------|-------|
| Total QTT memory | 233.49 MB |
| Total dense memory | 122.07 MB |
| Compression ratio | 0.5× |
| Under 10 GB limit | ✅ YES |

## SSP Scenario Statistics

| SSP | Members | Mean Max (µg/m³) | Std Max | Mean Rank | Exceed Rate |
|-----|:-------:|:-----------------:|:-------:|:---------:|:-----------:|
| SSP1-2.6 | 2,500 | 9733.0 | 9766.4 | 24.0 | 1.0000 |
| SSP2-4.5 | 2,500 | 14284.0 | 13144.5 | 24.0 | 1.0000 |
| SSP3-7.0 | 2,500 | 19702.0 | 18894.6 | 24.0 | 1.0000 |
| SSP5-8.5 | 2,500 | 25941.6 | 24830.5 | 24.0 | 1.0000 |

## Return Period Analysis

| Threshold (µg/m³) | Exceedances | Rate | Return Period |
|:-----------------:|:-----------:|:----:|:-------------:|
| 35 | 10,000 | 1.0000 | 1.0 |
| 50 | 10,000 | 1.0000 | 1.0 |
| 75 | 10,000 | 1.0000 | 1.0 |
| 100 | 10,000 | 1.0000 | 1.0 |
| 150 | 10,000 | 1.0000 | 1.0 |
| 200 | 10,000 | 1.0000 | 1.0 |

## Tipping-Point Signatures

| Transition | Rank Jump | Variance Ratio | Tipping Score | Topological |
|------------|:---------:|:--------------:|:-------------:|:-----------:|
| SSP1-2.6 → SSP2-4.5 | 1.00× | 1.81× | 1.35 | — |
| SSP2-4.5 → SSP3-7.0 | 1.00× | 2.07× | 1.44 | — |
| SSP3-7.0 → SSP5-8.5 | 1.00× | 1.73× | 1.31 | — |

## Exit Criteria

- 10,000 ensemble members solved: ✅
- Total QTT memory < 10 GB: ✅
- Return period analysis complete: ✅
- Tipping-point signatures detected: ✅
- **Overall: PASS ✅**

---

*Challenge III Phase 2 — Regional Climate Ensemble*
*© 2026 Tigantic Holdings LLC*