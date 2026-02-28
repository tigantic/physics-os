# Challenge VI · Phase 3 — Real-Time Verification Pipeline

**Date:** 2026-02-28 08:07 UTC
**Streams:** 20 (10 auth + 10 manip)
**Frames:** 6,000
**Physics tests:** 5
**Wall time:** 8.1 s

## Exit Criteria

- FPS ≥ 30: **PASS** (926 fps mean, 873 fps min)
- AUC > 0.90: **PASS** (0.9800)
- Tests ≥ 5: **PASS** (5)
- QTT ≥ 2×: **PASS** (3.6×)

## Throughput

| Stream | Auth? | Frames | Mean (ms) | Max (ms) | FPS |
|:------:|:-----:|:------:|:---------:|:--------:|:---:|
| 0 | ✓ | 300 | 1.08 | 3.56 | 927 |
| 1 | ✓ | 300 | 1.05 | 1.71 | 952 |
| 2 | ✓ | 300 | 1.08 | 1.78 | 929 |
| 3 | ✓ | 300 | 1.09 | 1.95 | 916 |
| 4 | ✓ | 300 | 1.08 | 1.48 | 923 |
| 5 | ✓ | 300 | 1.06 | 1.79 | 940 |
| 6 | ✓ | 300 | 1.05 | 1.49 | 950 |
| 7 | ✓ | 300 | 1.15 | 29.99 | 873 |
| 8 | ✓ | 300 | 1.08 | 1.52 | 929 |
| 9 | ✓ | 300 | 1.08 | 1.76 | 925 |
| 10 | ✗ | 300 | 1.08 | 1.61 | 923 |
| 11 | ✗ | 300 | 1.07 | 1.67 | 937 |
| 12 | ✗ | 300 | 1.08 | 1.51 | 926 |
| 13 | ✗ | 300 | 1.11 | 2.02 | 897 |
| 14 | ✗ | 300 | 1.06 | 1.77 | 940 |
| 15 | ✗ | 300 | 1.09 | 2.33 | 920 |
| 16 | ✗ | 300 | 1.08 | 2.13 | 928 |
| 17 | ✗ | 300 | 1.07 | 1.63 | 936 |
| 18 | ✗ | 300 | 1.07 | 1.69 | 933 |
| 19 | ✗ | 300 | 1.08 | 1.53 | 922 |

## Confidence Scores

| Stream | Auth? | Confidence | Peak Anomaly |
|:------:|:-----:|:----------:|:------------:|
| 0 | ✓ | 0.7769 | 1.0000 |
| 1 | ✓ | 0.7180 | 1.0000 |
| 2 | ✓ | 0.8209 | 1.0000 |
| 3 | ✓ | 0.7829 | 1.0000 |
| 4 | ✓ | 0.7541 | 1.0000 |
| 5 | ✓ | 0.7633 | 1.0000 |
| 6 | ✓ | 0.7531 | 1.0000 |
| 7 | ✓ | 0.7222 | 1.0000 |
| 8 | ✓ | 0.7665 | 1.0000 |
| 9 | ✓ | 0.7888 | 1.0000 |
| 10 | ✗ | 0.6977 | 1.0000 |
| 11 | ✗ | 0.7378 | 1.0000 |
| 12 | ✗ | 0.6809 | 1.0000 |
| 13 | ✗ | 0.6812 | 1.0000 |
| 14 | ✗ | 0.6807 | 1.0000 |
| 15 | ✗ | 0.6843 | 1.0000 |
| 16 | ✗ | 0.6890 | 1.0000 |
| 17 | ✗ | 0.6759 | 1.0000 |
| 18 | ✗ | 0.6954 | 1.0000 |
| 19 | ✗ | 0.7118 | 1.0000 |

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/verify/frame` | Submit a single frame for physics verification |
| POST | `/api/v1/verify/stream` | Submit a video stream for continuous analysis |
| GET | `/api/v1/verify/status/{job_id}` | Check status of an ongoing verification job |
| GET | `/api/v1/verify/report/{job_id}` | Download the verification report with heatmaps |
| POST | `/api/v1/verify/batch` | Submit multiple videos for batch processing |

**QTT compression:** 3.6×
