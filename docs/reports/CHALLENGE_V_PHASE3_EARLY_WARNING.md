# Challenge V · Phase 3 — Real-Time Early Warning

**Date:** 2026-02-28 08:06 UTC
**Vessels:** 500
**Ports:** 50, Routes: 80
**Scenarios:** 3
**Wall time:** 0.2 s

## Detection Performance

| Metric | Value |
|--------|-------|
| TP | 59 |
| FP | 267 |
| TN | 10927 |
| FN | 0 |
| Precision | 0.1810 |
| Recall | 1.0000 |
| FPR | 0.023852 |
| Lead time | 3.6 h |

## Propagation Predictions

| Scenario | Ports | Queue (TEU) | Depth | Impact ($) | Accuracy |
|----------|:-----:|:-----------:|:-----:|:----------:|:--------:|
| Suez Canal Blockage | 4 | 9,603 | 0 | $25,621,396 | 0.996 |
| Shanghai Lockdown | 10 | 21,814 | 5 | $156,608,931 | 0.838 |
| US West Coast Labor | 3 | 1,940 | 0 | $5,319,130 | 0.920 |

## Rerouting Recommendations

| Scenario | Vessels | Cost (%) | Time (d) | TEU Saved |
|----------|:-------:|:--------:|:--------:|:---------:|
| Suez Canal Blockage | 75 | +42.3% | +34.9 | 591,005 |
| Shanghai Lockdown | 98 | +37.4% | +63.2 | 823,981 |
| US West Coast Labor | 71 | +2.9% | +4.1 | 574,895 |

**QTT compression:** 2.9×
