# Challenge VI Phase 2: Video Temporal Consistency — Report

**Generated:** 2026-02-28T07:38:25.756325+00:00
**Pipeline time:** 124.1 s

## Configuration

- **FPS:** 30
- **Frames per clip:** 90
- **Image size:** 128×128
- **Authentic clips:** 50
- **Manipulated clips:** 50

## Temporal Physics Tests

| Test | AUC | Accuracy | TP | FP | TN | FN | QTT | Pass |
|------|:---:|:--------:|:--:|:--:|:--:|:--:|:---:|:----:|
| Temporal Lighting Consistency | 1.000 | 1.000 | 50 | 0 | 50 | 0 | 45.2× | ✅ |
| Motion Blur Consistency | 1.000 | 1.000 | 50 | 0 | 50 | 0 | 45.2× | ✅ |
| Depth-of-Field Coherence | 1.000 | 1.000 | 50 | 0 | 50 | 0 | 45.2× | ✅ |
| Reflection Dynamics | 1.000 | 1.000 | 50 | 0 | 50 | 0 | 45.2× | ✅ |
| Audio-Visual Alignment | 0.933 | 0.920 | 50 | 8 | 42 | 0 | 45.2× | ✅ |

## Exit Criteria

- Combined AUC > 0.93: ✅ (0.987)
- All tests AUC > 0.88: ✅ (5/5)
- QTT temporal compression: ✅
- **Overall: PASS ✅**

---
*Challenge VI Phase 2 — Video Temporal Consistency*
*© 2026 Tigantic Holdings LLC*