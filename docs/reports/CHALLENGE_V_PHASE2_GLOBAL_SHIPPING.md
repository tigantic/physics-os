# Challenge V Phase 2: Global Shipping Network — Report

**Generated:** 2026-02-28T07:19:59.727876+00:00
**Pipeline time:** 15.7 s

## Network Topology

- **Routes:** 50
- **Ports:** 99
- **Cells per route:** 64
- **Simulation:** 90 days at 0.5 day steps

## Disruption Scenarios

| Scenario | Peak Queue (TEU) | Port | Duration (days) | Impact ($B) | Cascade | Pass |
|----------|:-----------------:|------|:---------------:|:-----------:|:-------:|:----:|
| Suez Canal Blockage (2021) | 83,200 | Port Said | 46 | 10.13 | — | ✅ |
| Red Sea Crisis (2023-2024) | 597,740 | Port Said | 68 | 60.63 | — | ✅ |

## Multi-Commodity Flow

- **Container throughput:** 2,029,914 TEU
- **Bulk throughput:** 12,932,482 tonnes
- **Tanker throughput:** 59,739,020 barrels
- **Modal split factor:** 1.63

## Exit Criteria

- Suez blockage reproduced: ✅
- Red Sea rerouting validated: ✅
- Multi-commodity flow: ✅
- QTT compression: ✅ (0.5×)
- **Overall: PASS ✅**

---
*Challenge V Phase 2 — Global Shipping Network*
*© 2026 Tigantic Holdings LLC*