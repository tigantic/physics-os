# Rank Atlas v1.0 — HyperTensor Reference Baselines

## Baseline A: In-Solver QTT (Rank Atlas Campaign)

These results are from the Rank Atlas Campaign executed with The Physics OS's
native QTT solver infrastructure. They measure the bond dimension of QTT
states *during and after* solver time-integration with SVD truncation.

| Dataset | Measurements | Packs | n_bits Range | Source |
|---------|-------------|-------|-------------|--------|
| Pilot campaign | 352 | 20/20 | 4–7 | `rank_atlas_20pack.json` |
| Deep sweep (III, VI) | 162 | 2 | 4–9 | `rank_atlas_deep_III_VI.json` |
| Protocol expansion (V0.4) | 180 | 6 | 10 | `data/rank_atlas_v04_nbits10.json` |
| Protocol expansion (V0.2) | 42 | 14 | 10 | `data/rank_atlas_v02_nbits10.json` |
| Pack VI high-res | 15 | 1 | 8–12 | `data/rank_atlas_pack_vi_highres.json` |
| **Total** | **751+** | **20/20** | **4–12** | |

### Summary Results

- **All 20 packs measured** with zero failures.
- **All V0.4 packs Class A** (|α| < 0.1).
- **Gap statistic selects k = 1** (universality supported).
- **Grid independence: 16/20** strict, ≥18/20 lenient.
- **Pack VI saturates at χ = 25** for n_bits ≥ 11.

## Baseline B: Dense-to-QTT Encode (Dual Measurement)

These results measure *intrinsic* field compressibility by solving each
problem with a dense NumPy solver, then compressing the final field to QTT
via standalone TT-SVD.

| Dataset | Configs | Domains | n_bits Range | Source |
|---------|---------|---------|-------------|--------|
| Dual-measurement protocol | 20 matched | 4 | 6–14 | `data/dual_measurement_protocol.json` |
| Fixed-T supplementary | 3 | 1 | 6, 8, 10 | (same file) |

### Summary Results

- **Path A ≥ Path B in 20/20 configurations** (0 B_HIGHER).
- **Intrinsic rank χ_B ≤ 8** across all resolutions.
- **VM never artificially deflates rank** — Path A is a conservative upper bound.

## Reproducing Baselines

```bash
# Full 20-pack pilot campaign (n_bits 4–7, ~30 min)
python scripts/research/rank_atlas_campaign.py \
    --packs ALL --n-bits 4 5 6 7 --output baselines_pilot.json

# Protocol-compliant expansion (n_bits 10, ~2 hours)
python scripts/research/rank_atlas_campaign.py \
    --packs ALL --n-bits 10 --output baselines_nbits10.json

# Dual-measurement validation
python scripts/research/dual_measurement_protocol.py

# Validate results
python benchmarks/rank_atlas/validate.py baselines_pilot.json
```

## Hardware Used for Reference Baselines

- **GPU:** NVIDIA GeForce RTX 5070 Laptop GPU (7.96 GB VRAM, CC 12.0)
- **CUDA:** 12.8
- **Software:** Python 3.12.3, PyTorch 2.9.1+cu128, NumPy 2.2.3
- **OS:** Ubuntu 24.04 (WSL2)
