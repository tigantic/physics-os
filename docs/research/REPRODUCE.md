# Reproducing the χ-Regularity Evidence Bundle

This document provides exact instructions to reproduce all measurements
reported in the χ-Regularity Conjecture document (v2.3.1).

---

## Prerequisites

### Hardware

| Component       | Tested Configuration                             |
|-----------------|--------------------------------------------------|
| GPU             | NVIDIA GeForce RTX 5070 Laptop GPU (7.96 GB VRAM)|
| Compute Cap.    | 12.0                                             |
| CPU / RAM       | Any x86-64 with ≥ 16 GB                         |
| Disk            | ≥ 2 GB free for outputs                         |

The campaign is GPU-accelerated but will fall back to CPU if no CUDA device
is available.  Expect ~10× longer wall-clock on CPU.

### Software

| Package    | Version          | Install                              |
|------------|------------------|--------------------------------------|
| Python     | 3.12.3           | System / pyenv                       |
| PyTorch    | 2.9.1+cu128      | `pip install torch==2.9.1+cu128`     |
| CUDA       | 12.8             | NVIDIA driver ≥ 570                  |
| NumPy      | ≥ 1.26           | `pip install numpy`                  |
| SciPy      | ≥ 1.12           | `pip install scipy`                  |
| Pandas     | ≥ 2.1            | `pip install pandas`                 |
| Matplotlib | ≥ 3.8            | `pip install matplotlib`             |
| PyArrow    | ≥ 15.0           | `pip install pyarrow`                |

```bash
pip install -r requirements-lock.txt   # or
pip install torch numpy scipy pandas matplotlib pyarrow
```

---

## Repository Setup

```bash
git clone https://github.com/<org>/HyperTensor-VM.git
cd HyperTensor-VM
git checkout 799102423d88619a0c649cf4ab5c9a34b204269c
```

Verify the campaign script hash:

```bash
sha256sum scripts/research/rank_atlas_campaign.py
# expected: b5fbcc42ca2411c03032da704c32e7fd519a0d5b8403a356b655b0a77245b3e2
```

---

## Step 1 — Full 20-Pack Campaign (352 measurements)

```bash
python tools/scripts/research/rank_atlas_campaign.py \
    --packs ALL \
    --n-bits 4 5 6 7 \
    --output-json data/rank_atlas_20pack.json \
    --output-parquet data/rank_atlas_20pack.parquet \
    --atlas-dir data/atlas_results_20pack \
    --device cuda
```

### Expected Outputs

| Artifact                                     | Rows | Notes                            |
|----------------------------------------------|------|----------------------------------|
| `data/rank_atlas_20pack.json`                | 352  | One row per (pack, ξ, n_bits)    |
| `data/rank_atlas_20pack.parquet`             | 352  | Same data, columnar format       |
| `data/atlas_results_20pack/ATLAS_SUMMARY.md` | —    | Human-readable summary           |
| `data/atlas_results_20pack/scaling_classes.png` | —  | Scaling-class scatter plot       |
| `data/atlas_results_20pack/alpha_exponents.png` | —  | α-exponent distribution          |

### Expected Verdict

```
VERDICT: CONFIRMED — 20/20 packs pass lenient χ-bound (B-threshold < 2.0)
         352 measurements, 0 failures
```

If any pack **fails** (i.e., exhibits Class D rank divergence), the script
prints `VERDICT: FALSIFIED` and the specific pack+config.

### Approximate Wall-Clock

~12–18 minutes on the reference GPU.

---

## Step 2 — Deep Sweep on Packs III & VI (162 measurements)

```bash
python tools/scripts/research/rank_atlas_campaign.py \
    --packs III VI \
    --n-bits 4 5 6 7 8 9 \
    --output-json data/rank_atlas_deep_III_VI.json \
    --output-parquet data/rank_atlas_deep_III_VI.parquet \
    --atlas-dir data/atlas_results_deep_III_VI \
    --device cuda
```

### Expected Outputs

| Artifact                                          | Rows | Notes                    |
|---------------------------------------------------|------|--------------------------|
| `data/rank_atlas_deep_III_VI.json`                | 162  | Extended n_bits range    |
| `data/rank_atlas_deep_III_VI.parquet`             | 162  |                          |
| `data/atlas_results_deep_III_VI/ATLAS_SUMMARY.md` | —    |                          |

### Approximate Wall-Clock

~8–12 minutes on the reference GPU.

---

## Step 3 — Verify Artifact Integrity

From the repository root:

```bash
sha256sum -c docs/research/SHA256SUMS.txt
```

All lines should print `OK`.  The two `SELF`-marked files
(`chi_regularity_hypothesis.md`, `evidence_manifest.json`) are excluded from
the checksum file because their hashes change on every edit.

---

## Interpreting Results

The campaign script computes per-pack statistics automatically. Key metrics:

- **Scaling class** (`A` / `B` / `C` / `D`): see §3.5 of the hypothesis document.
  A–C indicate bounded or slowly growing rank; D indicates divergence (falsification).
- **|b|/a ratio**: grid-independence metric.  < 0.05 = pass, ≥ 0.05 = fail.
- **α exponent**: power-law fit χ ~ ξ^α.  α < 0.1 expected for smooth solutions.
- **q statistic**: median-normalized complexity stress measure.  q < 2.0 = within bound.

### Claim Ledger

See Appendix C of the hypothesis document and `docs/research/evidence_manifest.json`
for the complete mapping from claims to artifacts and acceptance criteria.

---

## Troubleshooting

| Symptom                            | Fix                                                   |
|------------------------------------|-------------------------------------------------------|
| `CUDA not available`               | Verify `torch.cuda.is_available()` — install cu128    |
| `ModuleNotFoundError: hypertensor` | Run from repo root; ensure `PYTHONPATH=.`             |
| OOM on GPU                         | Reduce `--n-bits` range or use `--device cpu`         |
| Hash mismatch on outputs           | Ensure commit `79910242`; FP non-determinism may cause minor JSON float diffs |

---

## Contact

Open an issue on the HyperTensor-VM repository for reproduction difficulties.
