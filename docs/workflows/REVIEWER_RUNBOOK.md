# HyperTensor Reviewer Runbook

**One command proves the thesis.**

---

## Quick Start (< 5 minutes)

```bash
# 1. Install
pip install -e .

# 2. Run flagship pipeline
python demos/flagship_pipeline.py

# 3. Verify evidence
python artifacts/evidence/flagship_pack/verify.py
```

**Expected output: `PASS`**

---

## What the Flagship Pipeline Does

The pipeline executes the complete HyperTensor thesis demonstration:

| Step | What Happens | Validates |
|------|-------------|-----------|
| 1 | Initialize Sod shock tube | Canonical CFD test case |
| 2 | Convert to Tensor-Train format | TT compression works |
| 3 | Apply WENO-TT reconstruction | Shock-capturing in TT format |
| 4 | Evolve with TDVP-CFD | Time evolution preserves structure |
| 5 | Derive plasma quantities | Ionization/blackout physics |
| 6 | Validate conservation | Mass/energy preserved |
| 7 | Emit evidence pack | Cryptographic verification |

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | Any x86_64 | 4+ cores |
| RAM | 4 GB | 8 GB |
| GPU | None required | CUDA 11.8+ for acceleration |
| Disk | 100 MB | 500 MB |

**Runtime:** ~30 seconds on typical hardware

---

## Dependencies

```bash
# Core (required)
torch>=2.0.0
numpy>=1.21.0

# Full installation
pip install -e ".[dev]"
```

---

## Evidence Pack Contents

After running `demos/flagship_pipeline.py`:

```
artifacts/evidence/flagship_pack/
├── manifest.json      # Results + SHA256 hashes + HMAC signature
├── verify.py          # Self-contained verification script
└── data/
    ├── x_grid.npy           # Spatial grid
    ├── rho_final.npy        # Final density field
    ├── u_final.npy          # Final velocity field
    ├── p_final.npy          # Final pressure field
    └── electron_density.npy # Plasma electron density
```

---

## Validation Checks

The pipeline validates:

1. **Conservation Laws**
   - Mass error < 1e-6
   - Energy error < 1e-6

2. **Shock Structure**
   - Density ratio across shock ≈ 4 (Rankine-Hugoniot)
   - Shock location at x ≈ 0.7-0.8

3. **WENO-TT Accuracy**
   - L2 error vs dense WENO < 5%

4. **Positivity**
   - ρ > 0, p > 0 everywhere

---

## Phase-Deferred Features

The following features are intentionally deferred (not bugs):

| Phase | Feature | Status |
|-------|---------|--------|
| 24 | Adjoint gradient computation | Deferred |
| 24 | Full MPO-to-gate extraction | Deferred |
| 25 | Real-time execution guarantees | Deferred |
| 25 | Jetson hardware validation | Deferred |
| 25 | ROM encode/decode | Deferred |

These raise `PhaseDeferredError` with explicit dependencies.

---

## Troubleshooting

### Import Error
```bash
pip install -e .
```

### CUDA Out of Memory
```python
# Pipeline uses CPU by default
# For GPU, set smaller grid: nx=100
```

### verify.py fails
```bash
# Re-run pipeline to regenerate evidence
python demos/flagship_pipeline.py
```

---

## Full Test Suite

```bash
# All tests
pytest tests/ -v

# Just flagship
pytest tests/integration/test_flagship_pipeline.py -v

# With coverage
pytest tests/ --cov=tensornet --cov-report=html
```

---

## Reproducibility Guarantee

The pipeline uses fixed seeds:
- `MASTER_SEED = 42`
- `torch.backends.cudnn.deterministic = True`

Same hardware + same seed = identical results.

---

## Contact

For questions about this evaluation:
- Open an issue on GitHub
- Reference: HyperTensor Phase 21-24 Implementation

---

*Last updated: 2024-12-22*
