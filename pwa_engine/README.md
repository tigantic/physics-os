# PWA Engine — Standalone Distribution

Partial Wave Analysis Compute Engine V3.0.0 extracted from
[HyperTensor-VM](https://github.com/tigantic/HyperTensor-VM).

Implements **Eq. 5.48** from Badui (2020) with Gram-matrix-accelerated
extended likelihood evaluation for meson photoproduction amplitude analysis.

## Install (in-repo)

```bash
cd pwa_engine/
pip install -e .
```

## Install (standalone)

Copy `experiments/pwa_engine/core.py` into `pwa_engine/core.py`, then:

```bash
pip install -e .
```

## Usage

```bash
# Convention reduction test (fast, ~0.5s)
pwa-engine --convention-only

# Full 10-experiment suite (requires full repo)
pwa-engine

# From Python
from pwa_engine import build_wave_set, convention_reduction_test
result = convention_reduction_test()
assert result["all_pass"]
```

## Reference

Badui, Bannon, et al. (2020), PhD Dissertation, Indiana University.
Adams (2026), HyperTensor-VM Platform V3.0.0.
