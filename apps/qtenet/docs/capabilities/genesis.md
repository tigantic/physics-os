# Genesis Primitives (OT/SGW/RMT/Tropical/RKHS/PH/GA)

**Capability ID:** `genesis`

**Target namespace:** `qtenet.genesis`

## Summary

Expose genesis modules with stable high-level APIs and explicit compute/memory envelopes.

## Notes

Seven meta-primitives as product modules.

## Product invariants

- No densification across pipeline stages
- Rank control explicitly applied between stages

## Observability (enterprise requirements)

- Per-stage compression ratio + ranks
- Emit pipeline manifest

## Canonical upstream sources (recommended top picks)

- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/genesis/__init__.py` (cat=genesis, lang=py, score=145)
- `ontic/genesis/benchmarks/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/logging.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/profiling.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/rsvd.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/validation.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/gpu_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/gpu_qtt_proper.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/hierarchical_qtt_compress.py` (cat=genesis, lang=py, score=143)

## Candidate set (ranked, truncated)

- `ontic/core/gpu.py` (cat=core, score=150)
- `ontic/genesis/__init__.py` (cat=genesis, score=145)
- `ontic/genesis/benchmarks/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, score=143)
- `ontic/genesis/core/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, score=143)
- `ontic/genesis/core/logging.py` (cat=genesis, score=143)
- `ontic/genesis/core/profiling.py` (cat=genesis, score=143)
- `ontic/genesis/core/rsvd.py` (cat=genesis, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `ontic/genesis/core/validation.py` (cat=genesis, score=143)
- `ontic/genesis/demos/gpu_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/gpu_qtt_proper.py` (cat=genesis, score=143)
- `ontic/genesis/demos/hierarchical_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/noaa_petabyte_compression.py` (cat=genesis, score=143)
- `ontic/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/stream_compress_1tb.py` (cat=genesis, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/fusion/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/fusion/genesis_fusion_demo.py` (cat=genesis, score=143)
- `ontic/genesis/fusion/geometric_types_pipeline.py` (cat=genesis, score=143)
- `ontic/genesis/ga/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/ga/cga.py` (cat=genesis, score=143)
- `ontic/genesis/ga/multivector.py` (cat=genesis, score=143)
- `ontic/genesis/ga/operations.py` (cat=genesis, score=143)
- `ontic/genesis/ga/products.py` (cat=genesis, score=143)
- `ontic/genesis/ga/qtt_ga_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/ga/qtt_multivector.py` (cat=genesis, score=143)
- `ontic/genesis/ga/rotors.py` (cat=genesis, score=143)
- `ontic/genesis/ot/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/ot/barycenters.py` (cat=genesis, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, score=143)
- `ontic/genesis/ot/qtt_ot_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, score=143)
- `ontic/genesis/ot/wasserstein.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/gp.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/qtt_rkhs_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/ridge.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/free_probability.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/resolvent.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- *(+63 more candidates in inventory)*

## Required tests (draft)

- Golden tests (known tensors/known ranks)
- Property tests (idempotence, tolerance bounds)
- No-dense guards where applicable

## Promotion checklist (experimental → stable)

- [ ] Stable API defined in `qtenet.sdk`
- [ ] Doc page exists
- [ ] Determinism documented
- [ ] Tests exist
- [ ] Benchmark envelope documented

