# Decomposition & Rounding: TT-SVD / truncation / rank control

**Capability ID:** `decomposition-rounding`

**Target namespace:** `qtenet.core.decomposition`

## Summary

Provides deterministic decomposition and rounding APIs used by everything else.

## Notes

Choose one canonical decomposition/rounding stack, then wrap everything else behind it.

## Product invariants

- Deterministic where possible (or disclosed non-determinism)
- Idempotence under repeated rounding (within tolerance)
- No silent densification

## Observability (enterprise requirements)

- Log SVD mode (full/randomized)
- Record truncation error + ranks

## Canonical upstream sources (recommended top picks)

- `ontic/core/__init__.py` (cat=core, lang=py, score=150)
- `ontic/core/decompositions.py` (cat=core, lang=py, score=150)
- `ontic/core/dense_guard.py` (cat=core, lang=py, score=150)
- `ontic/core/determinism.py` (cat=core, lang=py, score=150)
- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/core/mpo.py` (cat=core, lang=py, score=150)
- `ontic/core/mps.py` (cat=core, lang=py, score=150)
- `ontic/core/phase_deferred.py` (cat=core, lang=py, score=150)

## Candidate set (ranked, truncated)

- `ontic/core/__init__.py` (cat=core, score=150)
- `ontic/core/decompositions.py` (cat=core, score=150)
- `ontic/core/dense_guard.py` (cat=core, score=150)
- `ontic/core/determinism.py` (cat=core, score=150)
- `ontic/core/gpu.py` (cat=core, score=150)
- `ontic/core/mpo.py` (cat=core, score=150)
- `ontic/core/mps.py` (cat=core, score=150)
- `ontic/core/phase_deferred.py` (cat=core, score=150)
- `ontic/core/profiling.py` (cat=core, score=150)
- `ontic/core/states.py` (cat=core, score=150)
- `ontic/genesis/core/rsvd.py` (cat=genesis, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `ontic/genesis/demos/gpu_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/gpu_qtt_proper.py` (cat=genesis, score=143)
- `ontic/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/stream_compress_1tb.py` (cat=genesis, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/ga/qtt_multivector.py` (cat=genesis, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/graph_signals.py` (cat=genesis, score=143)
- `ontic/genesis/topology/boundary.py` (cat=genesis, score=143)
- `ontic/genesis/topology/qtt_native.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/qtt_native.py` (cat=genesis, score=143)
- `ontic/cfd/koopman_tt.py` (cat=cfd, score=135)
- `ontic/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `ontic/cfd/pure_qtt_ops.py` (cat=cfd, score=135)
- `ontic/cfd/qtt.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift_native.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_eval.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_streaming.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels_v2.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_ops.py` (cat=cfd, score=135)
- `ontic/cfd/tci_flux.py` (cat=cfd, score=135)
- `ontic/cfd/tt_cfd.py` (cat=cfd, score=135)
- `ontic/cfd/tt_poisson.py` (cat=cfd, score=135)
- `ontic/cfd/weno_native_tt.py` (cat=cfd, score=135)
- `ontic/cfd/weno_tt.py` (cat=cfd, score=135)
- `ontic/cuda/qtt_ntt.py` (cat=gpu, score=120)
- `fluidelite/core/decompositions.py` (cat=fluidelite, score=115)
- `fluidelite/core/elite_ops.py` (cat=fluidelite, score=115)
- `fluidelite/core/mps.py` (cat=fluidelite, score=115)
- `fluidelite/fe_tci/fluidelite_model.py` (cat=fluidelite, score=115)
- `fluidelite/noaa_slicer_real.py` (cat=fluidelite, score=115)
- *(+64 more candidates in inventory)*

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

