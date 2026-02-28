# GPU Acceleration: CUDA/Triton kernels and dispatch

**Capability ID:** `gpu`

**Target namespace:** `qtenet.gpu`

## Summary

Provide a dispatch layer that selects CPU/GPU implementations deterministically.

## Notes

Compute kernels, autotuning, GPU point-eval and apply operations.

## Product invariants

- Graceful CPU fallback
- GPU non-determinism disclosed

## Observability (enterprise requirements)

- Record backend selection (cpu/cuda/triton)
- Record kernel versions and autotune params

## Canonical upstream sources (recommended top picks)

- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `ontic/genesis/core/rsvd.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/gpu_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/gpu_qtt_proper.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rkhs/kernels.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rkhs/mmd.py` (cat=genesis, lang=py, score=143)

## Candidate set (ranked, truncated)

- `ontic/core/gpu.py` (cat=core, score=150)
- `ontic/mpo/laplacian_cuda.py` (cat=core, score=150)
- `ontic/genesis/core/rsvd.py` (cat=genesis, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `ontic/genesis/demos/gpu_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/gpu_qtt_proper.py` (cat=genesis, score=143)
- `ontic/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/ridge.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/chebyshev.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/wavelets.py` (cat=genesis, score=143)
- `ontic/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci_gpu.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels_v2.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_ops.py` (cat=cfd, score=135)
- `ontic/cuda/qtt_eval_gpu.py` (cat=gpu, score=120)
- `ontic/cuda/qtt_native_ops.py` (cat=gpu, score=120)
- `ontic/cuda/qtt_ntt.py` (cat=gpu, score=120)
- `ontic/cuda/setup.py` (cat=gpu, score=120)
- `ontic/gpu/kernel_autotune_cache.py` (cat=gpu, score=120)
- `fluidelite/bench_step.py` (cat=fluidelite, score=115)
- `fluidelite/benchmarks/bench_elite_ops.py` (cat=fluidelite, score=115)
- `fluidelite/core/elite_ops.py` (cat=fluidelite, score=115)
- `fluidelite/core/triton_kernels.py` (cat=fluidelite, score=115)
- `fluidelite/kernels/qtt_argmax_kernel.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_direct_train.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_features.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_gpu_real.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_inference.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_native_inference.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_ns_hunt.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_physics.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_physics_100m.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_physics_first.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_rank_sweep.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_scale.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_tci.py` (cat=fluidelite, score=115)
- `fluidelite/train_hybrid_triton.py` (cat=fluidelite, score=115)
- `fluidelite/utils/health.py` (cat=fluidelite, score=115)
- `fluidelite/utils/repo_integration.py` (cat=fluidelite, score=115)
- `oracle/coinbase_oracle.py` (cat=oracle, score=65)
- `oracle/cuda_graph_slicer.py` (cat=oracle, score=65)
- `oracle/qtt_encoder_cuda.py` (cat=oracle, score=65)
- `oracle/triton_slicer.py` (cat=oracle, score=65)
- `ontic/sovereign/implicit_qtt_renderer.py` (cat=qtt-misc, score=55)
- *(+21 more candidates in inventory)*

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

