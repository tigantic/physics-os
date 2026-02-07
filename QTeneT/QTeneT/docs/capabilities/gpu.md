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

- `tensornet/core/gpu.py` (cat=core, lang=py, score=150)
- `tensornet/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `tensornet/genesis/core/rsvd.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/demos/gpu_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/demos/gpu_qtt_proper.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/demos/triton_qtt.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/ot/cost_matrices.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/rkhs/kernel_matrix.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/rkhs/kernels.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/rkhs/mmd.py` (cat=genesis, lang=py, score=143)

## Candidate set (ranked, truncated)

- `tensornet/core/gpu.py` (cat=core, score=150)
- `tensornet/mpo/laplacian_cuda.py` (cat=core, score=150)
- `tensornet/genesis/core/rsvd.py` (cat=genesis, score=143)
- `tensornet/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `tensornet/genesis/demos/gpu_qtt_compress.py` (cat=genesis, score=143)
- `tensornet/genesis/demos/gpu_qtt_proper.py` (cat=genesis, score=143)
- `tensornet/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, score=143)
- `tensornet/genesis/demos/triton_qtt.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/ridge.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/chebyshev.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/wavelets.py` (cat=genesis, score=143)
- `tensornet/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `tensornet/cfd/qtt_tci_gpu.py` (cat=cfd, score=135)
- `tensornet/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- `tensornet/cfd/qtt_triton_kernels_v2.py` (cat=cfd, score=135)
- `tensornet/cfd/qtt_triton_ops.py` (cat=cfd, score=135)
- `tensornet/cuda/qtt_eval_gpu.py` (cat=gpu, score=120)
- `tensornet/cuda/qtt_native_ops.py` (cat=gpu, score=120)
- `tensornet/cuda/qtt_ntt.py` (cat=gpu, score=120)
- `tensornet/cuda/setup.py` (cat=gpu, score=120)
- `tensornet/gpu/kernel_autotune_cache.py` (cat=gpu, score=120)
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
- `tensornet/sovereign/implicit_qtt_renderer.py` (cat=qtt-misc, score=55)
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

