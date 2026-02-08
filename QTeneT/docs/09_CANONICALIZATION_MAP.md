# Canonicalization Map (Draft)

This document resolves **multiple implementations** of similarly-named QTT/TT/TCI primitives across the upstream monorepo into a **single preferred canonical location** for product packaging.

Rules are heuristic and *meant to be edited*. The goal is a clean, enterprise API boundary, not perfect semantic proof.

## Scoring heuristics (current)

Preference order (high → low): `tensornet/core` / `tensornet/genesis` / `tci` / `tensornet/cfd` / `fluidelite` / `sdk` / `compressor` (separate product) / demos & benchmarks / archived.

## Symbol-level canonical picks

Each entry lists: canonical path + all known alternates.

### `extract_features_kernel` (10 implementations)

**Canonical:** `fluidelite/bench_step.py` — `function` at L20 (score 125)

**Alternates:**

- `fluidelite/bench_step.py` — function L20 (cat=fluidelite)
- `fluidelite/qtt_direct_train.py` — function L34 (cat=fluidelite)
- `fluidelite/qtt_features.py` — function L33 (cat=fluidelite)
- `fluidelite/qtt_inference.py` — function L36 (cat=fluidelite)
- `fluidelite/qtt_native_inference.py` — function L32 (cat=fluidelite)
- `fluidelite/qtt_ns_hunt.py` — function L35 (cat=fluidelite)
- `fluidelite/qtt_physics.py` — function L39 (cat=fluidelite)
- `fluidelite/qtt_physics_100m.py` — function L43 (cat=fluidelite)
- `fluidelite/qtt_physics_first.py` — function L85 (cat=fluidelite)
- `fluidelite/qtt_tci.py` — function L29 (cat=fluidelite)


### `qtt_to_dense` (8 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L952 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L952 (cat=cfd)
- `tensornet/cfd/qtt_eval.py` — function L522 (cat=cfd)
- `fluidelite/qtt_accuracy_hunt.py` — function L45 (cat=fluidelite)
- `fluidelite/qtt_inference.py` — function L231 (cat=fluidelite)
- `fluidelite/qtt_native_inference.py` — function L73 (cat=fluidelite)
- `fluidelite/qtt_rank_sweep.py` — function L73 (cat=fluidelite)
- `fluidelite/qtt_scale_sweep.py` — function L51 (cat=fluidelite)
- `sdk/qtt-sdk/src/qtt_sdk/core.py` — function L230 (cat=sdk)


### `QTTMatrix` (8 implementations)

**Canonical:** `tensornet/genesis/ot/cost_matrices.py` — `class` at L45 (score 155)

**Alternates:**

- `tensornet/genesis/ot/cost_matrices.py` — class L45 (cat=genesis)
- `tensornet/genesis/topology/qtt_native.py` — class L154 (cat=genesis)
- `fluidelite/bench_step.py` — class L40 (cat=fluidelite)
- `fluidelite/qtt_direct_train.py` — class L80 (cat=fluidelite)
- `fluidelite/qtt_ns_hunt.py` — class L84 (cat=fluidelite)
- `fluidelite/qtt_physics.py` — class L229 (cat=fluidelite)
- `fluidelite/qtt_physics_100m.py` — class L267 (cat=fluidelite)
- `fluidelite/qtt_physics_first.py` — class L130 (cat=fluidelite)


### `tt_svd_gpu` (7 implementations)

**Canonical:** `tensornet/genesis/demos/gpu_qtt_compress.py` — `function` at L66 (score 153)

**Alternates:**

- `tensornet/genesis/demos/gpu_qtt_compress.py` — function L66 (cat=genesis)
- `tensornet/genesis/demos/gpu_qtt_proper.py` — function L85 (cat=genesis)
- `fluidelite/qtt_gpu_real.py` — function L75 (cat=fluidelite)
- `The_Compressor/qtt/spatial.py` — function L315 (cat=compressor)
- `qtt_50gb_cloud.py` — function L126 (cat=qtt-misc)
- `qtt_cloud_global.py` — function L111 (cat=qtt-misc)
- `qtt_global_gpu.py` — function L272 (cat=qtt-misc)


### `QTTState` (6 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `class` at L35 (score 147)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — class L35 (cat=cfd)
- `tensornet/cfd/qtt_tdvp.py` — class L99 (cat=cfd)
- `tensornet/cfd/qtt_triton_kernels_v2.py` — class L102 (cat=cfd)
- `tensornet/cfd/tci_flux.py` — class L588 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/core.py` — class L18 (cat=sdk)
- `archive/rns_qtt_ntt_butterfly_mpo_deprecated.py` — class L292 (cat=archive)


### `create_test_qtt` (5 implementations)

**Canonical:** `tensornet/cfd/qtt_eval.py` — `function` at L386 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_eval.py` — function L386 (cat=cfd)
- `tensornet/quantum/hybrid_qtt_renderer.py` — function L322 (cat=qtt-misc)
- `tensornet/quantum/qtt_glsl_bridge.py` — function L190 (cat=qtt-misc)
- `tensornet/visualization/tensor_slicer.py` — function L2061 (cat=other)
- `tests/benchmarks/qtt_render_benchmark.py` — function L42 (cat=qtt-misc)


### `MPO` (5 implementations)

**Canonical:** `tensornet/core/mpo.py` — `class` at L19 (score 162)

**Alternates:**

- `tensornet/core/mpo.py` — class L19 (cat=core)
- `tensornet/cfd/pure_qtt_ops.py` — class L64 (cat=cfd)
- `tensornet/cfd/qtt_triton_kernels_v2.py` — class L157 (cat=cfd)
- `fluidelite/core/mpo.py` — class L24 (cat=fluidelite)
- `sdk/qtt-sdk/src/qtt_sdk/core.py` — class L89 (cat=sdk)


### `QTT3DState` (5 implementations)

**Canonical:** `tensornet/cfd/fast_euler_3d.py` — `class` at L31 (score 147)

**Alternates:**

- `tensornet/cfd/fast_euler_3d.py` — class L31 (cat=cfd)
- `tensornet/sovereign/qtt_slice_extractor.py` — class L35 (cat=qtt-misc)
- `demos/black_swan_945_forensic.py` — class L36 (cat=demos)
- `demos/black_swan_reproduce.py` — class L34 (cat=demos)
- `demos/millennium_hunter.py` — class L46 (cat=demos)


### `QTTCore` (5 implementations)

**Canonical:** `tensornet/genesis/tropical/qtt_native.py` — `class` at L39 (score 155)

**Alternates:**

- `tensornet/genesis/tropical/qtt_native.py` — class L39 (cat=genesis)
- `tensornet/cfd/pure_qtt_ops.py` — class L28 (cat=cfd)
- `fluidelite/fe_tci/fluidelite_model.py` — class L18 (cat=fluidelite)
- `qtt_neural_connectome.py` — class L114 (cat=qtt-misc)
- `tig011a_multimechanism.py` — class L88 (cat=other)


### `generate_attestation` (4 implementations)

**Canonical:** `femto_fabricator_gauntlet.py` — `function` at L1120 (score 0)

**Alternates:**

- `femto_fabricator_gauntlet.py` — function L1120 (cat=other)
- `prometheus_gauntlet.py` — function L1611 (cat=other)
- `proteome_compiler_gauntlet.py` — function L1031 (cat=other)
- `tig011a_multimechanism.py` — function L1946 (cat=other)


### `qtt_forward` (4 implementations)

**Canonical:** `fluidelite/qtt_accuracy_hunt.py` — `function` at L56 (score 125)

**Alternates:**

- `fluidelite/qtt_accuracy_hunt.py` — function L56 (cat=fluidelite)
- `fluidelite/qtt_direct_train.py` — function L212 (cat=fluidelite)
- `fluidelite/qtt_ns_hunt.py` — function L118 (cat=fluidelite)
- `fluidelite/qtt_scale_sweep.py` — function L63 (cat=fluidelite)


### `tt_svd` (4 implementations)

**Canonical:** `tensornet/cfd/koopman_tt.py` — `function` at L141 (score 145)

**Alternates:**

- `tensornet/cfd/koopman_tt.py` — function L141 (cat=cfd)
- `tensornet/cfd/qtt.py` — function L88 (cat=cfd)
- `The_Compressor/qtt/spatial.py` — function L52 (cat=compressor)
- `demos/resolution_independence.py` — function L142 (cat=demos)


### `_evaluate_qtt_at_index` (3 implementations)

**Canonical:** `tensornet/genesis/ot/barycenters.py` — `function` at L594 (score 153)

**Alternates:**

- `tensornet/genesis/ot/barycenters.py` — function L594 (cat=genesis)
- `tensornet/genesis/ot/transport_plan.py` — function L932 (cat=genesis)
- `tensornet/genesis/ot/wasserstein.py` — function L277 (cat=genesis)


### `build_butterfly_mpo` (3 implementations)

**Canonical:** `tensornet/cuda/qtt_ntt.py` — `function` at L373 (score 130)

**Alternates:**

- `tensornet/cuda/qtt_ntt.py` — function L373 (cat=gpu)
- `tensornet_qtt_ntt.py` — function L139 (cat=qtt-misc)
- `archive/rns_qtt_ntt_butterfly_mpo_deprecated.py` — function L542 (cat=archive)


### `compress_to_qtt` (3 implementations)

**Canonical:** `demos/ingest_noaa_gfs.py` — `function` at L363 (score 30)

**Alternates:**

- `demos/ingest_noaa_gfs.py` — function L363 (cat=demos)
- `crypto_qtt_compress.py` — function L223 (cat=qtt-misc)
- `compress_crypto_data.py` — function L116 (cat=other)


### `EulerState3D` (3 implementations)

**Canonical:** `demos/black_swan_945_forensic.py` — `class` at L46 (score 32)

**Alternates:**

- `demos/black_swan_945_forensic.py` — class L46 (cat=demos)
- `demos/black_swan_reproduce.py` — class L45 (cat=demos)
- `demos/millennium_hunter.py` — class L61 (cat=demos)


### `identity_mpo` (3 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L153 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L153 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operators.py` — function L16 (cat=sdk)
- `yangmills/qtt.py` — function L634 (cat=qtt-misc)


### `MPS` (3 implementations)

**Canonical:** `tensornet/core/mps.py` — `class` at L22 (score 162)

**Alternates:**

- `tensornet/core/mps.py` — class L22 (cat=core)
- `fluidelite/core/mps.py` — class L81 (cat=fluidelite)
- `yangmills/tensor_network/mps.py` — class L24 (cat=other)


### `qtt_add` (3 implementations)

**Canonical:** `tensornet/genesis/ga/qtt_multivector.py` — `function` at L408 (score 153)

**Alternates:**

- `tensornet/genesis/ga/qtt_multivector.py` — function L408 (cat=genesis)
- `tensornet/cfd/pure_qtt_ops.py` — function L619 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operations.py` — function L16 (cat=sdk)


### `qtt_inner_product` (3 implementations)

**Canonical:** `tensornet/genesis/ga/qtt_multivector.py` — `function` at L734 (score 153)

**Alternates:**

- `tensornet/genesis/ga/qtt_multivector.py` — function L734 (cat=genesis)
- `tensornet/cfd/pure_qtt_ops.py` — function L819 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operations.py` — function L108 (cat=sdk)


### `qtt_scale` (3 implementations)

**Canonical:** `tensornet/genesis/ga/qtt_multivector.py` — `function` at L450 (score 153)

**Alternates:**

- `tensornet/genesis/ga/qtt_multivector.py` — function L450 (cat=genesis)
- `tensornet/cfd/pure_qtt_ops.py` — function L761 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operations.py` — function L84 (cat=sdk)


### `QTTCores` (3 implementations)

**Canonical:** `qtt_50gb_cloud.py` — `class` at L113 (score 12)

**Alternates:**

- `qtt_50gb_cloud.py` — class L113 (cat=qtt-misc)
- `qtt_cloud_global.py` — class L101 (cat=qtt-misc)
- `qtt_global_gpu.py` — class L197 (cat=qtt-misc)


### `QTTHeader` (3 implementations)

**Canonical:** `The_Compressor/qtt/container.py` — `class` at L92 (score 72)

**Alternates:**

- `The_Compressor/qtt/container.py` — class L92 (cat=compressor)
- `The_Compressor/qtt_container.py` — class L69 (cat=compressor)
- `The_Compressor/qtt_slicer.py` — class L61 (cat=compressor)


### `TTCore` (3 implementations)

**Canonical:** `tensornet/cfd/koopman_tt.py` — `class` at L57 (score 147)

**Alternates:**

- `tensornet/cfd/koopman_tt.py` — class L57 (cat=cfd)
- `fluidelite/qtt_gpu_real.py` — class L41 (cat=fluidelite)
- `tomahawk_cfd_gauntlet.py` — class L102 (cat=other)


### `_compute_qtt_cdf` (2 implementations)

**Canonical:** `tensornet/genesis/ot/barycenters.py` — `function` at L527 (score 153)

**Alternates:**

- `tensornet/genesis/ot/barycenters.py` — function L527 (cat=genesis)
- `tensornet/genesis/ot/transport_plan.py` — function L872 (cat=genesis)


### `_evaluate_qtt_at_indices` (2 implementations)

**Canonical:** `tensornet/cfd/tci_flux.py` — `function` at L752 (score 145)

**Alternates:**

- `tensornet/cfd/tci_flux.py` — function L752 (cat=cfd)
- `_archived_dense/tci_v2.py` — function L290 (cat=archive)


### `_qtt_quantile_search` (2 implementations)

**Canonical:** `tensornet/genesis/ot/barycenters.py` — `function` at L576 (score 153)

**Alternates:**

- `tensornet/genesis/ot/barycenters.py` — function L576 (cat=genesis)
- `tensornet/genesis/ot/wasserstein.py` — function L259 (cat=genesis)


### `_tt_add` (2 implementations)

**Canonical:** `tensornet/cfd/weno_tt.py` — `function` at L522 (score 145)

**Alternates:**

- `tensornet/cfd/weno_tt.py` — function L522 (cat=cfd)
- `tensornet/types/genesis_integration.py` — function L828 (cat=other)


### `_tt_svd` (2 implementations)

**Canonical:** `fluidelite/fe_tci/fluidelite_model.py` — `function` at L235 (score 125)

**Alternates:**

- `fluidelite/fe_tci/fluidelite_model.py` — function L235 (cat=fluidelite)
- `tensornet/types/genesis_integration.py` — function L783 (cat=other)


### `apply_mpo` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L471 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L471 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operators.py` — function L238 (cat=sdk)


### `apply_mpo_2d_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1702 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1702 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L1187 (cat=cfd)


### `apply_mpo_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1484 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1484 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L860 (cat=cfd)


### `benchmark_qtt_ntt` (2 implementations)

**Canonical:** `tensornet/cuda/qtt_ntt.py` — `function` at L1281 (score 130)

**Alternates:**

- `tensornet/cuda/qtt_ntt.py` — function L1281 (cat=gpu)
- `tensornet_qtt_ntt.py` — function L538 (cat=qtt-misc)


### `benchmark_tci_construction` (2 implementations)

**Canonical:** `tests/benchmarks/qtt_native_benchmark.py` — `function` at L74 (score 20)

**Alternates:**

- `tests/benchmarks/qtt_native_benchmark.py` — function L74 (cat=qtt-misc)
- `tests/benchmarks/optimized_pipeline_benchmark.py` — function L99 (cat=other)


### `check_imports` (2 implementations)

**Canonical:** `fluidelite/utils/health.py` — `function` at L142 (score 125)

**Alternates:**

- `fluidelite/utils/health.py` — function L142 (cat=fluidelite)
- `scripts/check_import_cycles.py` — function L21 (cat=other)


### `compress_frame_qtt` (2 implementations)

**Canonical:** `The_Compressor/compress_24h.py` — `function` at L98 (score 70)

**Alternates:**

- `The_Compressor/compress_24h.py` — function L98 (cat=compressor)
- `qtt_fullres_24h.py` — function L86 (cat=qtt-misc)


### `dense_to_qtt` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L895 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L895 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/core.py` — function L124 (cat=sdk)


### `dense_to_qtt_cores` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_eval.py` — `function` at L421 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_eval.py` — function L421 (cat=cfd)
- `tci_llm/qtt.py` — function L22 (cat=tci)


### `derivative_mpo` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L385 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L385 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operators.py` — function L107 (cat=sdk)


### `derivative_mpo_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1336 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1336 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L891 (cat=cfd)


### `extract_features_16k_kernel` (2 implementations)

**Canonical:** `fluidelite/qtt_rank_sweep.py` — `function` at L31 (score 125)

**Alternates:**

- `fluidelite/qtt_rank_sweep.py` — function L31 (cat=fluidelite)
- `fluidelite/qtt_scale.py` — function L39 (cat=fluidelite)


### `field_to_qtt` (2 implementations)

**Canonical:** `tensornet/cfd/qtt.py` — `function` at L217 (score 145)

**Alternates:**

- `tensornet/cfd/qtt.py` — function L217 (cat=cfd)
- `demos/resolution_independence.py` — function L163 (cat=demos)


### `fused_ultimate_kernel` (2 implementations)

**Canonical:** `The_Compressor/hybrid_triton_kernels.py` — `function` at L495 (score 70)

**Alternates:**

- `The_Compressor/hybrid_triton_kernels.py` — function L495 (cat=compressor)
- `The_Compressor/tci_triton.py` — function L347 (cat=compressor)


### `GlobalQTTCompressor` (2 implementations)

**Canonical:** `qtt_50gb_cloud.py` — `class` at L340 (score 12)

**Alternates:**

- `qtt_50gb_cloud.py` — class L340 (cat=qtt-misc)
- `qtt_global_gpu.py` — class L777 (cat=qtt-misc)


### `identity_mpo_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1257 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1257 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L804 (cat=cfd)


### `init_qtt_cores` (2 implementations)

**Canonical:** `fluidelite/qtt_accuracy_hunt.py` — `function` at L30 (score 125)

**Alternates:**

- `fluidelite/qtt_accuracy_hunt.py` — function L30 (cat=fluidelite)
- `fluidelite/qtt_scale_sweep.py` — function L36 (cat=fluidelite)


### `InvariantType` (2 implementations)

**Canonical:** `tensornet/exploit/invariant_hunter.py` — `class` at L32 (score 62)

**Alternates:**

- `tensornet/exploit/invariant_hunter.py` — class L32 (cat=other)
- `tensornet/exploit/invariants.py` — class L124 (cat=other)


### `KernelStats` (2 implementations)

**Canonical:** `tensornet/core/gpu.py` — `class` at L69 (score 162)

**Alternates:**

- `tensornet/core/gpu.py` — class L69 (cat=core)
- `tensornet/fieldos/kernel.py` — class L93 (cat=other)


### `laplacian_mpo` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L428 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L428 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operators.py` — function L180 (cat=sdk)


### `laplacian_mpo_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1377 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1377 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L925 (cat=cfd)


### `lattice_spacing` (2 implementations)

**Canonical:** `yangmills/transfer_matrix_rigorous.py` — `function` at L43 (score 10)

**Alternates:**

- `yangmills/transfer_matrix_rigorous.py` — function L43 (cat=other)
- `yangmills/weak_coupling_transmutation.py` — function L68 (cat=other)


### `mpo_sum` (2 implementations)

**Canonical:** `tensornet/core/mpo.py` — `function` at L187 (score 160)

**Alternates:**

- `tensornet/core/mpo.py` — function L187 (cat=core)
- `fluidelite/core/mpo.py` — function L193 (cat=fluidelite)


### `MPOCore` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `class` at L57 (score 147)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — class L57 (cat=cfd)
- `oracle/oracle_engine.py` — class L29 (cat=oracle)


### `QTT2DState` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_2d.py` — `class` at L103 (score 147)

**Alternates:**

- `tensornet/cfd/qtt_2d.py` — class L103 (cat=cfd)
- `tensornet/cfd/qtt_triton_kernels_v2.py` — class L129 (cat=cfd)


### `qtt_add_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1002 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1002 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L608 (cat=cfd)


### `qtt_eval_at_index` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_eval.py` — `function` at L126 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_eval.py` — function L126 (cat=cfd)
- `tci_llm/qtt.py` — function L127 (cat=tci)


### `qtt_eval_batch` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_eval.py` — `function` at L168 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_eval.py` — function L168 (cat=cfd)
- `tci_llm/qtt.py` — function L159 (cat=tci)


### `qtt_evaluate_batch_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1776 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1776 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L1203 (cat=cfd)


### `qtt_from_function_dense` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_tci.py` — `function` at L86 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_tci.py` — function L86 (cat=cfd)
- `tci_llm/qtt.py` — function L86 (cat=tci)


### `qtt_hadamard` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L768 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L768 (cat=cfd)
- `tensornet/cfd/qtt_hadamard.py` — function L48 (cat=cfd)


### `qtt_hadamard_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1173 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1173 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L730 (cat=cfd)


### `qtt_inner_product_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1223 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1223 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L766 (cat=cfd)


### `qtt_matvec_native` (2 implementations)

**Canonical:** `fluidelite/qtt_inference.py` — `function` at L77 (score 125)

**Alternates:**

- `fluidelite/qtt_inference.py` — function L77 (cat=fluidelite)
- `fluidelite/qtt_native_inference.py` — function L85 (cat=fluidelite)


### `qtt_norm` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L849 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L849 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operations.py` — function L152 (cat=sdk)


### `qtt_norm_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1247 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1247 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L795 (cat=cfd)


### `qtt_ntt_forward` (2 implementations)

**Canonical:** `tensornet/cuda/qtt_ntt.py` — `function` at L1226 (score 130)

**Alternates:**

- `tensornet/cuda/qtt_ntt.py` — function L1226 (cat=gpu)
- `tensornet_qtt_ntt.py` — function L483 (cat=qtt-misc)


### `qtt_ntt_inverse` (2 implementations)

**Canonical:** `tensornet/cuda/qtt_ntt.py` — `function` at L1242 (score 130)

**Alternates:**

- `tensornet/cuda/qtt_ntt.py` — function L1242 (cat=gpu)
- `tensornet_qtt_ntt.py` — function L499 (cat=qtt-misc)


### `qtt_poly_multiply` (2 implementations)

**Canonical:** `tensornet/cuda/qtt_ntt.py` — `function` at L1251 (score 130)

**Alternates:**

- `tensornet/cuda/qtt_ntt.py` — function L1251 (cat=gpu)
- `tensornet_qtt_ntt.py` — function L508 (cat=qtt-misc)


### `qtt_scale_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1161 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1161 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L723 (cat=cfd)


### `qtt_sum_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1061 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1061 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L658 (cat=cfd)


### `qtt_to_field` (2 implementations)

**Canonical:** `tensornet/cfd/qtt.py` — `function` at L287 (score 145)

**Alternates:**

- `tensornet/cfd/qtt.py` — function L287 (cat=cfd)
- `demos/resolution_independence.py` — function L344 (cat=demos)


### `QTTContainer` (2 implementations)

**Canonical:** `The_Compressor/qtt/container.py` — `class` at L233 (score 72)

**Alternates:**

- `The_Compressor/qtt/container.py` — class L233 (cat=compressor)
- `The_Compressor/qtt_container.py` — class L167 (cat=compressor)


### `QTTFooter` (2 implementations)

**Canonical:** `The_Compressor/qtt/container.py` — `class` at L169 (score 72)

**Alternates:**

- `The_Compressor/qtt/container.py` — class L169 (cat=compressor)
- `The_Compressor/qtt_container.py` — class L118 (cat=compressor)


### `QTTNTT` (2 implementations)

**Canonical:** `tensornet/cuda/qtt_ntt.py` — `class` at L531 (score 132)

**Alternates:**

- `tensornet/cuda/qtt_ntt.py` — class L531 (cat=gpu)
- `tensornet_qtt_ntt.py` — class L294 (cat=qtt-misc)


### `QTTResult` (2 implementations)

**Canonical:** `fluidelite/qtt_gpu_real.py` — `class` at L54 (score 127)

**Alternates:**

- `fluidelite/qtt_gpu_real.py` — class L54 (cat=fluidelite)
- `yang_mills_proof_pipeline.py` — class L41 (cat=other)


### `QTTSlicer` (2 implementations)

**Canonical:** `The_Compressor/qtt/slicer.py` — `class` at L78 (score 72)

**Alternates:**

- `The_Compressor/qtt/slicer.py` — class L78 (cat=compressor)
- `The_Compressor/qtt_slicer.py` — class L315 (cat=compressor)


### `QTTVector` (2 implementations)

**Canonical:** `tensornet/genesis/topology/qtt_native.py` — `class` at L41 (score 155)

**Alternates:**

- `tensornet/genesis/topology/qtt_native.py` — class L41 (cat=genesis)
- `tig011a_multimechanism.py` — class L117 (cat=other)


### `random_qtt` (2 implementations)

**Canonical:** `sdk/qtt-sdk/src/qtt_sdk/core.py` — `function` at L266 (score 106)

**Alternates:**

- `sdk/qtt-sdk/src/qtt_sdk/core.py` — function L266 (cat=sdk)
- `yangmills/qtt.py` — function L514 (cat=qtt-misc)


### `serialize_qtt` (2 implementations)

**Canonical:** `fluidelite/qtt_gpu_real.py` — `function` at L324 (score 125)

**Alternates:**

- `fluidelite/qtt_gpu_real.py` — function L324 (cat=fluidelite)
- `qtt_global_gpu.py` — function L551 (cat=qtt-misc)


### `shift_mpo` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L179 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L179 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operators.py` — function L38 (cat=sdk)


### `shift_mpo_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1272 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1272 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L817 (cat=cfd)


### `shift_mpo_x_2d_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1571 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1571 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L1095 (cat=cfd)


### `shift_mpo_y_2d_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L1634 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L1634 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L1141 (cat=cfd)


### `svd_truncated` (2 implementations)

**Canonical:** `tensornet/core/decompositions.py` — `function` at L28 (score 160)

**Alternates:**

- `tensornet/core/decompositions.py` — function L28 (cat=core)
- `fluidelite/core/decompositions.py` — function L131 (cat=fluidelite)


### `test_qtt_compression` (2 implementations)

**Canonical:** `The_Compressor/test_qtt_text_embeddings.py` — `function` at L125 (score 70)

**Alternates:**

- `The_Compressor/test_qtt_text_embeddings.py` — function L125 (cat=compressor)
- `tests/audit_layer_6.py` — function L55 (cat=other)


### `TestMPSOperations` (2 implementations)

**Canonical:** `fluidelite/tests/test_mps.py` — `class` at L130 (score 127)

**Alternates:**

- `fluidelite/tests/test_mps.py` — class L130 (cat=fluidelite)
- `Physics/tests/test_phase19.py` — class L541 (cat=other)


### `TestQTTCompression` (2 implementations)

**Canonical:** `tests/test_integration.py` — `class` at L533 (score 12)

**Alternates:**

- `tests/test_integration.py` — class L533 (cat=other)
- `tests/test_marrs_fusion.py` — class L607 (cat=other)


### `tropical_eigenvalue` (2 implementations)

**Canonical:** `tensornet/genesis/tropical/matrix.py` — `function` at L382 (score 153)

**Alternates:**

- `tensornet/genesis/tropical/matrix.py` — function L382 (cat=genesis)
- `tensornet/genesis/tropical/optimization.py` — function L47 (cat=genesis)


### `truncate_qtt` (2 implementations)

**Canonical:** `tensornet/cfd/pure_qtt_ops.py` — `function` at L539 (score 145)

**Alternates:**

- `tensornet/cfd/pure_qtt_ops.py` — function L539 (cat=cfd)
- `sdk/qtt-sdk/src/qtt_sdk/operations.py` — function L171 (cat=sdk)


### `truncate_qtt_gpu` (2 implementations)

**Canonical:** `tensornet/cfd/qtt_triton_kernels_v2.py` — `function` at L938 (score 145)

**Alternates:**

- `tensornet/cfd/qtt_triton_kernels_v2.py` — function L938 (cat=cfd)
- `tensornet/cfd/qtt_triton_ops.py` — function L551 (cat=cfd)


### `YangMillsMPO` (2 implementations)

**Canonical:** `yangmills/tensor_network/mpo.py` — `class` at L113 (score 12)

**Alternates:**

- `yangmills/tensor_network/mpo.py` — class L113 (cat=other)
- `elite_yang_mills_proof.py` — class L58 (cat=other)

