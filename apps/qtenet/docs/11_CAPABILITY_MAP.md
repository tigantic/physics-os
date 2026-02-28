# Capability Map (Draft)

This document is the **enterprise packaging map** for QTeneT.

It groups upstream implementations into **capabilities** and proposes:

- canonical module targets inside `qtenet.*`

- recommended upstream sources to wire first


It is intentionally opinionated, and should be edited as you unify implementations.

## Capability index

- [Core Types: QTTTensor / TT cores / MPS / MPO](#core-types) — 587 candidate artifacts
- [Decomposition & Rounding: TT-SVD / truncation / rank control](#decomposition-rounding) — 131 candidate artifacts
- [Construction: TCI / TT-Cross](#tci-ttcross) — 87 candidate artifacts
- [Operators: shift/roll, Hadamard/Walsh-Hadamard, spectral](#operators-shift-spectral) — 76 candidate artifacts
- [Operators: gradient/divergence/Laplacian/advection/diffusion](#operators-derivative-laplacian) — 71 candidate artifacts
- [Solvers: Euler/NS/PDE pipelines (IMEX/TDVP)](#solvers-pde-cfd) — 68 candidate artifacts
- [Genesis: QTT Optimal Transport (OT)](#genesis-ot) — 14 candidate artifacts
- [Genesis: QTT Spectral Graph Wavelets (SGW)](#genesis-sgw) — 26 candidate artifacts
- [Genesis: QTT Random Matrix Theory (RMT)](#genesis-rmt) — 17 candidate artifacts
- [Genesis: QTT Tropical Geometry](#genesis-tropical) — 14 candidate artifacts
- [Genesis: QTT RKHS / Kernel Methods](#genesis-rkhs) — 48 candidate artifacts
- [Genesis: QTT Persistent Homology](#genesis-ph) — 10 candidate artifacts
- [Genesis: QTT Geometric Algebra (GA)](#genesis-ga) — 9 candidate artifacts
- [GPU Acceleration: CUDA/Triton kernels and dispatch](#gpu-accel) — 71 candidate artifacts
- [Enterprise SDK: stable facade (what users import)](#sdk-facade) — 18 candidate artifacts
- [Oracle utilities: QTT slicing / encoding](#oracle-slicing-encoding) — 22 candidate artifacts
- [Provenance & Attestation artifacts](#provenance-attestation) — 47 candidate artifacts

---


## Core Types: QTTTensor / TT cores / MPS / MPO

<a id="core-types"></a>

**Target namespace:** `qtenet.core`

**Notes:** Canonical tensor network types and containers. This is the foundation layer.

### Recommended canonical upstream sources (top picks)

- `ontic/core/__init__.py` (cat=core, lang=py, score=150)
- `ontic/core/decompositions.py` (cat=core, lang=py, score=150)
- `ontic/core/dense_guard.py` (cat=core, lang=py, score=150)
- `ontic/core/determinism.py` (cat=core, lang=py, score=150)
- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/core/mpo.py` (cat=core, lang=py, score=150)
- `ontic/core/mps.py` (cat=core, lang=py, score=150)
- `ontic/core/phase_deferred.py` (cat=core, lang=py, score=150)

### Candidate set (ranked)

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
- `ontic/mpo/__init__.py` (cat=core, score=150)
- `ontic/mpo/atmospheric_solver.py` (cat=core, score=150)
- `ontic/mpo/laplacian_cuda.py` (cat=core, score=150)
- `ontic/mpo/operators.py` (cat=core, score=150)
- `ontic/mps/__init__.py` (cat=core, score=150)
- `ontic/mps/hamiltonians.py` (cat=core, score=150)
- `ontic/genesis/__init__.py` (cat=genesis, score=145)
- `ontic/genesis/core/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, score=143)
- `ontic/genesis/core/logging.py` (cat=genesis, score=143)
- `ontic/genesis/core/rsvd.py` (cat=genesis, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `ontic/genesis/core/validation.py` (cat=genesis, score=143)
- `ontic/genesis/ga/__init__.py` (cat=genesis, score=143)
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
- *(+547 more candidates)*

---


## Decomposition & Rounding: TT-SVD / truncation / rank control

<a id="decomposition-rounding"></a>

**Target namespace:** `qtenet.core.decomposition`

**Notes:** Choose one canonical decomposition/rounding stack, then wrap everything else behind it.

### Recommended canonical upstream sources (top picks)

- `ontic/core/__init__.py` (cat=core, lang=py, score=150)
- `ontic/core/decompositions.py` (cat=core, lang=py, score=150)
- `ontic/core/dense_guard.py` (cat=core, lang=py, score=150)
- `ontic/core/determinism.py` (cat=core, lang=py, score=150)
- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/core/mpo.py` (cat=core, lang=py, score=150)
- `ontic/core/mps.py` (cat=core, lang=py, score=150)
- `ontic/core/phase_deferred.py` (cat=core, lang=py, score=150)

### Candidate set (ranked)

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
- *(+91 more candidates)*

---


## Construction: TCI / TT-Cross

<a id="tci-ttcross"></a>

**Target namespace:** `qtenet.tci`

**Notes:** Black-box function → QTT construction. Includes sampling policies and maxvol/skeleton selection.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/wasserstein.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/qtt_native.py` (cat=genesis, lang=py, score=143)
- `ontic/cfd/euler2d_native.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, lang=py, score=135)

### Candidate set (ranked)

- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, score=143)
- `ontic/genesis/ot/wasserstein.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/qtt_native.py` (cat=genesis, score=143)
- `ontic/cfd/euler2d_native.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, score=135)
- `ontic/cfd/euler_nd_native.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_2d.py` (cat=cfd, score=135)
- `ontic/cfd/flux_2d_tci.py` (cat=cfd, score=135)
- `ontic/cfd/flux_batch.py` (cat=cfd, score=135)
- `ontic/cfd/kelvin_helmholtz.py` (cat=cfd, score=135)
- `ontic/cfd/local_flux_native.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_eval.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_multiscale.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_streaming.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci_gpu.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tdvp.py` (cat=cfd, score=135)
- `ontic/cfd/tci_benchmark_suite.py` (cat=cfd, score=135)
- `ontic/cfd/tci_flux.py` (cat=cfd, score=135)
- `ontic/cfd/tci_true.py` (cat=cfd, score=135)
- `ontic/cfd/weno_native_tt.py` (cat=cfd, score=135)
- `ontic/cuda/qtt_ntt.py` (cat=gpu, score=120)
- `fluidelite/benchmarks/wikitext.py` (cat=fluidelite, score=115)
- `fluidelite/core/cross.py` (cat=fluidelite, score=115)
- `fluidelite/fe_tci/__init__.py` (cat=fluidelite, score=115)
- `fluidelite/fe_tci/context_encoder.py` (cat=fluidelite, score=115)
- `fluidelite/fe_tci/fluidelite_model.py` (cat=fluidelite, score=115)
- `fluidelite/kernels/qtt_argmax_kernel.py` (cat=fluidelite, score=115)
- `fluidelite/llm/fluid_elite.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_rank_sweep.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_scale_sweep.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_tci.py` (cat=fluidelite, score=115)
- `tci_llm/__init__.py` (cat=tci, score=100)
- `tci_llm/demo.py` (cat=tci, score=100)
- `tci_llm/examples/demo.py` (cat=tci, score=100)
- `tci_llm/generalized_llm.py` (cat=tci, score=100)
- *(+47 more candidates)*

---


## Operators: shift/roll, Hadamard/Walsh-Hadamard, spectral

<a id="operators-shift-spectral"></a>

**Target namespace:** `qtenet.operators`

**Notes:** Pure operator library: return MPOs (or operator-like objects) + metadata (scheme/version).

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/matrix.py` (cat=genesis, lang=py, score=143)
- `ontic/cfd/adaptive_tt.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/comfort_metrics.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_native.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, lang=py, score=135)

### Candidate set (ranked)

- `ontic/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/matrix.py` (cat=genesis, score=143)
- `ontic/cfd/adaptive_tt.py` (cat=cfd, score=135)
- `ontic/cfd/comfort_metrics.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_native.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, score=135)
- `ontic/cfd/euler_1d.py` (cat=cfd, score=135)
- `ontic/cfd/euler_nd_native.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_2d.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_3d.py` (cat=cfd, score=135)
- `ontic/cfd/fast_vlasov_5d.py` (cat=cfd, score=135)
- `ontic/cfd/flux_2d_tci.py` (cat=cfd, score=135)
- `ontic/cfd/flux_batch.py` (cat=cfd, score=135)
- `ontic/cfd/kelvin_helmholtz.py` (cat=cfd, score=135)
- `ontic/cfd/koopman_tt.py` (cat=cfd, score=135)
- `ontic/cfd/local_flux_native.py` (cat=cfd, score=135)
- `ontic/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `ontic/cfd/ns2d_qtt_native.py` (cat=cfd, score=135)
- `ontic/cfd/pure_qtt_ops.py` (cat=cfd, score=135)
- `ontic/cfd/qtt.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift_native.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_cfd.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_checkpoint_stream.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_eval.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_hadamard.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_imex.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_multiscale.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_reciprocal.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_regularity.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_shift_stable.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_spectral.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_streaming.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci_gpu.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tdvp.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- *(+36 more candidates)*

---


## Operators: gradient/divergence/Laplacian/advection/diffusion

<a id="operators-derivative-laplacian"></a>

**Target namespace:** `qtenet.operators.pde`

**Notes:** MPO builders + application kernels used by PDE solvers.

### Recommended canonical upstream sources (top picks)

- `ontic/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `ontic/mpo/operators.py` (cat=core, lang=py, score=150)
- `ontic/genesis/sgw/filters.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/sgw/laplacian.py` (cat=genesis, lang=py, score=143)
- `ontic/cfd/adaptive_tt.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/comfort_metrics.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_native.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, lang=py, score=135)

### Candidate set (ranked)

- `ontic/mpo/laplacian_cuda.py` (cat=core, score=150)
- `ontic/mpo/operators.py` (cat=core, score=150)
- `ontic/genesis/sgw/filters.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/laplacian.py` (cat=genesis, score=143)
- `ontic/cfd/adaptive_tt.py` (cat=cfd, score=135)
- `ontic/cfd/comfort_metrics.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_native.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, score=135)
- `ontic/cfd/euler_1d.py` (cat=cfd, score=135)
- `ontic/cfd/euler_nd_native.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_2d.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_3d.py` (cat=cfd, score=135)
- `ontic/cfd/fast_vlasov_5d.py` (cat=cfd, score=135)
- `ontic/cfd/flux_2d_tci.py` (cat=cfd, score=135)
- `ontic/cfd/flux_batch.py` (cat=cfd, score=135)
- `ontic/cfd/kelvin_helmholtz.py` (cat=cfd, score=135)
- `ontic/cfd/koopman_tt.py` (cat=cfd, score=135)
- `ontic/cfd/local_flux_native.py` (cat=cfd, score=135)
- `ontic/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `ontic/cfd/ns2d_qtt_native.py` (cat=cfd, score=135)
- `ontic/cfd/pure_qtt_ops.py` (cat=cfd, score=135)
- `ontic/cfd/qtt.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift_native.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_cfd.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_checkpoint_stream.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_eval.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_hadamard.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_imex.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_multiscale.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_reciprocal.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_regularity.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_shift_stable.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_spectral.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_streaming.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci_gpu.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tdvp.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- *(+31 more candidates)*

---


## Solvers: Euler/NS/PDE pipelines (IMEX/TDVP)

<a id="solvers-pde-cfd"></a>

**Target namespace:** `qtenet.solvers`

**Notes:** End-to-end solver layer. The enterprise contract here is reproducibility + rank control + diagnostics.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/ga/rotors.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/persistence.py` (cat=genesis, lang=py, score=143)
- `ontic/cfd/adaptive_tt.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/comfort_metrics.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_native.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler_1d.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler_nd_native.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/fast_euler_2d.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/fast_euler_3d.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/fast_vlasov_5d.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/flux_2d_tci.py` (cat=cfd, lang=py, score=135)

### Candidate set (ranked)

- `ontic/genesis/ga/rotors.py` (cat=genesis, score=143)
- `ontic/genesis/topology/persistence.py` (cat=genesis, score=143)
- `ontic/cfd/adaptive_tt.py` (cat=cfd, score=135)
- `ontic/cfd/comfort_metrics.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_native.py` (cat=cfd, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, score=135)
- `ontic/cfd/euler_1d.py` (cat=cfd, score=135)
- `ontic/cfd/euler_nd_native.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_2d.py` (cat=cfd, score=135)
- `ontic/cfd/fast_euler_3d.py` (cat=cfd, score=135)
- `ontic/cfd/fast_vlasov_5d.py` (cat=cfd, score=135)
- `ontic/cfd/flux_2d_tci.py` (cat=cfd, score=135)
- `ontic/cfd/flux_batch.py` (cat=cfd, score=135)
- `ontic/cfd/kelvin_helmholtz.py` (cat=cfd, score=135)
- `ontic/cfd/koopman_tt.py` (cat=cfd, score=135)
- `ontic/cfd/local_flux_native.py` (cat=cfd, score=135)
- `ontic/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `ontic/cfd/ns2d_qtt_native.py` (cat=cfd, score=135)
- `ontic/cfd/pure_qtt_ops.py` (cat=cfd, score=135)
- `ontic/cfd/qtt.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_2d_shift_native.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_cfd.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_checkpoint_stream.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_eval.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_hadamard.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_imex.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_multiscale.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_reciprocal.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_regularity.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_shift_stable.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_spectral.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_streaming.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tci_gpu.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_tdvp.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels_v2.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_ops.py` (cat=cfd, score=135)
- *(+28 more candidates)*

---


## Genesis: QTT Optimal Transport (OT)

<a id="genesis-ot"></a>

**Target namespace:** `qtenet.genesis.ot`

**Notes:** OT primitives: distributions, cost matrices, Sinkhorn, Wasserstein distance.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/barycenters.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/qtt_ot_gauntlet.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, lang=py, score=143)

### Candidate set (ranked)

- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, score=143)
- `ontic/genesis/ot/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/ot/barycenters.py` (cat=genesis, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, score=143)
- `ontic/genesis/ot/qtt_ot_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, score=143)
- `ontic/genesis/ot/wasserstein.py` (cat=genesis, score=143)
- `ontic/genesis/topology/distances.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/convexity.py` (cat=genesis, score=143)
- `ontic/genesis/ot/README.md` (cat=genesis, score=133)
- `ontic/discovery/primitives/optimal_transport.py` (cat=other, score=48)
- `oracle_node/server.py` (cat=other, score=10)

---


## Genesis: QTT Spectral Graph Wavelets (SGW)

<a id="genesis-sgw"></a>

**Target namespace:** `qtenet.genesis.sgw`

**Notes:** Graph Laplacian, wavelet kernels, Chebyshev approximations.

### Recommended canonical upstream sources (top picks)

- `ontic/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `ontic/mpo/operators.py` (cat=core, lang=py, score=150)
- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/sgw/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/sgw/chebyshev.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/sgw/filters.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/sgw/graph_signals.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/sgw/laplacian.py` (cat=genesis, lang=py, score=143)

### Candidate set (ranked)

- `ontic/mpo/laplacian_cuda.py` (cat=core, score=150)
- `ontic/mpo/operators.py` (cat=core, score=150)
- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/chebyshev.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/filters.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/graph_signals.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/laplacian.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/qtt_sgw_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/wavelets.py` (cat=genesis, score=143)
- `ontic/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `ontic/cfd/pure_qtt_ops.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels_v2.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_ops.py` (cat=cfd, score=135)
- `ontic/cfd/tt_poisson.py` (cat=cfd, score=135)
- `ontic/genesis/sgw/README.md` (cat=genesis, score=133)
- `fluidelite/benchmarks/bench_elite_ops.py` (cat=fluidelite, score=115)
- `apps/sdk_legacy/qtt-sdk/src/qtt_sdk/operators.py` (cat=sdk, score=101)
- `ontic/benchmarks/profile_bottlenecks.py` (cat=other, score=50)
- `ontic/fieldops/operators.py` (cat=other, score=50)
- `ontic/discovery/primitives/spectral_wavelets.py` (cat=other, score=48)
- `oracle_node/server.py` (cat=other, score=10)
- `proofs/proof_phase_2.py` (cat=other, score=10)
- `tests/audit_layer_5.py` (cat=other, score=10)
- `tests/test_fieldops.py` (cat=other, score=10)
- `tests/test_mpo_solver.py` (cat=other, score=10)

---


## Genesis: QTT Random Matrix Theory (RMT)

<a id="genesis-rmt"></a>

**Target namespace:** `qtenet.genesis.rmt`

**Notes:** Ensembles, resolvents, spectral density estimation.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/rmt/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/ensembles.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/free_probability.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/resolvent.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rmt/universality.py` (cat=genesis, lang=py, score=143)
- `ontic/cfd/qtt_shift_stable.py` (cat=cfd, lang=py, score=135)

### Candidate set (ranked)

- `ontic/genesis/rmt/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/free_probability.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/resolvent.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/universality.py` (cat=genesis, score=143)
- `ontic/cfd/qtt_shift_stable.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_spectral.py` (cat=cfd, score=135)
- `ontic/cfd/stabilized_refine.py` (cat=cfd, score=135)
- `ontic/cfd/tt_poisson.py` (cat=cfd, score=135)
- `ontic/genesis/rmt/README.md` (cat=genesis, score=133)
- `ontic/discovery/primitives/spectral_wavelets.py` (cat=other, score=48)
- `hermes_gauntlet.py` (cat=other, score=10)
- `navier_stokes_millennium_pipeline.py` (cat=other, score=10)
- `oracle_node/server.py` (cat=other, score=10)
- `yangmills/transfer_matrix_rigorous.py` (cat=other, score=10)

---


## Genesis: QTT Tropical Geometry

<a id="genesis-tropical"></a>

**Target namespace:** `qtenet.genesis.tropical`

**Notes:** Tropical semiring, shortest paths, tropical eigen computations.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/fusion/genesis_fusion_demo.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/convexity.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/matrix.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/optimization.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/qtt_native.py` (cat=genesis, lang=py, score=143)

### Candidate set (ranked)

- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, score=143)
- `ontic/genesis/fusion/genesis_fusion_demo.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/convexity.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/matrix.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/optimization.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/qtt_native.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/qtt_tropical_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/semiring.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/shortest_path.py` (cat=genesis, score=143)
- `ontic/genesis/tropical/README.md` (cat=genesis, score=133)
- `qtt_native_gauntlet.py` (cat=qtt-misc, score=15)
- `genesis_benchmark_suite.py` (cat=other, score=10)

---


## Genesis: QTT RKHS / Kernel Methods

<a id="genesis-rkhs"></a>

**Target namespace:** `qtenet.genesis.rkhs`

**Notes:** Kernel ridge, MMD, GP regression primitives.

### Recommended canonical upstream sources (top picks)

- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/fusion/genesis_fusion_demo.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/fusion/geometric_types_pipeline.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/rkhs/__init__.py` (cat=genesis, lang=py, score=143)

### Candidate set (ranked)

- `ontic/core/gpu.py` (cat=core, score=150)
- `ontic/genesis/benchmarks/massacre.py` (cat=genesis, score=143)
- `ontic/genesis/core/exceptions.py` (cat=genesis, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/fusion/genesis_fusion_demo.py` (cat=genesis, score=143)
- `ontic/genesis/fusion/geometric_types_pipeline.py` (cat=genesis, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/gp.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/qtt_rkhs_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/ridge.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/chebyshev.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/wavelets.py` (cat=genesis, score=143)
- `ontic/cfd/qtt_triton_kernels.py` (cat=cfd, score=135)
- `ontic/cfd/qtt_triton_kernels_v2.py` (cat=cfd, score=135)
- `ontic/genesis/rkhs/README.md` (cat=genesis, score=133)
- `ontic/gpu/kernel_autotune_cache.py` (cat=gpu, score=120)
- `fluidelite/bench_step.py` (cat=fluidelite, score=115)
- `fluidelite/core/triton_kernels.py` (cat=fluidelite, score=115)
- `fluidelite/kernels/qtt_argmax_kernel.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_direct_train.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_features.py` (cat=fluidelite, score=115)
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
- `oracle/coinbase_oracle.py` (cat=oracle, score=65)
- `oracle/triton_slicer.py` (cat=oracle, score=65)
- `ontic/sovereign/implicit_qtt_renderer.py` (cat=qtt-misc, score=55)
- `ontic/fieldos/kernel.py` (cat=other, score=50)
- `ontic/substrate/stats.py` (cat=other, score=50)
- *(+8 more candidates)*

---


## Genesis: QTT Persistent Homology

<a id="genesis-ph"></a>

**Target namespace:** `qtenet.genesis.topology`

**Notes:** Boundary operators, persistence diagrams, Betti numbers.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/topology/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/boundary.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/distances.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/persistence.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/qtt_native.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/qtt_ph_gauntlet.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/simplicial.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/topology/README.md` (cat=genesis, lang=md, score=133)

### Candidate set (ranked)

- `ontic/genesis/topology/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/topology/boundary.py` (cat=genesis, score=143)
- `ontic/genesis/topology/distances.py` (cat=genesis, score=143)
- `ontic/genesis/topology/persistence.py` (cat=genesis, score=143)
- `ontic/genesis/topology/qtt_native.py` (cat=genesis, score=143)
- `ontic/genesis/topology/qtt_ph_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/topology/simplicial.py` (cat=genesis, score=143)
- `ontic/genesis/topology/README.md` (cat=genesis, score=133)
- `ontic/neural/genesis_optimizer.py` (cat=other, score=50)
- `qtt_native_gauntlet.py` (cat=qtt-misc, score=15)

---


## Genesis: QTT Geometric Algebra (GA)

<a id="genesis-ga"></a>

**Target namespace:** `qtenet.genesis.ga`

**Notes:** Clifford algebra, multivectors stored in QTT.

### Recommended canonical upstream sources (top picks)

- `ontic/genesis/ga/__init__.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/cga.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/multivector.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/operations.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/products.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/qtt_ga_gauntlet.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/qtt_multivector.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/rotors.py` (cat=genesis, lang=py, score=143)

### Candidate set (ranked)

- `ontic/genesis/ga/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/ga/cga.py` (cat=genesis, score=143)
- `ontic/genesis/ga/multivector.py` (cat=genesis, score=143)
- `ontic/genesis/ga/operations.py` (cat=genesis, score=143)
- `ontic/genesis/ga/products.py` (cat=genesis, score=143)
- `ontic/genesis/ga/qtt_ga_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/ga/qtt_multivector.py` (cat=genesis, score=143)
- `ontic/genesis/ga/rotors.py` (cat=genesis, score=143)
- `ontic/genesis/ga/README.md` (cat=genesis, score=133)

---


## GPU Acceleration: CUDA/Triton kernels and dispatch

<a id="gpu-accel"></a>

**Target namespace:** `qtenet.gpu`

**Notes:** Compute kernels, autotuning, GPU point-eval and apply operations.

### Recommended canonical upstream sources (top picks)

- `ontic/core/gpu.py` (cat=core, lang=py, score=150)
- `ontic/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `ontic/genesis/core/rsvd.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/gpu_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/gpu_qtt_proper.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/rsvd_qtt_compress.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/demos/triton_qtt.py` (cat=genesis, lang=py, score=143)

### Candidate set (ranked)

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
- *(+31 more candidates)*

---


## Enterprise SDK: stable facade (what users import)

<a id="sdk-facade"></a>

**Target namespace:** `qtenet.sdk`

**Notes:** The SDK layer should define the stable public surface.

### Recommended canonical upstream sources (top picks)

- `apps/sdk_legacy/qtt-sdk/examples/big_data_analytics.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/billion_point_real.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/digital_twin.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/generate_energy_decay_plot.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/irrefutable_proof.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/make_pdf.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, lang=py, score=103)

### Candidate set (ranked)

- `apps/sdk_legacy/qtt-sdk/examples/big_data_analytics.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/billion_point_real.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/digital_twin.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/generate_energy_decay_plot.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/irrefutable_proof.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/make_pdf.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/tests/test_qtt_sdk.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/src/qtt_sdk/__init__.py` (cat=sdk, score=101)
- `apps/sdk_legacy/qtt-sdk/src/qtt_sdk/core.py` (cat=sdk, score=101)
- `apps/sdk_legacy/qtt-sdk/src/qtt_sdk/operations.py` (cat=sdk, score=101)
- `apps/sdk_legacy/qtt-sdk/src/qtt_sdk/operators.py` (cat=sdk, score=101)
- `apps/sdk_legacy/qtt-sdk/CHANGELOG.md` (cat=sdk, score=95)
- `apps/sdk_legacy/qtt-sdk/README.md` (cat=sdk, score=95)
- `apps/sdk_legacy/qtt-sdk/examples/fluid_dynamics_certificate.json` (cat=sdk, score=93)
- `apps/sdk_legacy/qtt-sdk/examples/pressure_poisson_certificate.json` (cat=sdk, score=93)
- `apps/sdk_legacy/qtt-sdk/examples/proof_certificate.json` (cat=sdk, score=93)

---


## Oracle utilities: QTT slicing / encoding

<a id="oracle-slicing-encoding"></a>

**Target namespace:** `qtenet.integrations.oracle`

**Notes:** QTT encoders/slicers used for oracle-related workflows.

### Recommended canonical upstream sources (top picks)

- `fluidelite/fe_tci/context_encoder.py` (cat=fluidelite, lang=py, score=115)
- `fluidelite/noaa_slicer_real.py` (cat=fluidelite, lang=py, score=115)
- `apps/sdk_legacy/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, lang=py, score=103)
- `oracle/__init__.py` (cat=oracle, lang=py, score=65)
- `oracle/coinbase_oracle.py` (cat=oracle, lang=py, score=65)
- `oracle/cuda_graph_slicer.py` (cat=oracle, lang=py, score=65)
- `oracle/live_oracle_old.py` (cat=oracle, lang=py, score=65)

### Candidate set (ranked)

- `fluidelite/fe_tci/context_encoder.py` (cat=fluidelite, score=115)
- `fluidelite/noaa_slicer_real.py` (cat=fluidelite, score=115)
- `apps/sdk_legacy/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, score=103)
- `oracle/__init__.py` (cat=oracle, score=65)
- `oracle/coinbase_oracle.py` (cat=oracle, score=65)
- `oracle/cuda_graph_slicer.py` (cat=oracle, score=65)
- `oracle/live_oracle_old.py` (cat=oracle, score=65)
- `oracle/oracle_engine.py` (cat=oracle, score=65)
- `oracle/oracle_qtt_slicer.py` (cat=oracle, score=65)
- `oracle/qtt_encoder.py` (cat=oracle, score=65)
- `oracle/qtt_encoder_cuda.py` (cat=oracle, score=65)
- `oracle/triton_slicer.py` (cat=oracle, score=65)
- `ontic/exploit/state_encoder.py` (cat=other, score=50)
- `ontic/hypervisual/slicer.py` (cat=other, score=50)
- `ontic/hypervisual/slicing_core.py` (cat=other, score=50)
- `ontic/visualization/tensor_slicer.py` (cat=other, score=50)
- `demos/flagship_pipeline.py` (cat=demos, score=20)
- `demos/world_data_slicer.py` (cat=demos, score=20)
- `tests/test_exploit_state_encoder.py` (cat=other, score=10)
- `tests/test_tensor_slicer.py` (cat=other, score=10)
- `tests/test_visualization.py` (cat=other, score=10)

---


## Provenance & Attestation artifacts

<a id="provenance-attestation"></a>

**Target namespace:** `qtenet.provenance`

**Notes:** Run manifests, deterministic controls, attestations.

### Recommended canonical upstream sources (top picks)

- `ontic/core/determinism.py` (cat=core, lang=py, score=150)
- `ontic/cfd/koopman_tt.py` (cat=cfd, lang=py, score=135)
- `demos/pqc_sign_results.py` (cat=demos, lang=py, score=20)
- `demos/MILLENNIUM_HUNTER_ATTESTATION.json` (cat=demos, lang=json, score=10)
- `TURN_THE_KEY.py` (cat=other, lang=py, score=10)
- `docs/generate_attestation.py` (cat=other, lang=py, score=10)
- `femto_fabricator_gauntlet.py` (cat=other, lang=py, score=10)
- `oracle_node/server.py` (cat=other, lang=py, score=10)

### Candidate set (ranked)

- `ontic/core/determinism.py` (cat=core, score=150)
- `ontic/cfd/koopman_tt.py` (cat=cfd, score=135)
- `demos/pqc_sign_results.py` (cat=demos, score=20)
- `demos/MILLENNIUM_HUNTER_ATTESTATION.json` (cat=demos, score=10)
- `TURN_THE_KEY.py` (cat=other, score=10)
- `docs/generate_attestation.py` (cat=other, score=10)
- `femto_fabricator_gauntlet.py` (cat=other, score=10)
- `oracle_node/server.py` (cat=other, score=10)
- `prometheus_gauntlet.py` (cat=other, score=10)
- `proteome_compiler_gauntlet.py` (cat=other, score=10)
- `tools/tools/scripts/determinism_check.py` (cat=other, score=10)
- `sovereign_genesis_gauntlet.py` (cat=other, score=10)
- `tests/conftest.py` (cat=other, score=10)
- `tests/test_marrs_fusion.py` (cat=other, score=10)
- `tests/test_mpo_hamiltonians.py` (cat=other, score=10)
- `tests/test_sovereign.py` (cat=other, score=10)
- `tests/test_visualization.py` (cat=other, score=10)
- `tig011a_dynamic_validation.py` (cat=other, score=10)
- `tig011a_multimechanism.py` (cat=other, score=10)
- `CFD_HVAC/Attestations/TIER1_QTT_CFD_ATTESTATION.json` (cat=qtt-misc, score=5)
- `NS2D_QTT_NATIVE_ATTESTATION.json` (cat=qtt-misc, score=5)
- `QTT_MARRS_ATTESTATION.json` (cat=qtt-misc, score=5)
- `QTT_NATIVE_GAUNTLET_ATTESTATION.json` (cat=qtt-misc, score=5)
- `QTT_SEPARABLE_ATTESTATION.json` (cat=qtt-misc, score=5)
- `YM_PHASE_IV_4D_QTT_ATTESTATION.json` (cat=qtt-misc, score=5)
- `docs/QTT_BENCHMARK_ATTESTATION.json` (cat=qtt-misc, score=5)
- `artifacts/artifacts/evidence/drug_discovery/QTT_BENCHMARK_ATTESTATION.json` (cat=qtt-misc, score=5)
- `AI_MATHEMATICIAN_ATTESTATION.json` (cat=other, score=0)
- `CFD_HVAC/Attestations/TIER2_THERMAL_COMFORT_ATTESTATION.json` (cat=other, score=0)
- `CROSS_PRIMITIVE_PIPELINE_ATTESTATION.json` (cat=other, score=0)
- `FEMTO_FABRICATOR_ATTESTATION.json` (cat=other, score=0)
- `FEZK_V2_ATTESTATION.json` (cat=other, score=0)
- `FLUIDELITE_18TB_ATTESTATION.json` (cat=other, score=0)
- `GATE1_ATTESTATION.json` (cat=other, score=0)
- `GENESIS_BENCHMARK_ATTESTATION.json` (cat=other, score=0)
- `GENESIS_GAUNTLET_ATTESTATION.json` (cat=other, score=0)
- `NEURAL_CONNECTOME_REAL_ATTESTATION.json` (cat=other, score=0)
- `NEUROMORPHIC_INTEGRATION_ATTESTATION.json` (cat=other, score=0)
- `PRODUCTION_HARDENING_ATTESTATION.json` (cat=other, score=0)
- `PROMETHEUS_ATTESTATION.json` (cat=other, score=0)
- *(+7 more candidates)*
