# Solvers: Euler/NS/PDE pipelines (IMEX/TDVP)

**Capability ID:** `solvers`

**Target namespace:** `qtenet.solvers`

## Summary

Provide solver entrypoints with strict rank-control and provenance outputs.

## Notes

End-to-end solver layer. Enterprise contract: reproducibility + rank control + diagnostics.

## Product invariants

- Every timestep must apply rank control
- Conservation/diagnostics computed without densification when feasible

## Observability (enterprise requirements)

- Emit per-step ranks + truncation error
- Emit conservation deltas

## Canonical upstream sources (recommended top picks)

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
- `ontic/cfd/flux_batch.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/kelvin_helmholtz.py` (cat=cfd, lang=py, score=135)

## Candidate set (ranked, truncated)

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
- `ontic/cfd/singularity_hunter.py` (cat=cfd, score=135)
- `ontic/cfd/stabilized_refine.py` (cat=cfd, score=135)
- `ontic/cfd/tci_flux.py` (cat=cfd, score=135)
- `ontic/cfd/tci_true.py` (cat=cfd, score=135)
- `ontic/cfd/thermal_qtt.py` (cat=cfd, score=135)
- `ontic/cfd/tt_cfd.py` (cat=cfd, score=135)
- `ontic/cfd/tt_poisson.py` (cat=cfd, score=135)
- `ontic/cfd/weno_native_tt.py` (cat=cfd, score=135)
- `ontic/cfd/weno_tt.py` (cat=cfd, score=135)
- `ontic/cfd/FluidElite.md` (cat=cfd, score=125)
- *(+18 more candidates in inventory)*

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

