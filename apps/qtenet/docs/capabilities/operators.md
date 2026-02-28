# Operators Library (MPO builders)

**Capability ID:** `operators`

**Target namespace:** `qtenet.operators`

## Summary

Expose a stable set of operator builders with scheme/version metadata.

## Notes

Versioned operator builders returning MPO + metadata.

## Product invariants

- Operators are versioned identities (name/scheme/version)
- Operators must be composable without densification

## Observability (enterprise requirements)

- Operator meta is JSON-serializable
- Record operator identities in run manifests

## Canonical upstream sources (recommended top picks)

- `ontic/core/__init__.py` (cat=core, lang=py, score=150)
- `ontic/core/decompositions.py` (cat=core, lang=py, score=150)
- `ontic/core/mpo.py` (cat=core, lang=py, score=150)
- `ontic/mpo/__init__.py` (cat=core, lang=py, score=150)
- `ontic/mpo/atmospheric_solver.py` (cat=core, lang=py, score=150)
- `ontic/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `ontic/mpo/operators.py` (cat=core, lang=py, score=150)
- `ontic/mps/__init__.py` (cat=core, lang=py, score=150)
- `ontic/mps/hamiltonians.py` (cat=core, lang=py, score=150)
- `ontic/genesis/core/rsvd.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ga/rotors.py` (cat=genesis, lang=py, score=143)

## Candidate set (ranked, truncated)

- `ontic/core/__init__.py` (cat=core, score=150)
- `ontic/core/decompositions.py` (cat=core, score=150)
- `ontic/core/mpo.py` (cat=core, score=150)
- `ontic/mpo/__init__.py` (cat=core, score=150)
- `ontic/mpo/atmospheric_solver.py` (cat=core, score=150)
- `ontic/mpo/laplacian_cuda.py` (cat=core, score=150)
- `ontic/mpo/operators.py` (cat=core, score=150)
- `ontic/mps/__init__.py` (cat=core, score=150)
- `ontic/mps/hamiltonians.py` (cat=core, score=150)
- `ontic/genesis/core/rsvd.py` (cat=genesis, score=143)
- `ontic/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `ontic/genesis/ga/rotors.py` (cat=genesis, score=143)
- `ontic/genesis/ot/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `ontic/genesis/ot/qtt_ot_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, score=143)
- `ontic/genesis/ot/wasserstein.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/resolvent.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/filters.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/laplacian.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/qtt_sgw_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/topology/boundary.py` (cat=genesis, score=143)
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
- *(+177 more candidates in inventory)*

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

