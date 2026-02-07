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

- `tensornet/core/__init__.py` (cat=core, lang=py, score=150)
- `tensornet/core/decompositions.py` (cat=core, lang=py, score=150)
- `tensornet/core/mpo.py` (cat=core, lang=py, score=150)
- `tensornet/mpo/__init__.py` (cat=core, lang=py, score=150)
- `tensornet/mpo/atmospheric_solver.py` (cat=core, lang=py, score=150)
- `tensornet/mpo/laplacian_cuda.py` (cat=core, lang=py, score=150)
- `tensornet/mpo/operators.py` (cat=core, lang=py, score=150)
- `tensornet/mps/__init__.py` (cat=core, lang=py, score=150)
- `tensornet/mps/hamiltonians.py` (cat=core, lang=py, score=150)
- `tensornet/genesis/core/rsvd.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/core/triton_ops.py` (cat=genesis, lang=py, score=143)
- `tensornet/genesis/ga/rotors.py` (cat=genesis, lang=py, score=143)

## Candidate set (ranked, truncated)

- `tensornet/core/__init__.py` (cat=core, score=150)
- `tensornet/core/decompositions.py` (cat=core, score=150)
- `tensornet/core/mpo.py` (cat=core, score=150)
- `tensornet/mpo/__init__.py` (cat=core, score=150)
- `tensornet/mpo/atmospheric_solver.py` (cat=core, score=150)
- `tensornet/mpo/laplacian_cuda.py` (cat=core, score=150)
- `tensornet/mpo/operators.py` (cat=core, score=150)
- `tensornet/mps/__init__.py` (cat=core, score=150)
- `tensornet/mps/hamiltonians.py` (cat=core, score=150)
- `tensornet/genesis/core/rsvd.py` (cat=genesis, score=143)
- `tensornet/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `tensornet/genesis/ga/rotors.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/qtt_ot_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/transport_plan.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/wasserstein.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/resolvent.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/filters.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/laplacian.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/qtt_sgw_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/topology/boundary.py` (cat=genesis, score=143)
- `tensornet/genesis/tropical/matrix.py` (cat=genesis, score=143)
- `tensornet/cfd/adaptive_tt.py` (cat=cfd, score=135)
- `tensornet/cfd/comfort_metrics.py` (cat=cfd, score=135)
- `tensornet/cfd/euler2d_native.py` (cat=cfd, score=135)
- `tensornet/cfd/euler2d_strang.py` (cat=cfd, score=135)
- `tensornet/cfd/euler_1d.py` (cat=cfd, score=135)
- `tensornet/cfd/euler_nd_native.py` (cat=cfd, score=135)
- `tensornet/cfd/fast_euler_2d.py` (cat=cfd, score=135)
- `tensornet/cfd/fast_euler_3d.py` (cat=cfd, score=135)
- `tensornet/cfd/fast_vlasov_5d.py` (cat=cfd, score=135)
- `tensornet/cfd/flux_2d_tci.py` (cat=cfd, score=135)
- `tensornet/cfd/flux_batch.py` (cat=cfd, score=135)
- `tensornet/cfd/kelvin_helmholtz.py` (cat=cfd, score=135)
- `tensornet/cfd/koopman_tt.py` (cat=cfd, score=135)
- `tensornet/cfd/local_flux_native.py` (cat=cfd, score=135)
- `tensornet/cfd/nd_shift_mpo.py` (cat=cfd, score=135)
- `tensornet/cfd/ns2d_qtt_native.py` (cat=cfd, score=135)
- `tensornet/cfd/pure_qtt_ops.py` (cat=cfd, score=135)
- `tensornet/cfd/qtt.py` (cat=cfd, score=135)
- `tensornet/cfd/qtt_2d.py` (cat=cfd, score=135)
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

