# Core Types: QTTTensor / TT cores / MPS / MPO

**Capability ID:** `core-types`

**Target namespace:** `qtenet.core`

## Summary

Defines the canonical runtime objects representing QTT/TT tensors and linear operators.

## Notes

Canonical tensor network types and containers. This is the foundation layer.

## Product invariants

- Never Go Dense by default
- Tensor metadata must be carried (dims/ranks/layout/dtype)
- All ops must be rank-controlled (eps/max_rank)

## Observability (enterprise requirements)

- Expose ranks before/after ops
- Emit truncation error and time cost

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
- `ontic/genesis/rkhs/qtt_rkhs_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rkhs/ridge.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/__init__.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/free_probability.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/resolvent.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- `ontic/genesis/rmt/universality.py` (cat=genesis, score=143)
- `ontic/genesis/sgw/__init__.py` (cat=genesis, score=143)
- *(+537 more candidates in inventory)*

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

