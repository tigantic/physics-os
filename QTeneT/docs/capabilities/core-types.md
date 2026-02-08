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

- `tensornet/core/__init__.py` (cat=core, lang=py, score=150)
- `tensornet/core/decompositions.py` (cat=core, lang=py, score=150)
- `tensornet/core/dense_guard.py` (cat=core, lang=py, score=150)
- `tensornet/core/determinism.py` (cat=core, lang=py, score=150)
- `tensornet/core/gpu.py` (cat=core, lang=py, score=150)
- `tensornet/core/mpo.py` (cat=core, lang=py, score=150)
- `tensornet/core/mps.py` (cat=core, lang=py, score=150)
- `tensornet/core/phase_deferred.py` (cat=core, lang=py, score=150)

## Candidate set (ranked, truncated)

- `tensornet/core/__init__.py` (cat=core, score=150)
- `tensornet/core/decompositions.py` (cat=core, score=150)
- `tensornet/core/dense_guard.py` (cat=core, score=150)
- `tensornet/core/determinism.py` (cat=core, score=150)
- `tensornet/core/gpu.py` (cat=core, score=150)
- `tensornet/core/mpo.py` (cat=core, score=150)
- `tensornet/core/mps.py` (cat=core, score=150)
- `tensornet/core/phase_deferred.py` (cat=core, score=150)
- `tensornet/core/profiling.py` (cat=core, score=150)
- `tensornet/core/states.py` (cat=core, score=150)
- `tensornet/mpo/__init__.py` (cat=core, score=150)
- `tensornet/mpo/atmospheric_solver.py` (cat=core, score=150)
- `tensornet/mpo/laplacian_cuda.py` (cat=core, score=150)
- `tensornet/mpo/operators.py` (cat=core, score=150)
- `tensornet/mps/__init__.py` (cat=core, score=150)
- `tensornet/mps/hamiltonians.py` (cat=core, score=150)
- `tensornet/genesis/__init__.py` (cat=genesis, score=145)
- `tensornet/genesis/core/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/core/exceptions.py` (cat=genesis, score=143)
- `tensornet/genesis/core/logging.py` (cat=genesis, score=143)
- `tensornet/genesis/core/rsvd.py` (cat=genesis, score=143)
- `tensornet/genesis/core/triton_ops.py` (cat=genesis, score=143)
- `tensornet/genesis/core/validation.py` (cat=genesis, score=143)
- `tensornet/genesis/ga/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/ga/qtt_ga_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/ga/qtt_multivector.py` (cat=genesis, score=143)
- `tensornet/genesis/ga/rotors.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/barycenters.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/cost_matrices.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/distributions.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/qtt_ot_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/sinkhorn_qtt.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/transport_plan.py` (cat=genesis, score=143)
- `tensornet/genesis/ot/wasserstein.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/gp.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/kernel_matrix.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/kernels.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/mmd.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/qtt_rkhs_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/rkhs/ridge.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/__init__.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/ensembles.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/free_probability.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/qtt_rmt_gauntlet.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/resolvent.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/spectral_density.py` (cat=genesis, score=143)
- `tensornet/genesis/rmt/universality.py` (cat=genesis, score=143)
- `tensornet/genesis/sgw/__init__.py` (cat=genesis, score=143)
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

