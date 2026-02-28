# Construction: TCI / TT-Cross

**Capability ID:** `tci-ttcross`

**Target namespace:** `qtenet.tci`

## Summary

Build QTT representations from black-box functions or samples with evaluation budgets.

## Notes

Black-box function → QTT construction. Includes sampling policies and maxvol/skeleton selection.

## Product invariants

- Bound evaluations (n_evals tracked)
- Rank selection is explicit and auditable

## Observability (enterprise requirements)

- Emit n_evals, convergence/sweep stats
- Emit skeleton indices / maxvol diagnostics

## Canonical upstream sources (recommended top picks)

- `ontic/genesis/ot/cost_matrices.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/distributions.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/sinkhorn_qtt.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/transport_plan.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/ot/wasserstein.py` (cat=genesis, lang=py, score=143)
- `ontic/genesis/tropical/qtt_native.py` (cat=genesis, lang=py, score=143)
- `ontic/cfd/euler2d_native.py` (cat=cfd, lang=py, score=135)
- `ontic/cfd/euler2d_strang.py` (cat=cfd, lang=py, score=135)

## Candidate set (ranked, truncated)

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
- `tci_llm/generalized_tci.py` (cat=tci, score=100)
- `tci_llm/qtt.py` (cat=tci, score=100)
- `tci_llm/svd_llm.py` (cat=tci, score=100)
- `tci_llm/tci_llm.py` (cat=tci, score=100)
- `tci_llm/tests/__init__.py` (cat=tci, score=100)
- `tci_llm/tests/test_tci_llm.py` (cat=tci, score=100)
- `crates/tci_core_rust/python/tci_core/__init__.py` (cat=tci, score=98)
- `crates/tci_core_rust/src/indices.rs` (cat=tci, score=90)
- `crates/tci_core_rust/src/lib.rs` (cat=tci, score=90)
- `crates/tci_core_rust/src/maxvol.rs` (cat=tci, score=90)
- *(+37 more candidates in inventory)*

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

