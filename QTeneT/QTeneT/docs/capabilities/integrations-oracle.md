# Integration: Oracle slicing/encoding

**Capability ID:** `integrations-oracle`

**Target namespace:** `qtenet.integrations.oracle`

## Summary

Provide stable adapters for oracle systems without contaminating core APIs.

## Notes

QTT encoders/slicers used for oracle workflows.

## Product invariants

- Adapters do not own core tensor semantics

## Observability (enterprise requirements)

- Emit schema/version for encoded artifacts

## Canonical upstream sources (recommended top picks)

- `fluidelite/fe_tci/context_encoder.py` (cat=fluidelite, lang=py, score=115)
- `fluidelite/noaa_slicer_real.py` (cat=fluidelite, lang=py, score=115)
- `fluidelite/qtt_physics.py` (cat=fluidelite, lang=py, score=115)
- `fluidelite/qtt_physics_100m.py` (cat=fluidelite, lang=py, score=115)
- `fluidelite/qtt_tci.py` (cat=fluidelite, lang=py, score=115)
- `sdk/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, lang=py, score=103)
- `sdk/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, lang=py, score=103)
- `oracle/__init__.py` (cat=oracle, lang=py, score=65)

## Candidate set (ranked, truncated)

- `fluidelite/fe_tci/context_encoder.py` (cat=fluidelite, score=115)
- `fluidelite/noaa_slicer_real.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_physics.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_physics_100m.py` (cat=fluidelite, score=115)
- `fluidelite/qtt_tci.py` (cat=fluidelite, score=115)
- `sdk/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, score=103)
- `sdk/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, score=103)
- `oracle/__init__.py` (cat=oracle, score=65)
- `oracle/coinbase_oracle.py` (cat=oracle, score=65)
- `oracle/cuda_graph_slicer.py` (cat=oracle, score=65)
- `oracle/live_oracle_old.py` (cat=oracle, score=65)
- `oracle/oracle_engine.py` (cat=oracle, score=65)
- `oracle/oracle_qtt_slicer.py` (cat=oracle, score=65)
- `oracle/qtt_encoder.py` (cat=oracle, score=65)
- `oracle/qtt_encoder_cuda.py` (cat=oracle, score=65)
- `oracle/triton_slicer.py` (cat=oracle, score=65)
- `tensornet/exploit/evm_oracle.py` (cat=other, score=50)
- `tensornet/exploit/hypergrid.py` (cat=other, score=50)
- `tensornet/exploit/invariants.py` (cat=other, score=50)
- `tensornet/exploit/morpho_blue_hunt.py` (cat=other, score=50)
- `tensornet/exploit/state_encoder.py` (cat=other, score=50)
- `tensornet/exploit/toy_contracts.py` (cat=other, score=50)
- `tensornet/hypervisual/slicer.py` (cat=other, score=50)
- `tensornet/hypervisual/slicing_core.py` (cat=other, score=50)
- `tensornet/visualization/tensor_slicer.py` (cat=other, score=50)
- `demos/flagship_pipeline.py` (cat=demos, score=20)
- `demos/resolution_independence.py` (cat=demos, score=20)
- `demos/world_data_slicer.py` (cat=demos, score=20)
- `oracle_node/calibration.py` (cat=other, score=10)
- `oracle_node/server.py` (cat=other, score=10)
- `tests/test_exploit_state_encoder.py` (cat=other, score=10)
- `tests/test_tensor_slicer.py` (cat=other, score=10)
- `tests/test_visualization.py` (cat=other, score=10)

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

