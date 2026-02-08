# Provenance & Attestation

**Capability ID:** `provenance`

**Target namespace:** `qtenet.provenance`

## Summary

Define and emit run manifests and attach attestation artifacts where available.

## Notes

Run manifests, deterministic controls, attestations.

## Product invariants

- Every run has an identity
- Manifests are JSON-serializable

## Observability (enterprise requirements)

- Capture seeds, versions, operator identities

## Canonical upstream sources (recommended top picks)

- `tensornet/core/determinism.py` (cat=core, lang=py, score=150)
- `tensornet/cfd/koopman_tt.py` (cat=cfd, lang=py, score=135)
- `demos/forensic_hub.py` (cat=demos, lang=py, score=20)
- `demos/pqc_sign_results.py` (cat=demos, lang=py, score=20)
- `demos/MILLENNIUM_HUNTER_ATTESTATION.json` (cat=demos, lang=json, score=10)
- `TURN_THE_KEY.py` (cat=other, lang=py, score=10)
- `docs/generate_attestation.py` (cat=other, lang=py, score=10)
- `femto_fabricator_gauntlet.py` (cat=other, lang=py, score=10)

## Candidate set (ranked, truncated)

- `tensornet/core/determinism.py` (cat=core, score=150)
- `tensornet/cfd/koopman_tt.py` (cat=cfd, score=135)
- `demos/forensic_hub.py` (cat=demos, score=20)
- `demos/pqc_sign_results.py` (cat=demos, score=20)
- `demos/MILLENNIUM_HUNTER_ATTESTATION.json` (cat=demos, score=10)
- `TURN_THE_KEY.py` (cat=other, score=10)
- `docs/generate_attestation.py` (cat=other, score=10)
- `femto_fabricator_gauntlet.py` (cat=other, score=10)
- `oracle_node/server.py` (cat=other, score=10)
- `prometheus_gauntlet.py` (cat=other, score=10)
- `proteome_compiler_gauntlet.py` (cat=other, score=10)
- `scripts/determinism_check.py` (cat=other, score=10)
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
- `evidence/drug_discovery/QTT_BENCHMARK_ATTESTATION.json` (cat=qtt-misc, score=5)
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
- `PROTEOME_COMPILER_ATTESTATION.json` (cat=other, score=0)
- `SOVEREIGN_ACTIVATION_ATTESTATION.json` (cat=other, score=0)
- `SOVEREIGN_DAEMON_ATTESTATION.json` (cat=other, score=0)
- `docs/attestations/SOVEREIGN_ATTESTATION.md` (cat=other, score=0)
- `papers/YANGMILLS_MASSGAP_ATTESTATION.json` (cat=other, score=0)
- `papers/YANGMILLS_RIGOROUS_ATTESTATION.json` (cat=other, score=0)
- `transfer_matrix_proof_attestation.json` (cat=other, score=0)

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

