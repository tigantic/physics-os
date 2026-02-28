# Taxonomy Triage — `other` Bucket

**Classification:** Proprietary & Confidential

This report triages artifacts currently categorized as **`other`** in `inventory/qtt_repo_index.json`.

Goal: reduce due-diligence risk by reclassifying true QTT artifacts and explicitly excluding noise.

## Summary

- Total `other`: **437**
- Language breakdown: `{'py': 226, 'rs': 59, 'json': 57, 'md': 95}`

### Proposed reclassification breakdown

- **unresolved**: 271
- **core**: 122
- **tci**: 20
- **cfd**: 14
- **gpu**: 7
- **genesis**: 3

## Action Plan

### 1) Auto-reclassify candidates (high confidence)

Criteria: score ≥ 50 and proposed_category present. Count: **67**

Top 50:

- `tig011a_multimechanism.py` → **tci** (score=155, lang=py)
- `ontic/exploit/state_encoder.py` → **core** (score=130, lang=py)
- `ontic/zk/fluidelite_circuit_analyzer.py` → **tci** (score=125, lang=py)
- `ontic/docs/user_guides.py` → **core** (score=120, lang=py)
- `yangmills/__init__.py` → **core** (score=120, lang=py)
- `crates/ontic_core/src/lib.rs` → **core** (score=118, lang=rs)
- `compress_crypto_data.py` → **core** (score=105, lang=py)
- `proofs/proof_millennium.py` → **core** (score=105, lang=py)
- `ontic/visualization/tensor_slicer.py` → **gpu** (score=105, lang=py)
- `crates/tci_core/src/lib.rs` → **tci** (score=98, lang=rs)
- `fluidelite_ingest.py` → **core** (score=90, lang=py)
- `ontic/types/genesis_integration.py` → **core** (score=90, lang=py)
- `crates/fluidelite_zk/src/bin/fluid_block.rs` → **core** (score=88, lang=rs)
- `ontic/benchmarks/profile_bottlenecks.py` → **cfd** (score=80, lang=py)
- `ontic/types/__init__.py` → **core** (score=80, lang=py)
- `crates/fluidelite_zk/src/lib.rs` → **core** (score=78, lang=rs)
- `ontic/exploit/__init__.py` → **tci** (score=75, lang=py)
- `ontic/exploit/bounty_api.py` → **tci** (score=75, lang=py)
- `ontic/exploit/constants.py` → **tci** (score=75, lang=py)
- `ontic/exploit/euler_v2_hunt.py` → **tci** (score=75, lang=py)
- `ontic/exploit/exploit_hunter.py` → **tci** (score=75, lang=py)
- `ontic/exploit/morpho_blue_hunt.py` → **tci** (score=75, lang=py)
- `ontic/exploit/tx_space.py` → **tci** (score=75, lang=py)
- `ontic/substrate/morton_ops.py` → **core** (score=75, lang=py)
- `tests/audit_layer_4.py` → **core** (score=75, lang=py)
- `tests/benchmarks/optimized_pipeline_benchmark.py` → **tci** (score=75, lang=py)
- `tests/test_tci_integration.py` → **tci** (score=75, lang=py)
- `yangmills/ground_state_cuda.py` → **gpu** (score=75, lang=py)
- `yangmills/tests/test_gate5.py` → **core** (score=75, lang=py)
- `crates/fluidelite_zk/src/bin/fluid_s3_native.rs` → **core** (score=73, lang=rs)
- `decompress_crypto_data.py` → **core** (score=65, lang=py)
- `elite_yang_mills_proof.py` → **core** (score=65, lang=py)
- `femto_fabricator_gauntlet.py` → **core** (score=65, lang=py)
- `live_orderbook_fluid.py` → **core** (score=65, lang=py)
- `proofs/proof_master.py` → **core** (score=65, lang=py)
- `proofs/proof_phase_2.py` → **cfd** (score=65, lang=py)
- `proofs/proof_phase_3.py` → **core** (score=65, lang=py)
- `proteome_compiler_gauntlet.py` → **core** (score=65, lang=py)
- `real_yang_mills_engine.py` → **core** (score=65, lang=py)
- `tools/tools/scripts/check_import_cycles.py` → **core** (score=65, lang=py)
- `tools/tools/scripts/full_reproduce.py` → **core** (score=65, lang=py)
- `tools/tools/scripts/setup/hypertensor_core.py` → **core** (score=65, lang=py)
- `ontic/__init__.py` → **core** (score=65, lang=py)
- `ontic/gateway/orbital_command.py` → **core** (score=65, lang=py)
- `ontic/neural/genesis_optimizer.py` → **core** (score=65, lang=py)
- `ontic/sovereign/heatmap_generator.py` → **gpu** (score=65, lang=py)
- `ontic/substrate/field.py` → **core** (score=65, lang=py)
- `tests/test_integration.py` → **cfd** (score=65, lang=py)
- `tests/test_mpo_solver.py` → **cfd** (score=65, lang=py)
- `tests/test_visualization.py` → **core** (score=65, lang=py)

### 2) Investigate candidates (medium confidence)

Criteria: 25 ≤ score < 50. Count: **290**

Top 50:

- `TURN_THE_KEY.py` (score=40, lang=py)
- `ade_gauntlet.py` (score=40, lang=py)
- `ai_mathematician.py` (score=40, lang=py)
- `ai_scientist/__init__.py` (score=40, lang=py)
- `ai_scientist/conjecturer.py` (score=40, lang=py)
- `ai_scientist/demo.py` (score=40, lang=py)
- `ai_scientist/formalizer.py` (score=40, lang=py)
- `ai_scientist/pipeline.py` (score=40, lang=py)
- `chronos_gauntlet.py` (score=40, lang=py)
- `cornucopia_gauntlet.py` (score=40, lang=py)
- `docs/generate_attestation.py` (score=40, lang=py)
- `elite_yang_mills_proof_v2.py` (score=40, lang=py)
- `genesis_benchmark_suite.py` (score=40, lang=py)
- `live_market_fluid.py` (score=40, lang=py)
- `navier_stokes_millennium_pipeline.py` (score=40, lang=py)
- `ns_unified_black_swan.py` (score=40, lang=py)
- `ns_unified_black_swan_v2.py` (score=40, lang=py)
- `oracle_node/calibration.py` (score=40, lang=py)
- `oracle_node/server.py` (score=40, lang=py)
- `production_hardening_gauntlet.py` (score=40, lang=py)
- `prometheus_gauntlet.py` (score=40, lang=py)
- `proofs/proof_engine/Certified.py` (score=40, lang=py)
- `proofs/proof_engine/Certified_ARB.py` (score=40, lang=py)
- `proofs/proof_engine/__init__.py` (score=40, lang=py)
- `proofs/proof_engine/certificate.py` (score=40, lang=py)
- `proofs/proof_engine/constructive_qft.py` (score=40, lang=py)
- `proofs/proof_engine/lean_export.py` (score=40, lang=py)
- `proofs/cap_full_power.py` (score=40, lang=py)
- `proofs/proof_level_3b.py` (score=40, lang=py)
- `proofs/proof_markets_pipeline.py` (score=40, lang=py)
- `proofs/proof_molecular_pipeline.py` (score=40, lang=py)
- `tools/tools/scripts/generate_api_docs.py` (score=40, lang=py)
- `tools/tools/scripts/generate_pipeline_diagram.py` (score=40, lang=py)
- `tools/tools/scripts/setup/realtime_renderer.py` (score=40, lang=py)
- `tools/tools/scripts/testing/test_implicit_concept.py` (score=40, lang=py)
- `tools/tools/scripts/testing/test_phase4_integration.py` (score=40, lang=py)
- `tools/tools/scripts/testing/test_phase4_validation.py` (score=40, lang=py)
- `sovereign_daemon.py` (score=40, lang=py)
- `sovereign_genesis_gauntlet.py` (score=40, lang=py)
- `ontic/discovery/__main__.py` (score=40, lang=py)
- `ontic/discovery/api/gpu.py` (score=40, lang=py)
- `ontic/discovery/api/server.py` (score=40, lang=py)
- `ontic/discovery/engine_v2.py` (score=40, lang=py)
- `ontic/discovery/ingest/markets.py` (score=40, lang=py)
- `ontic/discovery/ingest/molecular.py` (score=40, lang=py)
- `ontic/discovery/ingest/plasma.py` (score=40, lang=py)
- `ontic/discovery/pipelines/markets_pipeline.py` (score=40, lang=py)
- `ontic/discovery/pipelines/molecular_pipeline.py` (score=40, lang=py)
- `ontic/discovery/primitives/__init__.py` (score=40, lang=py)
- `ontic/discovery/primitives/geometric_algebra.py` (score=40, lang=py)

### 3) Exclude candidates (likely noise / non-QTT)

Criteria: score < 0. Count: **0**

Top 50:


## Notes

- This triage is heuristic; it should be followed by a manual review pass of the top-ranked 100–200 items.
- The canonicalization engine can be extended with manual overrides once the reclassification is complete.
- Full machine-readable output: `inventory/other_triage.json`

