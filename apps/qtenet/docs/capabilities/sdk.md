# Enterprise SDK: stable facade (what users import)

**Capability ID:** `sdk`

**Target namespace:** `qtenet.sdk`

## Summary

Stabilize naming, types, and error semantics for external users.

## Notes

The SDK layer defines the stable public surface.

## Product invariants

- API stability contract enforced
- Errors are typed and actionable

## Observability (enterprise requirements)

- Emit JSON-friendly error payloads in CLI

## Canonical upstream sources (recommended top picks)

- `apps/sdk_legacy/qtt-sdk/examples/big_data_analytics.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/billion_point_real.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/digital_twin.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/generate_energy_decay_plot.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/irrefutable_proof.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/make_pdf.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_fluid_dynamics.py` (cat=sdk, lang=py, score=103)
- `apps/sdk_legacy/qtt-sdk/examples/proof_pressure_poisson.py` (cat=sdk, lang=py, score=103)

## Candidate set (ranked, truncated)

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

