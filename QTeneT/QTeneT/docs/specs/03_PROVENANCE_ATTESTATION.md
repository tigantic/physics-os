# Spec: Provenance & Attestation

## Objective
Enterprise-grade reproducibility and audit trails for QTT computations.

## Minimum run manifest
Every non-trivial execution SHOULD produce a manifest:
- code version identifiers (git sha when available)
- environment: Python, torch, CUDA availability
- operator identities (see operator versioning)
- rank control parameters (eps/max_rank)
- input dataset identifiers (hash/path)
- output identifiers (hash/path)

## Attestations
Where upstream uses attestation JSONs, QTeneT treats them as first-class artifacts.

## Determinism
- Global seed controls must be centralized.
- Non-deterministic GPU ops must be disclosed in manifests.
