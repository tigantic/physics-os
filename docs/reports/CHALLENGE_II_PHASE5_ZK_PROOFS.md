# Challenge II Phase 5: Trustless Binding Affinity Proofs

**Date:** 2026-02-27 20:52 UTC
**Author:** Bradly Biron Baker Adams

## Task 5.1: ZK Circuit for LJ Energy

| Molecule | Target | Constraints | Advice Cells | Pass |
|----------|--------|-------------|--------------|------|
| TIG-011a (erlotinib analogue) | EGFR T790M | 48 | 72 | ✓ |
| Nirmatrelvir analogue | SARS-CoV-2 Mpro | 48 | 72 | ✓ |
| Oseltamivir analogue | H5N1 Neuraminidase | 48 | 72 | ✓ |

## Task 5.2: Binding Minimum Proofs

| Molecule | Target | Min E | Threshold | Merkle Depth | Verified |
|----------|--------|-------|-----------|--------------|----------|
| TIG-011a (erlotinib analogue) | EGFR T790M | -5.0 | -2.0 | 15 | ✓ |
| Nirmatrelvir analogue | SARS-CoV-2 Mpro | -5.0 | -2.0 | 15 | ✓ |
| Oseltamivir analogue | H5N1 Neuraminidase | -5.0 | -2.0 | 15 | ✓ |

## Task 5.3: On-Chain Verifier

- Contract: `OnticBindingVerifier`
- Lines: 231
- Interface: `IBindingVerifier`
- Methods: verifyBinding, registerClaim, getClaim
- Source hash: `1a83448527dd02be79d9b4b6d3832f25...`

## Task 5.4: FDA IND Format

- Document: `docs/reports/FDA_IND_BINDING_EVIDENCE.txt`
- Hash: `ca04e63a7ca4ce1cc477b21575266831...`

## Task 5.5: IP Protection

| Molecule | Target | Commitment | LJ | Merkle | F-S | Binding |
|----------|--------|------------|----|---------|----|---------|
| TIG-011a (erlotinib analogue) | EGFR T790M | `c0757fae3543d2d5...` | ✓ | ✓ | ✓ | ✓ |
| Nirmatrelvir analogue | SARS-CoV-2 Mpro | `3c09b4582c2fa0b1...` | ✓ | ✓ | ✓ | ✓ |
| Oseltamivir analogue | H5N1 Neuraminidase | `d88e06ab038e826c...` | ✓ | ✓ | ✓ | ✓ |

## Exit Criteria

| Criterion | Status |
|-----------|--------|
| ZK proof of LJ energy | ✓ PASS |
| Binding minimum proof | ✓ PASS |
| On-chain verifier | ✓ PASS |
| FDA IND format | ✓ PASS |
| IP protection | ✓ PASS |

**Overall: ✓ PASS**
