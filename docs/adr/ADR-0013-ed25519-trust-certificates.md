# ADR-0013: Ed25519 for Trust Certificates

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

The Trustless Physics Certificate (TPC) system requires cryptographic signatures to attest that a simulation result was produced by a verified The Ontic Engine pipeline — bit-exact inputs, deterministic solver, verified conservation laws. The signature must be:

1. **Fast** — signing and verification must complete in <1 ms for pipeline throughput.
2. **Small** — certificate JSON payloads are stored alongside results; signature overhead matters.
3. **Deterministic** — no nonce-related side channels (unlike ECDSA without RFC 6979).
4. **Widely supported** — verification must work in Rust, Python, JavaScript, and Solidity.

Candidates evaluated: RSA-2048 (slow, large), ECDSA/secp256k1 (Ethereum-native but nonce-sensitive), Ed25519 (fast, small, deterministic).

## Decision

**All TPC signatures use Ed25519 (RFC 8032).** Specifically:

1. Key generation produces a 32-byte seed → 64-byte expanded private key + 32-byte public key.
2. Signing is performed by `proof_bridge` (Rust, via `ed25519-dalek`) or `ontic.tpc` (Python, via `PyNaCl`).
3. Certificates embed the 64-byte signature and 32-byte public key in base64.
4. On-chain verification uses a precompiled Ed25519 verifier contract (Solana-native, EVM via `Ed25519Verifier.sol`).
5. Key rotation follows a 90-day schedule with overlapping validity windows.

## Consequences

- **Easier:** Deterministic signatures — same message + key always produces the same signature (no k-nonce issues).
- **Easier:** 64-byte signatures vs. 256-byte RSA — smaller certificate payloads.
- **Easier:** Sub-millisecond sign/verify on commodity hardware.
- **Harder:** Not natively supported in EVM `ecrecover` — requires a dedicated verifier contract.
- **Risk:** Quantum vulnerability (all classical asymmetric crypto). Mitigated by documented migration path to CRYSTALS-Dilithium in ADR backlog.
