# ADR-0019: Halo2 for ZK Circuits

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

The Trustless Physics Certificate (TPC) pipeline requires zero-knowledge proofs that a simulation was executed correctly. The ZK system must:

1. Prove conservation law satisfaction for meshes up to 128³ (2M+ cells).
2. Generate proofs in <60 seconds for real-time digital twin use cases.
3. Verify proofs in <10 ms for on-chain settlement.
4. Support recursive composition (prove a batch of simulations in one proof).

Candidates evaluated:

| System | Proof Size | Prover Time (1M gates) | Verifier Time | Trusted Setup |
|--------|-----------|----------------------|---------------|---------------|
| Groth16 | 128 B | ~30 s | ~3 ms | Per-circuit |
| PLONK/Halo2 | ~5 KB | ~45 s | ~5 ms | Universal (KZG) |
| STARKs | ~50 KB | ~60 s | ~20 ms | None |
| Nova/folding | Variable | ~10 s (IVC step) | ~5 ms | Universal |

Halo2 (PLONKish) was selected because: (1) universal trusted setup (KZG ceremony, not per-circuit); (2) custom gates enable efficient fixed-point arithmetic without generic R1CS overhead; (3) lookup tables for range checks and Q16.16 multiplication; (4) Rust-native implementation by Privacy Scaling Explorations (PSE); (5) EVM verifier generation for on-chain attestation.

## Decision

**All TPC zero-knowledge circuits use Halo2 (PLONKish arithmetization with KZG commitments).** Specifically:

1. Circuit definitions live in `crates/fluidelite_circuits/` and `crates/fluidelite_zk/`.
2. Custom gates implement Q16.16 multiplication, addition, and comparison.
3. Lookup tables provide range checks [0, 2^16) and precomputed trigonometric values.
4. Proof generation is performed by `proof_bridge` (Rust) invoked from Python via `ontic_bridge`.
5. Verifier contracts are generated and deployed to `contracts/FluidEliteHalo2Verifier.sol`.
6. Groth16 is retained as a secondary option (via `fluidelite-zk`'s `groth16_real` binary) for applications requiring minimal proof size.

## Consequences

- **Easier:** Universal setup — no per-circuit ceremony, new circuits deploy without new trusted setup.
- **Easier:** Custom gates reduce constraint count by 3–5× vs. R1CS for fixed-point arithmetic.
- **Easier:** Rust-native — integrates directly with the `crates/` workspace.
- **Harder:** Proof size (~5 KB) larger than Groth16 (128 B) — higher on-chain gas cost.
- **Risk:** KZG ceremony compromise. Mitigated by using the Ethereum KZG ceremony (100K+ participants) and documenting migration path to FRI-based (transparent) commitments.
