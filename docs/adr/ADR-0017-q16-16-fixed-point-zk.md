# ADR-0017: Q16.16 Fixed-Point for ZK Arithmetic

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

Zero-knowledge proof systems (Groth16, Halo2, STARKs) operate over prime fields where all arithmetic is modular integer operations. IEEE 754 floating-point is incompatible with ZK circuits because:

1. Floating-point arithmetic is not a field (rounding, special values, non-associativity).
2. Encoding a float as a field element requires prohibitively expensive range checks.
3. Different hardware produces different rounding results — violating the bit-exactness requirement for TPC attestation.

The Physics OS's ZK pipeline must prove that a CFD solver produced the claimed result. This requires the solver's arithmetic to be representable as native field operations.

## Decision

**All ZK-proven computations use Q16.16 fixed-point arithmetic (16-bit integer, 16-bit fraction).** Specifically:

1. Q16.16 values are stored as `i32` with implicit divisor of 2^16 (65536).
2. Range: [-32768.0, +32767.99998] with precision of ~1.5e-5.
3. Multiplication: `(a * b) >> 16` with explicit overflow checks.
4. All QTT core entries in ZK-proven paths are Q16.16.
5. The Rust crates `qtt_cem`, `qtt_fea`, and `qtt_opt` implement Q16.16 natively.
6. Conversion from f64 to Q16.16 is performed at the ZK boundary with documented precision loss.
7. Conservation law verification operates in Q16.16 with tolerance adjusted for fixed-point precision.

## Consequences

- **Easier:** Direct mapping to Halo2/Groth16 field elements — no floating-point emulation circuits.
- **Easier:** Bit-exact reproducibility across all platforms (no IEEE 754 rounding variance).
- **Easier:** Circuit size is O(N) for N multiplications, not O(N · mantissa_bits).
- **Harder:** Precision limited to ~1.5e-5 — insufficient for some scientific computations. Mitigated by using f64 for non-ZK paths and validating that Q16.16 precision is adequate for the specific conservation check.
- **Risk:** Overflow in intermediate products for large field values. Mitigated by `i64` intermediates with explicit saturation and overflow traps in debug builds.
