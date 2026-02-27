# ZK Circuit Bounty Hunt Summary

## Date: January 2025
## Methodology: ORACLE Circom Parser + Manual Analysis

---

## Executive Summary

Analyzed 11 ZK circuit repositories targeting projects with active bug bounties.
Found **1 confirmed HIGH-severity finding** in Polygon zkEVM proverjs.

---

## Findings Table

| Repository | Finding | Severity | Bounty | Status |
|------------|---------|----------|--------|--------|
| **polygon-proverjs** | Unconstrained `rootC[4]` public input in recursive1.circom | **HIGH** | $1M (Polygon) | ✅ READY TO SUBMIT |
| hermez-circuits | Weak constraints only (not exploitable) | LOW | Polygon? | ❌ Not submittable |
| railgun-circuits-v2 | False positives (Num2Bits already constrains leafIndex) | N/A | Unknown | ❌ Not submittable |
| maci-circuits | `commandPollId` and `commandSalt` unconstrained | MEDIUM | **NO BOUNTY** | ❌ No bounty |
| zkp2p-circuits | Demo code with commented-out constraints | N/A | NO BOUNTY | ❌ Not production |
| circomlib | Foundational library, findings are likely design patterns | N/A | N/A | ❌ Skip |
| scroll-circuits | Uses Halo2/Rust, not Circom | N/A | $1M (Scroll) | ❌ Need Rust analyzer |
| zksync-circuits | Uses Rust, not Circom | N/A | $1.1M | ❌ Need Rust analyzer |
| linea-circuits | Uses gnark/Go, not Circom | N/A | $100K | ❌ Need Go analyzer |

---

## Finding #1: Polygon zkEVM - Unconstrained rootC

### Location
- **File:** `recursive/recursive1.circom`
- **Lines:** 9, 78
- **Repository:** https://github.com/0xPolygonHermez/zkevm-proverjs

### Description

The `rootC[4]` signal is declared as a **public input** but is **never constrained** in the circuit:

```circom
signal input rootC[4];  // Line 9 - declared but never used!
// ...
component main {public [publics, rootC]}= Main();  // Line 78 - made public
```

The StarkVerifier from `c12a.verifier.circom` has `rootC` hardcoded internally (generated without `--verkeyInput`), so there's no `vA.rootC` input to connect to.

### Impact

- **Proof Malleability:** A prover can generate valid proofs with arbitrary `rootC` values
- **4 Unconstrained Public Inputs:** The R1CS has 48 public inputs, but only 44 are constrained
- **Violated Assumption:** The proof claims to verify a certain `rootC` but actually uses hardcoded values

### Severity: HIGH

Using Immunefi's severity matrix:
- **Impact:** High (affects soundness of recursive STARK verification)
- **Probability:** Medium (requires understanding of the system)

### Recommended Fix

**Option 1:** Remove `rootC` from public inputs (since it's not constrained):
```circom
// Change from:
component main {public [publics, rootC]}= Main();
// To:
component main {public [publics]}= Main();
```

**Option 2:** Generate `c12a.verifier.circom` with `--verkeyInput` and add:
```circom
vA.rootC <== rootC;
```

### Full Details

See: [POLYGON_FINDING_001.md](./POLYGON_FINDING_001.md)

---

## Finding #2: MACI - Unconstrained commandPollId/commandSalt

### Location
- **File:** `packages/circuits/circom/utils/full/StateLeafAndBallotTransformer.circom`
- **Lines:** 47, 49

### Description

Both `commandPollId` and `commandSalt` are declared as inputs but never passed to `MessageValidatorFull()` or used in any constraint.

### Impact

- Proof malleability for vote commands
- Poll ID and salt can be set arbitrarily

### Status: NO BOUNTY

MACI does not have an active bug bounty program on Immunefi or elsewhere.

---

## Repositories Analyzed

### Circom-Based (Analyzable)
1. **polygon-proverjs** - ✅ FOUND VULNERABILITY
2. **hermez-circuits** - Weak constraints only
3. **railgun-circuits-v2** - False positives
4. **maci-circuits** - Findings but no bounty
5. **zkp2p-circuits** - Demo code
6. **circomlib** - Library, not target
7. **TS-Circom (Term Structure)** - 144 templates analyzed, well-designed with proper division handling
8. **semaphore** - Properly constrained, no issues found
9. **protocols (DeGate/Loopring)** - C++ libsnark circuits, proper div-by-zero checks

### Non-Circom (Need Different Tools)
10. **scroll-circuits** - Halo2/Rust
11. **zksync-circuits** - Rust
12. **linea-circuits** - gnark/Go
13. **noir-lang** - Noir language (not ZK circuits repo)
14. **appliedzkp-maci** - Duplicate of maci-circuits
15. **light-protocol** - No Circom circuits found (Rust-based)

---

## Next Steps

### Immediate
1. **Submit Polygon Finding** - High-severity finding ready for Immunefi submission under "Primacy of Impact"

### Future
2. **Expand ORACLE** - Add Halo2/Rust analyzer for Scroll
3. **Add gnark support** - For Linea circuits
4. **Monitor new repos** - Watch for Circom updates from major projects

---

## ORACLE Parser Statistics

| Repository | Circom Files | Findings | Critical | High | Medium |
|------------|-------------|----------|----------|------|--------|
| polygon-proverjs | 2 | 1 | 1 | 0 | 0 |
| hermez-circuits | 15 | 2 | 0 | 2 | 0 |
| maci-circuits | 36 | 11 | 11 | 0 | 0 |
| railgun-circuits-v2 | 3 | 2 | 0 | 0 | 2 |
| zkp2p-circuits | 75 | ~100 | 5 | Many | Many |
| circomlib | 104 | 90 | N/A | N/A | N/A |

Note: Many findings are false positives or design patterns. Manual review is required.

---

## Bounty Potential

| Program | Max Bounty | Finding Severity | Estimated Reward |
|---------|------------|------------------|------------------|
| Polygon | $1,000,000 | HIGH | $20,000 (flat) |

Polygon's bounty structure:
- Critical: $50K-$1M (10% of funds at risk)
- High: $20K flat
- Medium: $5K flat

The rootC finding would likely be classified as **HIGH** due to:
- Real soundness weakness
- Production code
- Mitigated by downstream circuits (reduces to HIGH from CRITICAL)

---

*Generated by ORACLE ZK Circuit Analyzer*
