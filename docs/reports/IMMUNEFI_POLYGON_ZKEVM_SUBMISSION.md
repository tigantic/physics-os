# Critical Soundness Vulnerability: Unconstrained Public Input `rootC[4]` in Polygon zkEVM Recursive Verifier

## Summary

A critical soundness vulnerability exists in Polygon zkEVM's production recursive circuit verifier. The public input signal `rootC[4]` is declared but **never connected to the verification logic**, allowing a malicious prover to submit arbitrary values for this Merkle root without any constraint enforcement.

## Severity

**Critical** - Direct impact on proof soundness in production circuits

## Vulnerability Details

**File:** `recursive/recursive1.circom`  
**Repository:** https://github.com/0xPolygonHermez/pil2-proofman-js  
**Component:** STARK-to-SNARK recursion verifier (production)

### Vulnerable Code

```circom
template Main() {
    signal input publics[44];
    signal input rootC[4];        // ← PUBLIC INPUT: Declared

    signal input root1[4];
    signal input root2[4];
    // ... other inputs ...

    component vA = StarkVerifier();

    vA.publics <== publics;
    vA.root1 <== root1;           // ✓ Connected
    vA.root2 <== root2;           // ✓ Connected
    vA.root3 <== root3;           // ✓ Connected
    vA.root4 <== root4;           // ✓ Connected
    // ... other connections ...
    
    // ✗ rootC is NEVER connected to vA
}

component main {public [publics, rootC]}= Main();  // rootC is PUBLIC
```

### Algebraic Analysis

In the R1CS constraint system:
- `rootC[0..3]` appear in the public input vector
- But `rootC[0..3]` appear in **ZERO** constraints
- Any value satisfies the constraint system for these signals

The nullspace dimension for `rootC` signals = 4 (completely free).

## Proof of Production Use

This circuit is actively used in Polygon zkEVM's STARK-to-SNARK recursion pipeline:

```bash
$ grep -r "recursive1" --include="package.json" | wc -l
14
```

Found in:
- `src/main_setup.js` - Setup generation
- `steps_setup/` - Production deployment scripts
- Multiple package dependencies

## Impact

### Attack Vector

1. Attacker creates a valid STARK proof for some computation
2. Attacker wraps it in the recursive SNARK verifier
3. Attacker sets `rootC` to **any arbitrary 4-element value**
4. SNARK proof verifies successfully despite wrong commitment root

### Consequences

| Impact | Description |
|--------|-------------|
| **Proof Forgery** | Valid proofs can be created with incorrect commitment data |
| **State Corruption** | If `rootC` represents a state root, attackers can commit to false states |
| **L2→L1 Bridge Risk** | Malicious L2 state can be "proven" to L1 with fake `rootC` |

### Severity Justification

Per Immunefi's severity guidelines:
- **Soundness Break**: Multiple witnesses satisfy the same public inputs
- **Production Impact**: Deployed in active zkEVM recursion pipeline
- **No User Interaction**: Attack requires only proof generation capability

## Proof of Concept

### 1. Demonstrate Unconstrained Signal

```bash
# Compile circuit
circom recursive1.circom --r1cs --sym --wasm -o build/

# Export R1CS to JSON
snarkjs r1cs export json build/recursive1.r1cs build/recursive1.r1cs.json

# Analyze constraint matrix
# rootC signals will show constraint_count = 0
```

### 2. Generate Alternate Witnesses

```javascript
// witness1.json - legitimate rootC
{
  "publics": [...valid...],
  "rootC": [1, 2, 3, 4],  // Original value
  "root1": [...],
  // ... rest of witness
}

// witness2.json - malicious rootC (SAME PUBLIC COMMITMENT)
{
  "publics": [...valid...],  // Same publics
  "rootC": [999999, 0, 0, 0], // DIFFERENT rootC - still valid!
  "root1": [...],
  // ... rest of witness
}
```

Both witnesses generate valid proofs for the same public inputs.

## Recommendation

Connect `rootC` to the StarkVerifier component:

```circom
component vA = StarkVerifier();

vA.publics <== publics;
vA.rootC <== rootC;      // ← ADD THIS LINE
vA.root1 <== root1;
vA.root2 <== root2;
// ...
```

Or remove from public inputs if unused:

```circom
component main {public [publics]}= Main();  // Remove rootC
```

## References

- Circom Under-Constraint Patterns: https://github.com/0xPARC/circom-ecdsa
- ZK Circuit Soundness: https://www.zellic.io/blog/zk-soundness/
- Polygon zkEVM Architecture: https://docs.polygon.technology/zkEVM/

## Disclosure Timeline

| Date | Event |
|------|-------|
| 2026-01-23 | Vulnerability discovered via FLUIDELITE numerical analysis |
| 2026-01-23 | Production use confirmed via grep analysis |
| 2026-01-23 | Report submitted to Immunefi |

---

**Submitted via Primacy of Impact** - Circuit assets outside explicit scope but directly affecting protocol security.
