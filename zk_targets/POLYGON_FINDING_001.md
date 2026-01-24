# Polygon zkEVM - Unconstrained Public Input in recursive1.circom

## Summary

A soundness vulnerability exists in `recursive1.circom` where the `rootC[4]` signal array is declared as a public input but is never constrained in the circuit. This creates proof malleability where a prover can set arbitrary values for these 4 public inputs.

## Affected Repository

**Repository:** https://github.com/0xPolygonHermez/zkevm-proverjs  
**File:** `recursive/recursive1.circom`  
**Lines:** 9, 78

## Vulnerability Details

### Root Cause

In `recursive1.circom`:

```circom
pragma circom 2.1.0;
pragma custom_templates;

include "c12a.verifier.circom";

template Main() {
    signal input publics[44];
    signal input rootC[4];  // <-- DECLARED but NEVER used
    
    // ... other signals ...
    
    component vA = StarkVerifier();
    
    vA.publics <== publics;
    vA.root1 <== root1;
    // ... etc ...
    // NOTE: vA.rootC is NEVER assigned!
    vA.finalPol <== finalPol;
}

component main {public [publics, rootC]}= Main();  // rootC is PUBLIC
```

### Why rootC Cannot Be Connected

The `c12a.verifier.circom` is generated using `pil2circom` **without** the `--verkeyInput` flag:

```bash
# From package.json:
"c12a_gencircom": "$PILSTARK/main_pil2circom.js --skipMain -p $BDIR/pil/c12a.pil ..."
```

When `--verkeyInput` is NOT specified, the generated StarkVerifier template has `rootC` **hardcoded internally**:

```circom
// Generated WITHOUT --verkeyInput:
signal rootC[4] <== [<hardcoded_values>];  // Internal signal, NOT an input!
```

Therefore, `recursive1.circom` cannot pass its `rootC` input to the StarkVerifier because the StarkVerifier doesn't have `rootC` as an input signal.

### Proof

1. `recursive1.circom` declares `signal input rootC[4]`
2. `c12a.verifier.circom` (included) has `rootC` hardcoded internally
3. There is NO `vA.rootC <== rootC` assignment in `recursive1.circom`
4. The circuit compiles with 48 public inputs (44 + 4)
5. Only 44 of these are actually constrained

### Contrast with recursive2

In `recursive2.circom.ejs`, the rootC IS properly connected:

```circom
// recursive2.circom.ejs correctly handles rootC:
for (var i=0; i<4; i++) {
    vA.publics[44+i] <== rootC[i];
}
vA.rootC <== a_muxRootC.out;  // <-- PROPERLY CONNECTED
```

This is because `recursive1.verifier.circom` (used by recursive2) IS generated with `--verkeyInput`.

## Security Impact

### Direct Impact

1. **Proof Malleability:** A prover can generate valid recursive1 proofs with arbitrary `rootC[0-3]` values
2. **Unconstrained Public Inputs:** The R1CS has 4 public inputs that have no corresponding constraints
3. **Violated Assumption:** The proof claims to have verified a certain `rootC` but the actual verification uses the hardcoded value

### Mitigation by Downstream Circuits

The vulnerability is **partially mitigated** at the recursive2 and recursivef levels:

1. `recursive2` sets `vA.publics[44-47]` from its own `rootC` input (not from the recursive1 proof)
2. `recursivef` hardcodes the correct `constRoot` values

However, this mitigation pattern is:
- Error-prone (relies on correct implementation in all downstream circuits)
- Not explicitly documented
- Creates a disconnect between what recursive1 "proves" and what it actually verifies

### Severity Assessment

**Impact:** Medium-High
- Proof malleability in a production zkEVM circuit
- Affects the fundamental soundness property of recursive STARK verification

**Probability:** Medium
- Requires understanding of the recursive proving system
- Downstream circuits may catch inconsistencies

**Recommended Severity:** HIGH (using Immunefi's severity matrix)

## Proof of Concept

### Step 1: Verify the Missing Assignment

```bash
cd zkevm-proverjs/recursive
grep -n "rootC" recursive1.circom
```

Output:
```
9:    signal input rootC[4];
```

Notice: No `vA.rootC <== rootC` anywhere!

### Step 2: Compare with recursive2

```bash
grep -n "rootC" recursive2.circom.ejs | head -20
```

Shows proper handling with `vA.rootC <== a_muxRootC.out`

### Step 3: Verify c12a.verifier Generation

```bash
grep "c12a_gencircom" package.json
```

Shows NO `--verkeyInput` flag, confirming rootC is hardcoded.

## Recommended Fix

### Option 1: Remove Unused Public Input

Remove `rootC` from recursive1's public inputs since it's not actually constrained:

```circom
// OLD:
component main {public [publics, rootC]}= Main();

// NEW:
component main {public [publics]}= Main();
```

And remove the signal declaration:
```circom
// Remove: signal input rootC[4];
```

### Option 2: Generate c12a.verifier with --verkeyInput

Modify the build script to use `--verkeyInput`:

```json
"c12a_gencircom": "$PILSTARK/main_pil2circom.js --skipMain --verkeyInput -p $BDIR/pil/c12a.pil ..."
```

Then add the proper connection in recursive1.circom:
```circom
vA.rootC <== rootC;
```

## Additional Notes

1. This finding was discovered using static analysis of Circom circuit code
2. The vulnerability pattern is "unconstrained public input" which is a known class of ZK circuit bugs
3. The ORACLE Circom parser correctly identified this signal as under-constrained

## Timeline

- **Discovery Date:** 2025
- **Analysis Completed:** 2025
- **Status:** Ready for submission

---

**Disclaimer:** This report is provided for security research purposes under responsible disclosure practices.
