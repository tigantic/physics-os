# FLUIDELITE ZK Circuit Analysis Report

**Generated**: 2026-01-23T14:49:47.337406
**Circuits Analyzed**: 1

## Summary

| Circuit | Signals | Constraints | Critical | High | Medium |
|---------|---------|-------------|----------|------|--------|
| email-verifier.circom | 903 | 0 | 0 | 577 | 0 |

## Detailed Findings

### email-verifier.circom

## HIGH: Rank-Deficient Constraint System

### Description
Constraint matrix has rank 0 but expected at least 576. This means 576 signals are under-constrained.

### Affected Signals
sha[0], sha[1], sha[2], sha[3], sha[4], sha[5], sha[6], sha[7], sha[8], sha[9], sha[10], sha[11], sha[12], sha[13], sha[14], sha[15], sha[16], sha[17], sha[18], sha[19], sha[20], sha[21], sha[22], sha[23], sha[24], sha[25], sha[26], sha[27], sha[28], sha[29], sha[30], sha[31], sha[32], sha[33], sha[34], sha[35], sha[36], sha[37], sha[38], sha[39], sha[40], sha[41], sha[42], sha[43], sha[44], sha[45], sha[46], sha[47], sha[48], sha[49], sha[50], sha[51], sha[52], sha[53], sha[54], sha[55], sha[56], sha[57], sha[58], sha[59], sha[60], sha[61], sha[62], sha[63], sha[64], sha[65], sha[66], sha[67], sha[68], sha[69], sha[70], sha[71], sha[72], sha[73], sha[74], sha[75], sha[76], sha[77], sha[78], sha[79], sha[80], sha[81], sha[82], sha[83], sha[84], sha[85], sha[86], sha[87], sha[88], sha[89], sha[90], sha[91], sha[92], sha[93], sha[94], sha[95], sha[96], sha[97], sha[98], sha[99], sha[100], sha[101], sha[102], sha[103], sha[104], sha[105], sha[106], sha[107], sha[108], sha[109], sha[110], sha[111], sha[112], sha[113], sha[114], sha[115], sha[116], sha[117], sha[118], sha[119], sha[120], sha[121], sha[122], sha[123], sha[124], sha[125], sha[126], sha[127], sha[128], sha[129], sha[130], sha[131], sha[132], sha[133], sha[134], sha[135], sha[136], sha[137], sha[138], sha[139], sha[140], sha[141], sha[142], sha[143], sha[144], sha[145], sha[146], sha[147], sha[148], sha[149], sha[150], sha[151], sha[152], sha[153], sha[154], sha[155], sha[156], sha[157], sha[158], sha[159], sha[160], sha[161], sha[162], sha[163], sha[164], sha[165], sha[166], sha[167], sha[168], sha[169], sha[170], sha[171], sha[172], sha[173], sha[174], sha[175], sha[176], sha[177], sha[178], sha[179], sha[180], sha[181], sha[182], sha[183], sha[184], sha[185], sha[186], sha[187], sha[188], sha[189], sha[190], sha[191], sha[192], sha[193], sha[194], sha[195], sha[196], sha[197], sha[198], sha[199], sha[200], sha[201], sha[202], sha[203], sha[204], sha[205], sha[206], sha[207], sha[208], sha[209], sha[210], sha[211], sha[212], sha[213], sha[214], sha[215], sha[216], sha[217], sha[218], sha[219], sha[220], sha[221], sha[222], sha[223], sha[224], sha[225], sha[226], sha[227], sha[228], sha[229], sha[230], sha[231], sha[232], sha[233], sha[234], sha[235], sha[236], sha[237], sha[238], sha[239], sha[240], sha[241], sha[242], sha[243], sha[244], sha[245], sha[246], sha[247], sha[248], sha[249], sha[250], sha[251], sha[252], sha[253], sha[254], sha[255], bhBase64[0], bhBase64[1], bhBase64[2], bhBase64[3], bhBase64[4], bhBase64[5], bhBase64[6], bhBase64[7], bhBase64[8], bhBase64[9], bhBase64[10], bhBase64[11], bhBase64[12], bhBase64[13], bhBase64[14], bhBase64[15], bhBase64[16], bhBase64[17], bhBase64[18], bhBase64[19], bhBase64[20], bhBase64[21], bhBase64[22], bhBase64[23], bhBase64[24], bhBase64[25], bhBase64[26], bhBase64[27], bhBase64[28], bhBase64[29], bhBase64[30], bhBase64[31], headerBodyHash[0], headerBodyHash[1], headerBodyHash[2], headerBodyHash[3], headerBodyHash[4], headerBodyHash[5], headerBodyHash[6], headerBodyHash[7], headerBodyHash[8], headerBodyHash[9], headerBodyHash[10], headerBodyHash[11], headerBodyHash[12], headerBodyHash[13], headerBodyHash[14], headerBodyHash[15], headerBodyHash[16], headerBodyHash[17], headerBodyHash[18], headerBodyHash[19], headerBodyHash[20], headerBodyHash[21], headerBodyHash[22], headerBodyHash[23], headerBodyHash[24], headerBodyHash[25], headerBodyHash[26], headerBodyHash[27], headerBodyHash[28], headerBodyHash[29], headerBodyHash[30], headerBodyHash[31], computedBodyHash[0], computedBodyHash[1], computedBodyHash[2], computedBodyHash[3], computedBodyHash[4], computedBodyHash[5], computedBodyHash[6], computedBodyHash[7], computedBodyHash[8], computedBodyHash[9], computedBodyHash[10], computedBodyHash[11], computedBodyHash[12], computedBodyHash[13], computedBodyHash[14], computedBodyHash[15], computedBodyHash[16], computedBodyHash[17], computedBodyHash[18], computedBodyHash[19], computedBodyHash[20], computedBodyHash[21], computedBodyHash[22], computedBodyHash[23], computedBodyHash[24], computedBodyHash[25], computedBodyHash[26], computedBodyHash[27], computedBodyHash[28], computedBodyHash[29], computedBodyHash[30], computedBodyHash[31], computedBodyHash[32], computedBodyHash[33], computedBodyHash[34], computedBodyHash[35], computedBodyHash[36], computedBodyHash[37], computedBodyHash[38], computedBodyHash[39], computedBodyHash[40], computedBodyHash[41], computedBodyHash[42], computedBodyHash[43], computedBodyHash[44], computedBodyHash[45], computedBodyHash[46], computedBodyHash[47], computedBodyHash[48], computedBodyHash[49], computedBodyHash[50], computedBodyHash[51], computedBodyHash[52], computedBodyHash[53], computedBodyHash[54], computedBodyHash[55], computedBodyHash[56], computedBodyHash[57], computedBodyHash[58], computedBodyHash[59], computedBodyHash[60], computedBodyHash[61], computedBodyHash[62], computedBodyHash[63], computedBodyHash[64], computedBodyHash[65], computedBodyHash[66], computedBodyHash[67], computedBodyHash[68], computedBodyHash[69], computedBodyHash[70], computedBodyHash[71], computedBodyHash[72], computedBodyHash[73], computedBodyHash[74], computedBodyHash[75], computedBodyHash[76], computedBodyHash[77], computedBodyHash[78], computedBodyHash[79], computedBodyHash[80], computedBodyHash[81], computedBodyHash[82], computedBodyHash[83], computedBodyHash[84], computedBodyHash[85], computedBodyHash[86], computedBodyHash[87], computedBodyHash[88], computedBodyHash[89], computedBodyHash[90], computedBodyHash[91], computedBodyHash[92], computedBodyHash[93], computedBodyHash[94], computedBodyHash[95], computedBodyHash[96], computedBodyHash[97], computedBodyHash[98], computedBodyHash[99], computedBodyHash[100], computedBodyHash[101], computedBodyHash[102], computedBodyHash[103], computedBodyHash[104], computedBodyHash[105], computedBodyHash[106], computedBodyHash[107], computedBodyHash[108], computedBodyHash[109], computedBodyHash[110], computedBodyHash[111], computedBodyHash[112], computedBodyHash[113], computedBodyHash[114], computedBodyHash[115], computedBodyHash[116], computedBodyHash[117], computedBodyHash[118], computedBodyHash[119], computedBodyHash[120], computedBodyHash[121], computedBodyHash[122], computedBodyHash[123], computedBodyHash[124], computedBodyHash[125], computedBodyHash[126], computedBodyHash[127], computedBodyHash[128], computedBodyHash[129], computedBodyHash[130], computedBodyHash[131], computedBodyHash[132], computedBodyHash[133], computedBodyHash[134], computedBodyHash[135], computedBodyHash[136], computedBodyHash[137], computedBodyHash[138], computedBodyHash[139], computedBodyHash[140], computedBodyHash[141], computedBodyHash[142], computedBodyHash[143], computedBodyHash[144], computedBodyHash[145], computedBodyHash[146], computedBodyHash[147], computedBodyHash[148], computedBodyHash[149], computedBodyHash[150], computedBodyHash[151], computedBodyHash[152], computedBodyHash[153], computedBodyHash[154], computedBodyHash[155], computedBodyHash[156], computedBodyHash[157], computedBodyHash[158], computedBodyHash[159], computedBodyHash[160], computedBodyHash[161], computedBodyHash[162], computedBodyHash[163], computedBodyHash[164], computedBodyHash[165], computedBodyHash[166], computedBodyHash[167], computedBodyHash[168], computedBodyHash[169], computedBodyHash[170], computedBodyHash[171], computedBodyHash[172], computedBodyHash[173], computedBodyHash[174], computedBodyHash[175], computedBodyHash[176], computedBodyHash[177], computedBodyHash[178], computedBodyHash[179], computedBodyHash[180], computedBodyHash[181], computedBodyHash[182], computedBodyHash[183], computedBodyHash[184], computedBodyHash[185], computedBodyHash[186], computedBodyHash[187], computedBodyHash[188], computedBodyHash[189], computedBodyHash[190], computedBodyHash[191], computedBodyHash[192], computedBodyHash[193], computedBodyHash[194], computedBodyHash[195], computedBodyHash[196], computedBodyHash[197], computedBodyHash[198], computedBodyHash[199], computedBodyHash[200], computedBodyHash[201], computedBodyHash[202], computedBodyHash[203], computedBodyHash[204], computedBodyHash[205], computedBodyHash[206], computedBodyHash[207], computedBodyHash[208], computedBodyHash[209], computedBodyHash[210], computedBodyHash[211], computedBodyHash[212], computedBodyHash[213], computedBodyHash[214], computedBodyHash[215], computedBodyHash[216], computedBodyHash[217], computedBodyHash[218], computedBodyHash[219], computedBodyHash[220], computedBodyHash[221], computedBodyHash[222], computedBodyHash[223], computedBodyHash[224], computedBodyHash[225], computedBodyHash[226], computedBodyHash[227], computedBodyHash[228], computedBodyHash[229], computedBodyHash[230], computedBodyHash[231], computedBodyHash[232], computedBodyHash[233], computedBodyHash[234], computedBodyHash[235], computedBodyHash[236], computedBodyHash[237], computedBodyHash[238], computedBodyHash[239], computedBodyHash[240], computedBodyHash[241], computedBodyHash[242], computedBodyHash[243], computedBodyHash[244], computedBodyHash[245], computedBodyHash[246], computedBodyHash[247], computedBodyHash[248], computedBodyHash[249], computedBodyHash[250], computedBodyHash[251], computedBodyHash[252], computedBodyHash[253], computedBodyHash[254], computedBodyHash[255]

### Affected Constraints
N/A

### Impact
Under-constrained signals allow multiple valid witnesses for the same public inputs, breaking soundness.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add constraints to fully determine all private signals.


## HIGH: Unconstrained Signal: sha[0]

### Description
Signal `sha[0]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[0]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[1]

### Description
Signal `sha[1]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[1]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[2]

### Description
Signal `sha[2]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[2]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[3]

### Description
Signal `sha[3]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[3]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[4]

### Description
Signal `sha[4]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[4]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[5]

### Description
Signal `sha[5]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[5]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[6]

### Description
Signal `sha[6]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[6]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[7]

### Description
Signal `sha[7]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[7]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[8]

### Description
Signal `sha[8]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[8]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[9]

### Description
Signal `sha[9]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[9]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[10]

### Description
Signal `sha[10]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[10]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[11]

### Description
Signal `sha[11]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[11]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[12]

### Description
Signal `sha[12]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[12]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[13]

### Description
Signal `sha[13]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[13]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[14]

### Description
Signal `sha[14]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[14]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[15]

### Description
Signal `sha[15]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[15]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[16]

### Description
Signal `sha[16]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[16]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[17]

### Description
Signal `sha[17]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[17]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[18]

### Description
Signal `sha[18]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[18]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[19]

### Description
Signal `sha[19]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[19]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[20]

### Description
Signal `sha[20]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[20]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[21]

### Description
Signal `sha[21]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[21]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[22]

### Description
Signal `sha[22]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[22]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[23]

### Description
Signal `sha[23]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[23]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[24]

### Description
Signal `sha[24]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[24]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[25]

### Description
Signal `sha[25]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[25]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[26]

### Description
Signal `sha[26]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[26]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[27]

### Description
Signal `sha[27]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[27]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[28]

### Description
Signal `sha[28]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[28]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[29]

### Description
Signal `sha[29]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[29]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[30]

### Description
Signal `sha[30]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[30]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[31]

### Description
Signal `sha[31]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[31]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[32]

### Description
Signal `sha[32]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[32]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[33]

### Description
Signal `sha[33]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[33]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[34]

### Description
Signal `sha[34]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[34]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[35]

### Description
Signal `sha[35]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[35]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[36]

### Description
Signal `sha[36]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[36]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[37]

### Description
Signal `sha[37]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[37]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[38]

### Description
Signal `sha[38]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[38]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[39]

### Description
Signal `sha[39]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[39]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[40]

### Description
Signal `sha[40]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[40]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[41]

### Description
Signal `sha[41]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[41]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[42]

### Description
Signal `sha[42]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[42]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[43]

### Description
Signal `sha[43]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[43]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[44]

### Description
Signal `sha[44]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[44]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[45]

### Description
Signal `sha[45]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[45]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[46]

### Description
Signal `sha[46]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[46]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[47]

### Description
Signal `sha[47]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[47]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[48]

### Description
Signal `sha[48]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[48]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[49]

### Description
Signal `sha[49]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[49]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[50]

### Description
Signal `sha[50]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[50]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[51]

### Description
Signal `sha[51]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[51]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[52]

### Description
Signal `sha[52]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[52]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[53]

### Description
Signal `sha[53]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[53]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[54]

### Description
Signal `sha[54]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[54]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[55]

### Description
Signal `sha[55]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[55]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[56]

### Description
Signal `sha[56]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[56]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[57]

### Description
Signal `sha[57]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[57]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[58]

### Description
Signal `sha[58]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[58]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[59]

### Description
Signal `sha[59]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[59]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[60]

### Description
Signal `sha[60]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[60]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[61]

### Description
Signal `sha[61]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[61]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[62]

### Description
Signal `sha[62]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[62]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[63]

### Description
Signal `sha[63]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[63]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[64]

### Description
Signal `sha[64]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[64]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[65]

### Description
Signal `sha[65]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[65]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[66]

### Description
Signal `sha[66]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[66]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[67]

### Description
Signal `sha[67]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[67]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[68]

### Description
Signal `sha[68]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[68]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[69]

### Description
Signal `sha[69]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[69]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[70]

### Description
Signal `sha[70]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[70]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[71]

### Description
Signal `sha[71]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[71]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[72]

### Description
Signal `sha[72]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[72]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[73]

### Description
Signal `sha[73]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[73]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[74]

### Description
Signal `sha[74]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[74]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[75]

### Description
Signal `sha[75]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[75]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[76]

### Description
Signal `sha[76]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[76]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[77]

### Description
Signal `sha[77]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[77]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[78]

### Description
Signal `sha[78]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[78]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[79]

### Description
Signal `sha[79]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[79]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[80]

### Description
Signal `sha[80]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[80]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[81]

### Description
Signal `sha[81]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[81]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[82]

### Description
Signal `sha[82]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[82]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[83]

### Description
Signal `sha[83]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[83]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[84]

### Description
Signal `sha[84]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[84]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[85]

### Description
Signal `sha[85]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[85]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[86]

### Description
Signal `sha[86]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[86]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[87]

### Description
Signal `sha[87]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[87]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[88]

### Description
Signal `sha[88]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[88]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[89]

### Description
Signal `sha[89]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[89]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[90]

### Description
Signal `sha[90]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[90]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[91]

### Description
Signal `sha[91]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[91]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[92]

### Description
Signal `sha[92]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[92]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[93]

### Description
Signal `sha[93]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[93]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[94]

### Description
Signal `sha[94]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[94]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[95]

### Description
Signal `sha[95]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[95]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[96]

### Description
Signal `sha[96]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[96]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[97]

### Description
Signal `sha[97]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[97]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[98]

### Description
Signal `sha[98]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[98]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[99]

### Description
Signal `sha[99]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[99]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[100]

### Description
Signal `sha[100]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[100]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[101]

### Description
Signal `sha[101]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[101]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[102]

### Description
Signal `sha[102]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[102]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[103]

### Description
Signal `sha[103]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[103]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[104]

### Description
Signal `sha[104]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[104]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[105]

### Description
Signal `sha[105]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[105]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[106]

### Description
Signal `sha[106]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[106]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[107]

### Description
Signal `sha[107]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[107]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[108]

### Description
Signal `sha[108]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[108]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[109]

### Description
Signal `sha[109]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[109]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[110]

### Description
Signal `sha[110]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[110]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[111]

### Description
Signal `sha[111]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[111]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[112]

### Description
Signal `sha[112]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[112]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[113]

### Description
Signal `sha[113]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[113]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[114]

### Description
Signal `sha[114]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[114]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[115]

### Description
Signal `sha[115]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[115]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[116]

### Description
Signal `sha[116]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[116]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[117]

### Description
Signal `sha[117]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[117]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[118]

### Description
Signal `sha[118]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[118]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[119]

### Description
Signal `sha[119]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[119]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[120]

### Description
Signal `sha[120]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[120]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[121]

### Description
Signal `sha[121]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[121]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[122]

### Description
Signal `sha[122]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[122]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[123]

### Description
Signal `sha[123]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[123]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[124]

### Description
Signal `sha[124]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[124]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[125]

### Description
Signal `sha[125]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[125]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[126]

### Description
Signal `sha[126]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[126]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[127]

### Description
Signal `sha[127]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[127]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[128]

### Description
Signal `sha[128]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[128]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[129]

### Description
Signal `sha[129]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[129]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[130]

### Description
Signal `sha[130]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[130]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[131]

### Description
Signal `sha[131]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[131]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[132]

### Description
Signal `sha[132]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[132]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[133]

### Description
Signal `sha[133]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[133]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[134]

### Description
Signal `sha[134]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[134]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[135]

### Description
Signal `sha[135]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[135]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[136]

### Description
Signal `sha[136]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[136]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[137]

### Description
Signal `sha[137]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[137]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[138]

### Description
Signal `sha[138]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[138]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[139]

### Description
Signal `sha[139]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[139]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[140]

### Description
Signal `sha[140]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[140]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[141]

### Description
Signal `sha[141]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[141]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[142]

### Description
Signal `sha[142]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[142]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[143]

### Description
Signal `sha[143]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[143]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[144]

### Description
Signal `sha[144]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[144]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[145]

### Description
Signal `sha[145]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[145]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[146]

### Description
Signal `sha[146]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[146]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[147]

### Description
Signal `sha[147]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[147]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[148]

### Description
Signal `sha[148]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[148]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[149]

### Description
Signal `sha[149]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[149]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[150]

### Description
Signal `sha[150]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[150]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[151]

### Description
Signal `sha[151]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[151]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[152]

### Description
Signal `sha[152]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[152]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[153]

### Description
Signal `sha[153]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[153]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[154]

### Description
Signal `sha[154]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[154]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[155]

### Description
Signal `sha[155]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[155]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[156]

### Description
Signal `sha[156]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[156]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[157]

### Description
Signal `sha[157]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[157]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[158]

### Description
Signal `sha[158]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[158]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[159]

### Description
Signal `sha[159]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[159]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[160]

### Description
Signal `sha[160]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[160]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[161]

### Description
Signal `sha[161]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[161]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[162]

### Description
Signal `sha[162]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[162]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[163]

### Description
Signal `sha[163]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[163]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[164]

### Description
Signal `sha[164]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[164]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[165]

### Description
Signal `sha[165]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[165]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[166]

### Description
Signal `sha[166]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[166]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[167]

### Description
Signal `sha[167]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[167]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[168]

### Description
Signal `sha[168]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[168]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[169]

### Description
Signal `sha[169]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[169]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[170]

### Description
Signal `sha[170]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[170]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[171]

### Description
Signal `sha[171]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[171]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[172]

### Description
Signal `sha[172]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[172]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[173]

### Description
Signal `sha[173]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[173]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[174]

### Description
Signal `sha[174]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[174]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[175]

### Description
Signal `sha[175]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[175]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[176]

### Description
Signal `sha[176]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[176]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[177]

### Description
Signal `sha[177]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[177]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[178]

### Description
Signal `sha[178]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[178]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[179]

### Description
Signal `sha[179]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[179]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[180]

### Description
Signal `sha[180]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[180]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[181]

### Description
Signal `sha[181]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[181]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[182]

### Description
Signal `sha[182]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[182]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[183]

### Description
Signal `sha[183]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[183]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[184]

### Description
Signal `sha[184]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[184]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[185]

### Description
Signal `sha[185]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[185]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[186]

### Description
Signal `sha[186]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[186]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[187]

### Description
Signal `sha[187]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[187]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[188]

### Description
Signal `sha[188]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[188]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[189]

### Description
Signal `sha[189]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[189]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[190]

### Description
Signal `sha[190]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[190]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[191]

### Description
Signal `sha[191]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[191]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[192]

### Description
Signal `sha[192]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[192]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[193]

### Description
Signal `sha[193]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[193]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[194]

### Description
Signal `sha[194]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[194]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[195]

### Description
Signal `sha[195]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[195]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[196]

### Description
Signal `sha[196]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[196]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[197]

### Description
Signal `sha[197]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[197]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[198]

### Description
Signal `sha[198]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[198]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[199]

### Description
Signal `sha[199]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[199]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[200]

### Description
Signal `sha[200]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[200]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[201]

### Description
Signal `sha[201]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[201]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[202]

### Description
Signal `sha[202]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[202]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[203]

### Description
Signal `sha[203]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[203]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[204]

### Description
Signal `sha[204]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[204]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[205]

### Description
Signal `sha[205]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[205]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[206]

### Description
Signal `sha[206]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[206]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[207]

### Description
Signal `sha[207]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[207]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[208]

### Description
Signal `sha[208]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[208]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[209]

### Description
Signal `sha[209]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[209]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[210]

### Description
Signal `sha[210]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[210]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[211]

### Description
Signal `sha[211]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[211]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[212]

### Description
Signal `sha[212]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[212]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[213]

### Description
Signal `sha[213]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[213]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[214]

### Description
Signal `sha[214]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[214]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[215]

### Description
Signal `sha[215]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[215]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[216]

### Description
Signal `sha[216]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[216]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[217]

### Description
Signal `sha[217]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[217]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[218]

### Description
Signal `sha[218]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[218]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[219]

### Description
Signal `sha[219]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[219]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[220]

### Description
Signal `sha[220]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[220]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[221]

### Description
Signal `sha[221]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[221]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[222]

### Description
Signal `sha[222]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[222]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[223]

### Description
Signal `sha[223]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[223]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[224]

### Description
Signal `sha[224]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[224]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[225]

### Description
Signal `sha[225]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[225]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[226]

### Description
Signal `sha[226]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[226]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[227]

### Description
Signal `sha[227]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[227]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[228]

### Description
Signal `sha[228]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[228]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[229]

### Description
Signal `sha[229]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[229]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[230]

### Description
Signal `sha[230]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[230]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[231]

### Description
Signal `sha[231]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[231]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[232]

### Description
Signal `sha[232]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[232]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[233]

### Description
Signal `sha[233]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[233]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[234]

### Description
Signal `sha[234]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[234]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[235]

### Description
Signal `sha[235]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[235]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[236]

### Description
Signal `sha[236]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[236]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[237]

### Description
Signal `sha[237]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[237]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[238]

### Description
Signal `sha[238]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[238]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[239]

### Description
Signal `sha[239]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[239]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[240]

### Description
Signal `sha[240]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[240]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[241]

### Description
Signal `sha[241]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[241]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[242]

### Description
Signal `sha[242]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[242]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[243]

### Description
Signal `sha[243]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[243]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[244]

### Description
Signal `sha[244]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[244]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[245]

### Description
Signal `sha[245]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[245]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[246]

### Description
Signal `sha[246]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[246]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[247]

### Description
Signal `sha[247]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[247]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[248]

### Description
Signal `sha[248]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[248]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[249]

### Description
Signal `sha[249]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[249]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[250]

### Description
Signal `sha[250]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[250]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[251]

### Description
Signal `sha[251]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[251]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[252]

### Description
Signal `sha[252]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[252]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[253]

### Description
Signal `sha[253]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[253]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[254]

### Description
Signal `sha[254]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[254]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: sha[255]

### Description
Signal `sha[255]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
sha[255]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[0]

### Description
Signal `bhBase64[0]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[0]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[1]

### Description
Signal `bhBase64[1]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[1]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[2]

### Description
Signal `bhBase64[2]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[2]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[3]

### Description
Signal `bhBase64[3]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[3]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[4]

### Description
Signal `bhBase64[4]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[4]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[5]

### Description
Signal `bhBase64[5]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[5]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[6]

### Description
Signal `bhBase64[6]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[6]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[7]

### Description
Signal `bhBase64[7]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[7]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[8]

### Description
Signal `bhBase64[8]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[8]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[9]

### Description
Signal `bhBase64[9]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[9]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[10]

### Description
Signal `bhBase64[10]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[10]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[11]

### Description
Signal `bhBase64[11]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[11]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[12]

### Description
Signal `bhBase64[12]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[12]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[13]

### Description
Signal `bhBase64[13]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[13]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[14]

### Description
Signal `bhBase64[14]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[14]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[15]

### Description
Signal `bhBase64[15]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[15]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[16]

### Description
Signal `bhBase64[16]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[16]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[17]

### Description
Signal `bhBase64[17]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[17]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[18]

### Description
Signal `bhBase64[18]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[18]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[19]

### Description
Signal `bhBase64[19]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[19]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[20]

### Description
Signal `bhBase64[20]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[20]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[21]

### Description
Signal `bhBase64[21]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[21]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[22]

### Description
Signal `bhBase64[22]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[22]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[23]

### Description
Signal `bhBase64[23]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[23]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[24]

### Description
Signal `bhBase64[24]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[24]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[25]

### Description
Signal `bhBase64[25]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[25]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[26]

### Description
Signal `bhBase64[26]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[26]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[27]

### Description
Signal `bhBase64[27]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[27]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[28]

### Description
Signal `bhBase64[28]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[28]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[29]

### Description
Signal `bhBase64[29]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[29]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[30]

### Description
Signal `bhBase64[30]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[30]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: bhBase64[31]

### Description
Signal `bhBase64[31]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
bhBase64[31]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[0]

### Description
Signal `headerBodyHash[0]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[0]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[1]

### Description
Signal `headerBodyHash[1]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[1]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[2]

### Description
Signal `headerBodyHash[2]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[2]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[3]

### Description
Signal `headerBodyHash[3]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[3]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[4]

### Description
Signal `headerBodyHash[4]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[4]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[5]

### Description
Signal `headerBodyHash[5]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[5]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[6]

### Description
Signal `headerBodyHash[6]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[6]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[7]

### Description
Signal `headerBodyHash[7]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[7]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[8]

### Description
Signal `headerBodyHash[8]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[8]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[9]

### Description
Signal `headerBodyHash[9]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[9]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[10]

### Description
Signal `headerBodyHash[10]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[10]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[11]

### Description
Signal `headerBodyHash[11]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[11]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[12]

### Description
Signal `headerBodyHash[12]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[12]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[13]

### Description
Signal `headerBodyHash[13]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[13]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[14]

### Description
Signal `headerBodyHash[14]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[14]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[15]

### Description
Signal `headerBodyHash[15]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[15]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[16]

### Description
Signal `headerBodyHash[16]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[16]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[17]

### Description
Signal `headerBodyHash[17]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[17]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[18]

### Description
Signal `headerBodyHash[18]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[18]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[19]

### Description
Signal `headerBodyHash[19]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[19]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[20]

### Description
Signal `headerBodyHash[20]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[20]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[21]

### Description
Signal `headerBodyHash[21]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[21]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[22]

### Description
Signal `headerBodyHash[22]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[22]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[23]

### Description
Signal `headerBodyHash[23]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[23]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[24]

### Description
Signal `headerBodyHash[24]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[24]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[25]

### Description
Signal `headerBodyHash[25]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[25]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[26]

### Description
Signal `headerBodyHash[26]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[26]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[27]

### Description
Signal `headerBodyHash[27]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[27]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[28]

### Description
Signal `headerBodyHash[28]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[28]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[29]

### Description
Signal `headerBodyHash[29]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[29]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[30]

### Description
Signal `headerBodyHash[30]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[30]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: headerBodyHash[31]

### Description
Signal `headerBodyHash[31]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
headerBodyHash[31]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[0]

### Description
Signal `computedBodyHash[0]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[0]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[1]

### Description
Signal `computedBodyHash[1]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[1]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[2]

### Description
Signal `computedBodyHash[2]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[2]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[3]

### Description
Signal `computedBodyHash[3]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[3]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[4]

### Description
Signal `computedBodyHash[4]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[4]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[5]

### Description
Signal `computedBodyHash[5]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[5]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[6]

### Description
Signal `computedBodyHash[6]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[6]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[7]

### Description
Signal `computedBodyHash[7]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[7]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[8]

### Description
Signal `computedBodyHash[8]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[8]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[9]

### Description
Signal `computedBodyHash[9]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[9]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[10]

### Description
Signal `computedBodyHash[10]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[10]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[11]

### Description
Signal `computedBodyHash[11]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[11]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[12]

### Description
Signal `computedBodyHash[12]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[12]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[13]

### Description
Signal `computedBodyHash[13]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[13]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[14]

### Description
Signal `computedBodyHash[14]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[14]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[15]

### Description
Signal `computedBodyHash[15]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[15]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[16]

### Description
Signal `computedBodyHash[16]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[16]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[17]

### Description
Signal `computedBodyHash[17]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[17]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[18]

### Description
Signal `computedBodyHash[18]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[18]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[19]

### Description
Signal `computedBodyHash[19]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[19]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[20]

### Description
Signal `computedBodyHash[20]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[20]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[21]

### Description
Signal `computedBodyHash[21]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[21]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[22]

### Description
Signal `computedBodyHash[22]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[22]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[23]

### Description
Signal `computedBodyHash[23]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[23]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[24]

### Description
Signal `computedBodyHash[24]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[24]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[25]

### Description
Signal `computedBodyHash[25]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[25]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[26]

### Description
Signal `computedBodyHash[26]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[26]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[27]

### Description
Signal `computedBodyHash[27]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[27]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[28]

### Description
Signal `computedBodyHash[28]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[28]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[29]

### Description
Signal `computedBodyHash[29]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[29]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[30]

### Description
Signal `computedBodyHash[30]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[30]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[31]

### Description
Signal `computedBodyHash[31]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[31]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[32]

### Description
Signal `computedBodyHash[32]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[32]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[33]

### Description
Signal `computedBodyHash[33]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[33]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[34]

### Description
Signal `computedBodyHash[34]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[34]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[35]

### Description
Signal `computedBodyHash[35]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[35]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[36]

### Description
Signal `computedBodyHash[36]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[36]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[37]

### Description
Signal `computedBodyHash[37]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[37]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[38]

### Description
Signal `computedBodyHash[38]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[38]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[39]

### Description
Signal `computedBodyHash[39]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[39]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[40]

### Description
Signal `computedBodyHash[40]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[40]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[41]

### Description
Signal `computedBodyHash[41]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[41]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[42]

### Description
Signal `computedBodyHash[42]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[42]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[43]

### Description
Signal `computedBodyHash[43]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[43]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[44]

### Description
Signal `computedBodyHash[44]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[44]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[45]

### Description
Signal `computedBodyHash[45]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[45]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[46]

### Description
Signal `computedBodyHash[46]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[46]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[47]

### Description
Signal `computedBodyHash[47]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[47]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[48]

### Description
Signal `computedBodyHash[48]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[48]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[49]

### Description
Signal `computedBodyHash[49]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[49]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[50]

### Description
Signal `computedBodyHash[50]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[50]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[51]

### Description
Signal `computedBodyHash[51]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[51]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[52]

### Description
Signal `computedBodyHash[52]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[52]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[53]

### Description
Signal `computedBodyHash[53]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[53]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[54]

### Description
Signal `computedBodyHash[54]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[54]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[55]

### Description
Signal `computedBodyHash[55]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[55]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[56]

### Description
Signal `computedBodyHash[56]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[56]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[57]

### Description
Signal `computedBodyHash[57]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[57]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[58]

### Description
Signal `computedBodyHash[58]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[58]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[59]

### Description
Signal `computedBodyHash[59]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[59]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[60]

### Description
Signal `computedBodyHash[60]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[60]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[61]

### Description
Signal `computedBodyHash[61]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[61]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[62]

### Description
Signal `computedBodyHash[62]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[62]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[63]

### Description
Signal `computedBodyHash[63]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[63]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[64]

### Description
Signal `computedBodyHash[64]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[64]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[65]

### Description
Signal `computedBodyHash[65]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[65]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[66]

### Description
Signal `computedBodyHash[66]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[66]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[67]

### Description
Signal `computedBodyHash[67]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[67]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[68]

### Description
Signal `computedBodyHash[68]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[68]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[69]

### Description
Signal `computedBodyHash[69]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[69]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[70]

### Description
Signal `computedBodyHash[70]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[70]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[71]

### Description
Signal `computedBodyHash[71]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[71]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[72]

### Description
Signal `computedBodyHash[72]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[72]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[73]

### Description
Signal `computedBodyHash[73]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[73]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[74]

### Description
Signal `computedBodyHash[74]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[74]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[75]

### Description
Signal `computedBodyHash[75]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[75]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[76]

### Description
Signal `computedBodyHash[76]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[76]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[77]

### Description
Signal `computedBodyHash[77]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[77]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[78]

### Description
Signal `computedBodyHash[78]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[78]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[79]

### Description
Signal `computedBodyHash[79]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[79]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[80]

### Description
Signal `computedBodyHash[80]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[80]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[81]

### Description
Signal `computedBodyHash[81]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[81]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[82]

### Description
Signal `computedBodyHash[82]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[82]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[83]

### Description
Signal `computedBodyHash[83]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[83]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[84]

### Description
Signal `computedBodyHash[84]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[84]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[85]

### Description
Signal `computedBodyHash[85]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[85]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[86]

### Description
Signal `computedBodyHash[86]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[86]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[87]

### Description
Signal `computedBodyHash[87]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[87]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[88]

### Description
Signal `computedBodyHash[88]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[88]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[89]

### Description
Signal `computedBodyHash[89]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[89]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[90]

### Description
Signal `computedBodyHash[90]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[90]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[91]

### Description
Signal `computedBodyHash[91]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[91]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[92]

### Description
Signal `computedBodyHash[92]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[92]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[93]

### Description
Signal `computedBodyHash[93]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[93]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[94]

### Description
Signal `computedBodyHash[94]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[94]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[95]

### Description
Signal `computedBodyHash[95]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[95]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[96]

### Description
Signal `computedBodyHash[96]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[96]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[97]

### Description
Signal `computedBodyHash[97]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[97]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[98]

### Description
Signal `computedBodyHash[98]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[98]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[99]

### Description
Signal `computedBodyHash[99]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[99]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[100]

### Description
Signal `computedBodyHash[100]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[100]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[101]

### Description
Signal `computedBodyHash[101]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[101]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[102]

### Description
Signal `computedBodyHash[102]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[102]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[103]

### Description
Signal `computedBodyHash[103]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[103]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[104]

### Description
Signal `computedBodyHash[104]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[104]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[105]

### Description
Signal `computedBodyHash[105]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[105]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[106]

### Description
Signal `computedBodyHash[106]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[106]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[107]

### Description
Signal `computedBodyHash[107]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[107]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[108]

### Description
Signal `computedBodyHash[108]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[108]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[109]

### Description
Signal `computedBodyHash[109]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[109]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[110]

### Description
Signal `computedBodyHash[110]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[110]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[111]

### Description
Signal `computedBodyHash[111]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[111]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[112]

### Description
Signal `computedBodyHash[112]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[112]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[113]

### Description
Signal `computedBodyHash[113]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[113]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[114]

### Description
Signal `computedBodyHash[114]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[114]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[115]

### Description
Signal `computedBodyHash[115]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[115]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[116]

### Description
Signal `computedBodyHash[116]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[116]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[117]

### Description
Signal `computedBodyHash[117]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[117]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[118]

### Description
Signal `computedBodyHash[118]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[118]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[119]

### Description
Signal `computedBodyHash[119]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[119]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[120]

### Description
Signal `computedBodyHash[120]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[120]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[121]

### Description
Signal `computedBodyHash[121]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[121]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[122]

### Description
Signal `computedBodyHash[122]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[122]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[123]

### Description
Signal `computedBodyHash[123]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[123]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[124]

### Description
Signal `computedBodyHash[124]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[124]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[125]

### Description
Signal `computedBodyHash[125]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[125]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[126]

### Description
Signal `computedBodyHash[126]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[126]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[127]

### Description
Signal `computedBodyHash[127]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[127]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[128]

### Description
Signal `computedBodyHash[128]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[128]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[129]

### Description
Signal `computedBodyHash[129]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[129]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[130]

### Description
Signal `computedBodyHash[130]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[130]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[131]

### Description
Signal `computedBodyHash[131]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[131]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[132]

### Description
Signal `computedBodyHash[132]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[132]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[133]

### Description
Signal `computedBodyHash[133]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[133]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[134]

### Description
Signal `computedBodyHash[134]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[134]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[135]

### Description
Signal `computedBodyHash[135]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[135]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[136]

### Description
Signal `computedBodyHash[136]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[136]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[137]

### Description
Signal `computedBodyHash[137]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[137]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[138]

### Description
Signal `computedBodyHash[138]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[138]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[139]

### Description
Signal `computedBodyHash[139]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[139]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[140]

### Description
Signal `computedBodyHash[140]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[140]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[141]

### Description
Signal `computedBodyHash[141]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[141]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[142]

### Description
Signal `computedBodyHash[142]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[142]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[143]

### Description
Signal `computedBodyHash[143]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[143]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[144]

### Description
Signal `computedBodyHash[144]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[144]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[145]

### Description
Signal `computedBodyHash[145]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[145]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[146]

### Description
Signal `computedBodyHash[146]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[146]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[147]

### Description
Signal `computedBodyHash[147]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[147]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[148]

### Description
Signal `computedBodyHash[148]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[148]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[149]

### Description
Signal `computedBodyHash[149]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[149]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[150]

### Description
Signal `computedBodyHash[150]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[150]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[151]

### Description
Signal `computedBodyHash[151]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[151]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[152]

### Description
Signal `computedBodyHash[152]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[152]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[153]

### Description
Signal `computedBodyHash[153]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[153]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[154]

### Description
Signal `computedBodyHash[154]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[154]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[155]

### Description
Signal `computedBodyHash[155]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[155]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[156]

### Description
Signal `computedBodyHash[156]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[156]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[157]

### Description
Signal `computedBodyHash[157]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[157]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[158]

### Description
Signal `computedBodyHash[158]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[158]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[159]

### Description
Signal `computedBodyHash[159]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[159]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[160]

### Description
Signal `computedBodyHash[160]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[160]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[161]

### Description
Signal `computedBodyHash[161]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[161]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[162]

### Description
Signal `computedBodyHash[162]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[162]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[163]

### Description
Signal `computedBodyHash[163]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[163]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[164]

### Description
Signal `computedBodyHash[164]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[164]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[165]

### Description
Signal `computedBodyHash[165]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[165]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[166]

### Description
Signal `computedBodyHash[166]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[166]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[167]

### Description
Signal `computedBodyHash[167]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[167]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[168]

### Description
Signal `computedBodyHash[168]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[168]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[169]

### Description
Signal `computedBodyHash[169]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[169]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[170]

### Description
Signal `computedBodyHash[170]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[170]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[171]

### Description
Signal `computedBodyHash[171]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[171]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[172]

### Description
Signal `computedBodyHash[172]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[172]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[173]

### Description
Signal `computedBodyHash[173]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[173]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[174]

### Description
Signal `computedBodyHash[174]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[174]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[175]

### Description
Signal `computedBodyHash[175]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[175]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[176]

### Description
Signal `computedBodyHash[176]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[176]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[177]

### Description
Signal `computedBodyHash[177]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[177]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[178]

### Description
Signal `computedBodyHash[178]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[178]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[179]

### Description
Signal `computedBodyHash[179]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[179]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[180]

### Description
Signal `computedBodyHash[180]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[180]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[181]

### Description
Signal `computedBodyHash[181]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[181]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[182]

### Description
Signal `computedBodyHash[182]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[182]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[183]

### Description
Signal `computedBodyHash[183]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[183]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[184]

### Description
Signal `computedBodyHash[184]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[184]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[185]

### Description
Signal `computedBodyHash[185]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[185]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[186]

### Description
Signal `computedBodyHash[186]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[186]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[187]

### Description
Signal `computedBodyHash[187]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[187]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[188]

### Description
Signal `computedBodyHash[188]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[188]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[189]

### Description
Signal `computedBodyHash[189]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[189]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[190]

### Description
Signal `computedBodyHash[190]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[190]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[191]

### Description
Signal `computedBodyHash[191]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[191]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[192]

### Description
Signal `computedBodyHash[192]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[192]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[193]

### Description
Signal `computedBodyHash[193]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[193]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[194]

### Description
Signal `computedBodyHash[194]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[194]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[195]

### Description
Signal `computedBodyHash[195]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[195]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[196]

### Description
Signal `computedBodyHash[196]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[196]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[197]

### Description
Signal `computedBodyHash[197]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[197]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[198]

### Description
Signal `computedBodyHash[198]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[198]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[199]

### Description
Signal `computedBodyHash[199]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[199]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[200]

### Description
Signal `computedBodyHash[200]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[200]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[201]

### Description
Signal `computedBodyHash[201]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[201]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[202]

### Description
Signal `computedBodyHash[202]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[202]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[203]

### Description
Signal `computedBodyHash[203]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[203]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[204]

### Description
Signal `computedBodyHash[204]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[204]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[205]

### Description
Signal `computedBodyHash[205]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[205]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[206]

### Description
Signal `computedBodyHash[206]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[206]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[207]

### Description
Signal `computedBodyHash[207]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[207]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[208]

### Description
Signal `computedBodyHash[208]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[208]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[209]

### Description
Signal `computedBodyHash[209]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[209]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[210]

### Description
Signal `computedBodyHash[210]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[210]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[211]

### Description
Signal `computedBodyHash[211]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[211]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[212]

### Description
Signal `computedBodyHash[212]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[212]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[213]

### Description
Signal `computedBodyHash[213]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[213]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[214]

### Description
Signal `computedBodyHash[214]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[214]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[215]

### Description
Signal `computedBodyHash[215]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[215]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[216]

### Description
Signal `computedBodyHash[216]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[216]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[217]

### Description
Signal `computedBodyHash[217]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[217]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[218]

### Description
Signal `computedBodyHash[218]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[218]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[219]

### Description
Signal `computedBodyHash[219]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[219]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[220]

### Description
Signal `computedBodyHash[220]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[220]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[221]

### Description
Signal `computedBodyHash[221]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[221]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[222]

### Description
Signal `computedBodyHash[222]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[222]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[223]

### Description
Signal `computedBodyHash[223]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[223]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[224]

### Description
Signal `computedBodyHash[224]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[224]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[225]

### Description
Signal `computedBodyHash[225]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[225]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[226]

### Description
Signal `computedBodyHash[226]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[226]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[227]

### Description
Signal `computedBodyHash[227]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[227]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[228]

### Description
Signal `computedBodyHash[228]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[228]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[229]

### Description
Signal `computedBodyHash[229]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[229]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[230]

### Description
Signal `computedBodyHash[230]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[230]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[231]

### Description
Signal `computedBodyHash[231]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[231]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[232]

### Description
Signal `computedBodyHash[232]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[232]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[233]

### Description
Signal `computedBodyHash[233]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[233]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[234]

### Description
Signal `computedBodyHash[234]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[234]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[235]

### Description
Signal `computedBodyHash[235]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[235]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[236]

### Description
Signal `computedBodyHash[236]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[236]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[237]

### Description
Signal `computedBodyHash[237]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[237]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[238]

### Description
Signal `computedBodyHash[238]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[238]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[239]

### Description
Signal `computedBodyHash[239]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[239]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[240]

### Description
Signal `computedBodyHash[240]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[240]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[241]

### Description
Signal `computedBodyHash[241]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[241]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[242]

### Description
Signal `computedBodyHash[242]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[242]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[243]

### Description
Signal `computedBodyHash[243]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[243]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[244]

### Description
Signal `computedBodyHash[244]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[244]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[245]

### Description
Signal `computedBodyHash[245]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[245]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[246]

### Description
Signal `computedBodyHash[246]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[246]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[247]

### Description
Signal `computedBodyHash[247]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[247]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[248]

### Description
Signal `computedBodyHash[248]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[248]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[249]

### Description
Signal `computedBodyHash[249]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[249]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[250]

### Description
Signal `computedBodyHash[250]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[250]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[251]

### Description
Signal `computedBodyHash[251]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[251]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[252]

### Description
Signal `computedBodyHash[252]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[252]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[253]

### Description
Signal `computedBodyHash[253]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[253]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[254]

### Description
Signal `computedBodyHash[254]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[254]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.


## HIGH: Unconstrained Signal: computedBodyHash[255]

### Description
Signal `computedBodyHash[255]` appears in nullspace and has no direct constraints. It can take any value.

### Affected Signals
computedBodyHash[255]

### Affected Constraints
N/A

### Impact
Attacker can set this signal to any value and still generate a valid proof.

### Proof of Concept
```
See attached witness files
```

### Recommendation
Add missing constraints to fully constrain the affected signals.

