# TIG-011a Synthesis Route

## Target Compound

| Property | Value |
|----------|-------|
| **Name** | 4-(4-methylpiperazin-1-yl)-7-methoxyquinazoline |
| **SMILES** | `COc1ccc2ncnc(N3CCN(C)CC3)c2c1` |
| **MW** | 258.32 g/mol |
| **Formula** | C₁₄H₁₈N₄O |

```
        OCH₃
          |
    ┌─────┴─────┐
    │           │
    │     N     │
    │    ╱ ╲    │
    │   N   N───┤
    │           │
    └─────┬─────┘
          │
          N
         ╱ ╲
        N   CH₃
         ╲ ╱
          N
          |
          H
```

---

## Retrosynthetic Analysis

```
TIG-011a
   │
   ▼ Disconnection 1: C-N bond (piperazine → quinazoline)
   │
┌──┴──────────────────────────────────────┐
│                                          │
▼                                          ▼
4-Chloro-7-methoxyquinazoline    +    N-methylpiperazine
(Electrophile)                        (Nucleophile)
   │
   ▼ Disconnection 2: Ring formation
   │
2-Amino-4-methoxybenzonitrile + Formamidine acetate
        (or via cyclization)
```

---

## Synthetic Route (3 Steps)

### Step 1: Prepare 4-Chloro-7-methoxyquinazoline

**Starting Materials:**
- 2-Amino-4-methoxybenzamide (commercial)
- POCl₃ (phosphorus oxychloride)

**Procedure:**
```
2-Amino-4-methoxybenzamide + HC(OEt)₃ → 7-Methoxyquinazolin-4(3H)-one
                                              │
                                              ▼ POCl₃, reflux
                                    4-Chloro-7-methoxyquinazoline
```

**Conditions:**
- Triethyl orthoformate, 120°C, 4h
- POCl₃, catalytic DMF, reflux 3h
- Yield: ~75%

---

### Step 2: Nucleophilic Aromatic Substitution

**Reaction:**
```
4-Chloro-7-methoxyquinazoline + N-methylpiperazine → TIG-011a + HCl
```

**Conditions:**
- Solvent: n-BuOH or DMSO
- Base: K₂CO₃ or DIPEA (2 eq)
- Temperature: 100-120°C
- Time: 6-12h
- Yield: 80-90%

**Workup:**
1. Cool to RT
2. Pour into ice water
3. Extract with EtOAc (3×)
4. Wash with brine
5. Dry over Na₂SO₄
6. Column chromatography (EtOAc/MeOH 95:5)

---

### Step 3: Salt Formation (Optional, for stability)

**Form HCl salt for crystalline solid:**
```
TIG-011a (free base) + HCl/Et₂O → TIG-011a·HCl
```

**Conditions:**
- Dissolve in minimal Et₂O
- Add 1M HCl in Et₂O
- Filter precipitate
- Wash with cold Et₂O
- Dry under vacuum

---

## Commercial Starting Materials

| Compound | CAS | Vendor | ~Price |
|----------|-----|--------|--------|
| 2-Amino-4-methoxybenzamide | 17481-23-5 | Sigma-Aldrich | $50/5g |
| N-Methylpiperazine | 109-01-3 | Sigma-Aldrich | $30/100mL |
| POCl₃ | 10025-87-3 | Fisher | $40/500mL |
| Triethyl orthoformate | 122-51-0 | Sigma-Aldrich | $25/500mL |

**Total estimated cost for 1g synthesis: ~$200**

---

## Alternative: One-Pot Route

For faster iteration, use commercial 4-chloro-7-methoxyquinazoline:

| Compound | CAS | Vendor | Price |
|----------|-----|--------|-------|
| 4-Chloro-7-methoxyquinazoline | 18592-14-2 | Enamine | $85/1g |

**Single step:**
```
4-Chloro-7-methoxyquinazoline + N-methylpiperazine 
    → K₂CO₃, n-BuOH, 110°C, 8h 
    → TIG-011a (85% yield)
```

---

## Characterization Targets

### ¹H NMR (DMSO-d₆, 400 MHz)

| δ (ppm) | Multiplicity | Integration | Assignment |
|---------|--------------|-------------|------------|
| 8.45 | s | 1H | H-2 (quinazoline) |
| 7.85 | d, J=9 Hz | 1H | H-5 |
| 7.15 | dd | 1H | H-6 |
| 7.05 | d, J=2 Hz | 1H | H-8 |
| 3.90 | s | 3H | OCH₃ |
| 3.75 | t | 4H | piperazine CH₂ |
| 2.45 | t | 4H | piperazine CH₂ |
| 2.25 | s | 3H | N-CH₃ |

### MS (ESI+)
- [M+H]⁺ = 259.15

### HPLC
- Method: C18, MeCN/H₂O + 0.1% TFA
- Expected Rt: ~6-8 min
- Purity target: >95%

---

## Scale-Up Considerations

| Scale | Method | Timeline | Cost |
|-------|--------|----------|------|
| 100 mg | Lab bench | 1 week | $300 |
| 1 g | Parallel synthesis | 2 weeks | $500 |
| 10 g | Process chemistry | 4 weeks | $2,000 |
| 100 g | CRO (WuXi, Pharmaron) | 8 weeks | $15,000 |

---

## Next Steps After Synthesis

1. **Confirm structure** — NMR, MS, HPLC
2. **Solubility** — PBS, DMSO, simulated gastric fluid
3. **Stability** — 37°C in PBS over 24h
4. **In vitro binding** — SPR or ITC against KRAS G12D
5. **Selectivity** — Compare binding to WT KRAS
6. **Cell assay** — G12D-mutant vs WT cell lines

---

## CRO Recommendations

| CRO | Specialty | Location | Turnaround |
|-----|-----------|----------|------------|
| **WuXi AppTec** | Full service | China | 4-6 weeks |
| **Enamine** | Building blocks | Ukraine | 2-3 weeks |
| **Pharmaron** | Process chem | China/US | 4-8 weeks |
| **ChemPartner** | Discovery | China | 3-5 weeks |

---

*Synthesis route designed for TIG-011a based on physics-first drug design.*
*Target: KRAS G12D selective inhibitor via salt bridge mechanism.*

**Date:** January 5, 2026  
**Status:** READY FOR SYNTHESIS
