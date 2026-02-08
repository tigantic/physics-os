# SnHf-F Quantum Well Resist Synthesis Protocol

## Lab Notebook Entry

**Date:** January 5, 2026  
**Operator:** _______________  
**Notebook Page:** _______________

---

## Objective

Synthesize **SnHf-F Quantum Well Resist** (Sn₈Hf₄O₁₂(CF₃COO)₁₂) for sub-1nm EUV lithography.

**Target:** 5g, >99% purity (semiconductor grade)

---

## Safety — CRITICAL

| Hazard | Precaution |
|--------|------------|
| HfCl₄ | Highly reactive with moisture. Handle under N₂/Ar. Burns skin. |
| SnCl₂ | Toxic. Avoid inhalation. Gloves required. |
| CF₃COOH (TFA) | Corrosive, volatile. Fume hood MANDATORY. |
| THF | Flammable, peroxide former. Check for peroxides before use. |
| Toluene | Flammable, CNS depressant. Fume hood. |

**PPE Required:** Lab coat, nitrile gloves (double), safety glasses, face shield for TFA, closed-toe shoes

**Environment:** All reactions under inert atmosphere (N₂ or Ar). Schlenk line or glovebox required.

---

## Materials Checklist

### Step 1: Core Formation

| Item | Amount | MW | mmol | ✓ |
|------|--------|-----|------|---|
| SnCl₂ (anhydrous) | 1.52 g | 189.6 | 8.0 | ☐ |
| HfCl₄ (anhydrous) | 1.28 g | 320.3 | 4.0 | ☐ |
| H₂O (degassed) | 0.22 mL | 18.0 | 12.0 | ☐ |
| THF (anhydrous) | 50 mL | - | - | ☐ |
| Triethylamine | 3.4 mL | 101.2 | 24.0 | ☐ |

### Step 2: Ligand Exchange

| Item | Amount | MW | mmol | ✓ |
|------|--------|-----|------|---|
| CF₃COOH (TFA) | 1.8 mL | 114.0 | 24.0 | ☐ |
| Toluene (anhydrous) | 100 mL | - | - | ☐ |

### Step 3: Purification

| Item | Amount | ✓ |
|------|--------|---|
| Hexane (anhydrous) | 200 mL | ☐ |
| Molecular sieves 4Å | 20 g | ☐ |
| Sublimation apparatus | - | ☐ |

---

## Equipment Checklist

| Item | ✓ |
|------|---|
| Schlenk line (N₂/vacuum) | ☐ |
| 250 mL Schlenk flask (×2) | ☐ |
| Magnetic stir bars | ☐ |
| Ice bath | ☐ |
| Oil bath + hot plate | ☐ |
| Reflux condenser | ☐ |
| Cannula + septa | ☐ |
| Syringe (10 mL, glass) | ☐ |
| Schlenk frit (medium porosity) | ☐ |
| Sublimation apparatus | ☐ |
| Vacuum pump (10⁻⁶ Torr capable) | ☐ |
| Glovebox (optional but recommended) | ☐ |

---

## Procedure

### Step 1: Core Formation — Sn₈Hf₄O₁₂ (4 hours)

**Reaction:**
```
8 SnCl₂ + 4 HfCl₄ + 12 H₂O + 24 NEt₃ → Sn₈Hf₄O₁₂ + 24 [HNEt₃]Cl
```

1. ☐ Set up 250 mL Schlenk flask under N₂
2. ☐ Add **SnCl₂** (1.52 g, 8.0 mmol) to flask
3. ☐ Add **HfCl₄** (1.28 g, 4.0 mmol) to flask
4. ☐ Add **anhydrous THF** (50 mL) via cannula
5. ☐ Stir until dissolved (~10 min, may need gentle warming)
6. ☐ Cool to **0°C** (ice bath)
7. ☐ Add **triethylamine** (3.4 mL, 24 mmol) dropwise over 10 min
8. ☐ Add **degassed H₂O** (0.22 mL, 12 mmol) dropwise over 5 min

   > ⚠️ **CRITICAL:** Add water SLOWLY. Exothermic reaction. White precipitate of Et₃N·HCl will form.

9. ☐ Warm to **room temperature**
10. ☐ Stir for **2 hours**
11. ☐ Filter through Schlenk frit to remove Et₃N·HCl
12. ☐ Wash solid with THF (2 × 20 mL)
13. ☐ Concentrate filtrate under vacuum

**Intermediate yield:** _______ g (theoretical: ~1.8 g crude core)  
**Appearance:** _______ (expect: white to off-white solid)

---

### Step 2: Ligand Exchange — Fluorination (6 hours)

**Reaction:**
```
Sn₈Hf₄O₁₂ + 12 CF₃COOH → Sn₈Hf₄O₁₂(CF₃COO)₁₂ + 6 H₂O
```

1. ☐ Transfer crude core to clean 250 mL Schlenk flask
2. ☐ Add **anhydrous toluene** (100 mL)
3. ☐ Add **molecular sieves 4Å** (5 g) — to absorb H₂O byproduct
4. ☐ Heat to **60°C**
5. ☐ Add **CF₃COOH** (1.8 mL, 24 mmol) dropwise over 15 min

   > ⚠️ **FUME HOOD MANDATORY.** TFA is volatile and corrosive.

6. ☐ Heat to **reflux** (~110°C)
7. ☐ Stir under reflux for **4 hours**
8. ☐ Cool to room temperature
9. ☐ Filter to remove molecular sieves
10. ☐ Concentrate under vacuum to ~20 mL

**Observation:** Solution should become clear during reaction.

---

### Step 3: Purification (2 hours + overnight sublimation)

**Option A: Precipitation (Quick)**

1. ☐ Add concentrated solution dropwise to **cold hexane** (200 mL, -20°C)
2. ☐ White precipitate should form immediately
3. ☐ Filter on Schlenk frit
4. ☐ Wash with cold hexane (3 × 20 mL)
5. ☐ Dry under vacuum (1 hour)

**Crude yield:** _______ g  
**Appearance:** _______ (expect: white powder)

**Option B: Sublimation (High Purity — REQUIRED for lithography)**

1. ☐ Load crude product into sublimation apparatus
2. ☐ Apply vacuum to **10⁻⁶ Torr**
3. ☐ Heat gradually:
   - 80°C for 1 hour (remove volatiles)
   - 120°C for 1 hour (remove residual solvent)
   - **180°C** for sublimation
4. ☐ Collect sublimate on cold finger (water-cooled or dry ice)
5. ☐ Scrape pure product from cold finger

**Pure yield:** _______ g  
**% Yield:** _______ % (target: >60% from SnCl₂)  
**Appearance:** _______ (expect: white crystalline powder)

---

## Characterization

### Visual

| Property | Observed | Expected |
|----------|----------|----------|
| Color | _______ | White |
| Form | _______ | Crystalline powder |
| Odor | _______ | None (TFA removed) |

### Melting/Sublimation Point

| Property | Observed | Expected |
|----------|----------|----------|
| Sublimation | _______°C | ~180°C at 10⁻⁶ Torr |
| Decomposition | _______°C | >250°C |

### Elemental Analysis

| Element | Expected % | Observed % | ✓ |
|---------|------------|------------|---|
| Sn | 29.2 | _______ | ☐ |
| Hf | 22.0 | _______ | ☐ |
| C | 8.9 | _______ | ☐ |
| F | 21.0 | _______ | ☐ |
| O | 18.9 | _______ | ☐ |

### Mass Spec (MALDI-TOF or ESI)

| Observed m/z | Expected | Assignment | ✓ |
|--------------|----------|------------|---|
| _______ | 3246 | [M]⁺ | ☐ |
| _______ | 3133 | [M - CF₃COO]⁺ | ☐ |

### ¹⁹F NMR (CDCl₃ or C₆D₆)

| δ (ppm) | Expected | Observed | ✓ |
|---------|----------|----------|---|
| -75 to -78 | CF₃ (s) | _______ | ☐ |

### IR (ATR)

| Wavenumber (cm⁻¹) | Assignment | Observed | ✓ |
|-------------------|------------|----------|---|
| 1680-1720 | C=O stretch (TFA) | _______ | ☐ |
| 1150-1200 | C-F stretch | _______ | ☐ |
| 500-600 | Sn-O / Hf-O | _______ | ☐ |

### XPS (If available)

| Binding Energy (eV) | Assignment | ✓ |
|---------------------|------------|---|
| 486-488 | Sn 3d₅/₂ | ☐ |
| 16-18 | Hf 4f₇/₂ | ☐ |
| 687-689 | F 1s | ☐ |

---

## Lithography Testing (Downstream)

### Film Preparation

1. ☐ Dissolve product in PGMEA (2-5 wt%)
2. ☐ Filter through 0.1 µm PTFE filter
3. ☐ Spin coat: 2000 rpm, 30 sec
4. ☐ Post-apply bake: 100°C, 60 sec
5. ☐ Target thickness: 30-50 nm

### EUV Exposure (Requires synchrotron or scanner)

| Parameter | Target |
|-----------|--------|
| Wavelength | 13.5 nm |
| Dose to clear | <30 mJ/cm² |
| Line/Space | 1 nm / 1 nm (target) |
| LER | <1.0 nm |

---

## Storage

| Condition | Instruction |
|-----------|-------------|
| Container | Amber glass vial, PTFE-lined cap |
| Atmosphere | N₂ or Ar (moisture sensitive) |
| Temperature | Room temp (dark) or 4°C for long term |
| Desiccant | Yes |
| Light | Protect from UV/ambient light |
| Shelf life | 6 months sealed, 1 week after opening |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Yellow/brown product | Oxidation | Repeat under stricter inert conditions |
| Low yield in sublimation | Decomposition | Lower sublimation temp to 160°C |
| Residual TFA smell | Incomplete drying | Extended vacuum drying, add mol sieves |
| Product not dissolving in PGMEA | Particle size | Ball mill or sonicate |
| Poor film quality | Aggregation | Filter, dilute, optimize spin conditions |

---

## Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Operator | _______ | _______ | _______ |
| Reviewer | _______ | _______ | _______ |
| Safety Officer | _______ | _______ | _______ |

---

## Results Summary

```
┌─────────────────────────────────────────────────────┐
│  SnHf-F QUANTUM WELL RESIST SYNTHESIS               │
├─────────────────────────────────────────────────────┤
│  Batch ID: SNHF-_______                             │
│  Mass obtained: _______ g                           │
│  Yield: _______% (target >60%)                      │
│  Purity: _______% (target >99%)                     │
│  Sublimation: _______°C (expect ~180°C)             │
│  ¹⁹F NMR δ: _______ ppm (expect -75 to -78)         │
│  Status: ☐ PASS  ☐ FAIL                             │
└─────────────────────────────────────────────────────┘
```

---

## Cost Estimate

| Reagent | Amount | Cost |
|---------|--------|------|
| HfCl₄ | 2 g | $150 |
| SnCl₂ | 2 g | $25 |
| CF₃COOH | 5 mL | $20 |
| THF (anhydrous) | 100 mL | $30 |
| Toluene (anhydrous) | 200 mL | $25 |
| Triethylamine | 10 mL | $10 |
| Hexane | 500 mL | $20 |
| **Total** | | **~$280** |

**Expected yield:** 3-4 g (enough for ~100 wafer coatings)

---

*Protocol for SnHf-F Quantum Well Resist synthesis*
*Target: Sub-1nm EUV lithography*
*Physics-first semiconductor design → Bench chemistry*
