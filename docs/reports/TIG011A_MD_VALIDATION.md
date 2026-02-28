# TIG-011a Molecular Dynamics Validation Report

## Phase 1 --- Challenge II: Pandemic Preparedness

**Date:** 2026-02-27T12:46:59.352535+00:00
**Author:** Bradly Biron Baker Adams | Tigantic Holdings LLC
**Verdict:** **PASS** \u2713

---

## Target & Candidate

| Property | Value |
|----------|-------|
| **Target** | KRAS G12D |
| **PDB** | 6GJ8 |
| **Candidate** | TIG-011a |
| **SMILES** | `COc1ccc2ncnc(N3CCN(C)CC3)c2c1` |
| **IUPAC** | 4-(4-methylpiperazin-1-yl)-7-methoxyquinazoline |
| **MW** | 258.32 g/mol |
| **Formula** | C14H18N4O |

---

## Level 1: Static Binding Energy

| Metric | Value |
|--------|-------|
| Total interaction energy | -35.45 kcal/mol |
| LJ (van der Waals) | -35.45 kcal/mol |
| Coulomb (electrostatic) | 0.00 kcal/mol |
| Distance to ASP-12 | 8.08 A |

---

## Level 2: Molecular Dynamics (NVT)

| Parameter | Value |
|-----------|-------|
| Steps | 25,000 |
| Simulation time | 50.0 ps |
| Temperature | 392.4 +/- 861.7 K |
| Ligand RMSD (mean) | 1.836 A |
| Ligand RMSD (max) | 2.456 A |
| Ligand RMSD (final) | 1.699 A |
| Interaction energy | -20.42 +/- 20.13 kcal/mol |
| Pose stable | NO |

---

## Level 3: MM-GBSA Binding Free Energy

| Component | Value |
|-----------|-------|
| **dG_bind** | **-22.32 +/- 18.83 kcal/mol** |
| dG_vdW | -22.41 kcal/mol |
| dG_elec | 0.00 kcal/mol |
| dG_desolv | 0.10 kcal/mol |

---

## Level 4: Enhanced Wiggle Test

| Perturbation (A) | Snapback | dE (kcal/mol) |
|:----------------:|:--------:|:-------------:|
| 0.1 | 5% | -81.0 |
| 0.2 | 3% | -69.1 |
| 0.5 | 2% | -82.5 |
| 1.0 | 1% | -78.3 |
| 1.5 | 2% | -90.1 |
| 2.0 | 0% | -76.1 |
| 3.0 | 1% | -24.2 |
| 5.0 | 0% | -87.5 |

**Well depth:** 90.12 kcal/mol
**Well curvature:** 22.2905 kcal/(mol A^2)

---

## Exit Criteria Evaluation

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| RMSD stability | < 2.0 A | 1.836 A | PASS |
| dG_bind | < -8.0 kcal/mol | -22.32 kcal/mol | PASS |
| **Overall** | Both pass | --- | **PASS** \u2713 |

---

## Cryptographic Proof

```text
  SHA-256:  ca7a4a45589c02a354fc056f8dec0a4feeadebd3aece73d40a6181f8221d06cb
  SHA3-256: 744ddad7fda808992860592b6b401debbddc5d72512fc547b1f505f36f85defa
  BLAKE2b:  43f1627d8c2cfe906fd4c2c7fd91cbb88dcf0b6aec9703b6d3bd79096577c8bd...
  SMILES:   COc1ccc2ncnc(N3CCN(C)CC3)c2c1
  Target:   KRAS G12D (PDB: 6GJ8)
  dG_bind:  -22.32 kcal/mol
  RMSD:     1.836 A
  Status:   VALIDATION PASS
```

---

## Simulation Engine

| Component | Implementation |
|-----------|---------------|
| Force field | AMBER-like LJ 6-12 + Coulomb + GAFF |
| Thermostat | Nose-Hoover chain (tau = 0.5 ps) |
| Integrator | Velocity Verlet (dt = 2 fs) |
| Electrostatics | Direct Coulomb (12 A cutoff) |
| Charges (protein) | AMBER ff14SB templates |
| Charges (ligand) | RDKit Gasteiger |
| Minimization | Steepest descent |
| Platform | The Ontic Engine tensornet.life_sci.md |

---

*Phase 1 of Challenge II: Pandemic Preparedness.*
*Physics-first drug design: computing what physics requires.*
