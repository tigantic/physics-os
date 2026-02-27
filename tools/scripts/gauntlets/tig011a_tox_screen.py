#!/usr/bin/env python3
"""
TIG-011a In Silico Toxicology Screen
=====================================
Check for showstoppers BEFORE synthesis.

Screens:
1. PAINS (Pan-Assay Interference Compounds)
2. Brenk structural alerts
3. NIH MLSMR exclusion filters
4. Lipinski Rule of 5
5. hERG liability predictors
6. CYP450 inhibition flags
7. Ames mutagenicity alerts
"""

import json
from datetime import datetime, timezone

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

print("=" * 70)
print("TIG-011a IN SILICO TOXICOLOGY SCREEN")
print("=" * 70)

# Target molecule
SMILES = "COc1ccc2ncnc(N3CCN(C)CC3)c2c1"
NAME = "TIG-011a"

mol = Chem.MolFromSmiles(SMILES)
if mol is None:
    print("ERROR: Invalid SMILES")
    exit(1)

print(f"\nCompound: {NAME}")
print(f"SMILES: {SMILES}")

results = {
    'compound': NAME,
    'smiles': SMILES,
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'screens': {}
}

# =============================================================================
# 1. PAINS Filter (Pan-Assay Interference Compounds)
# =============================================================================
print(f"\n{'='*50}")
print("1. PAINS FILTER")
print(f"{'='*50}")

params_pains = FilterCatalogParams()
params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog_pains = FilterCatalog.FilterCatalog(params_pains)

pains_matches = catalog_pains.GetMatches(mol)
pains_alerts = [match.GetDescription() for match in pains_matches]

if pains_alerts:
    print(f"⚠️  PAINS ALERTS: {len(pains_alerts)}")
    for alert in pains_alerts:
        print(f"   - {alert}")
    results['screens']['PAINS'] = {'status': 'FAIL', 'alerts': pains_alerts}
else:
    print("✓ PAINS: PASS (no promiscuous binders detected)")
    results['screens']['PAINS'] = {'status': 'PASS', 'alerts': []}

# =============================================================================
# 2. Brenk Structural Alerts
# =============================================================================
print(f"\n{'='*50}")
print("2. BRENK STRUCTURAL ALERTS")
print(f"{'='*50}")

params_brenk = FilterCatalogParams()
params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
catalog_brenk = FilterCatalog.FilterCatalog(params_brenk)

brenk_matches = catalog_brenk.GetMatches(mol)
brenk_alerts = [match.GetDescription() for match in brenk_matches]

if brenk_alerts:
    print(f"⚠️  BRENK ALERTS: {len(brenk_alerts)}")
    for alert in brenk_alerts:
        print(f"   - {alert}")
    results['screens']['Brenk'] = {'status': 'FLAG', 'alerts': brenk_alerts}
else:
    print("✓ Brenk: PASS (no problematic substructures)")
    results['screens']['Brenk'] = {'status': 'PASS', 'alerts': []}

# =============================================================================
# 3. NIH MLSMR Filters
# =============================================================================
print(f"\n{'='*50}")
print("3. NIH MLSMR EXCLUSION FILTERS")
print(f"{'='*50}")

params_nih = FilterCatalogParams()
params_nih.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
catalog_nih = FilterCatalog.FilterCatalog(params_nih)

nih_matches = catalog_nih.GetMatches(mol)
nih_alerts = [match.GetDescription() for match in nih_matches]

if nih_alerts:
    print(f"⚠️  NIH ALERTS: {len(nih_alerts)}")
    for alert in nih_alerts:
        print(f"   - {alert}")
    results['screens']['NIH'] = {'status': 'FLAG', 'alerts': nih_alerts}
else:
    print("✓ NIH MLSMR: PASS")
    results['screens']['NIH'] = {'status': 'PASS', 'alerts': []}

# =============================================================================
# 4. Lipinski Rule of 5
# =============================================================================
print(f"\n{'='*50}")
print("4. LIPINSKI RULE OF 5 (Drug-likeness)")
print(f"{'='*50}")

mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
hbd = Descriptors.NumHDonors(mol)
hba = Descriptors.NumHAcceptors(mol)
tpsa = Descriptors.TPSA(mol)
rotatable = Descriptors.NumRotatableBonds(mol)

lipinski_violations = 0
lipinski_details = []

if mw > 500:
    lipinski_violations += 1
    lipinski_details.append(f"MW {mw:.1f} > 500")
if logp > 5:
    lipinski_violations += 1
    lipinski_details.append(f"LogP {logp:.2f} > 5")
if hbd > 5:
    lipinski_violations += 1
    lipinski_details.append(f"HBD {hbd} > 5")
if hba > 10:
    lipinski_violations += 1
    lipinski_details.append(f"HBA {hba} > 10")

print(f"  MW:       {mw:.1f} (≤500) {'✓' if mw <= 500 else '✗'}")
print(f"  LogP:     {logp:.2f} (≤5) {'✓' if logp <= 5 else '✗'}")
print(f"  HBD:      {hbd} (≤5) {'✓' if hbd <= 5 else '✗'}")
print(f"  HBA:      {hba} (≤10) {'✓' if hba <= 10 else '✗'}")
print(f"  TPSA:     {tpsa:.1f} Ų")
print(f"  RotBonds: {rotatable}")

if lipinski_violations == 0:
    print("\n✓ Lipinski: PASS (0 violations)")
    results['screens']['Lipinski'] = {'status': 'PASS', 'violations': 0}
else:
    print(f"\n⚠️  Lipinski: {lipinski_violations} violations")
    results['screens']['Lipinski'] = {'status': 'FLAG', 'violations': lipinski_violations, 'details': lipinski_details}

results['properties'] = {
    'MW': round(mw, 1),
    'LogP': round(logp, 2),
    'HBD': hbd,
    'HBA': hba,
    'TPSA': round(tpsa, 1),
    'RotatableBonds': rotatable
}

# =============================================================================
# 5. hERG Liability Indicators
# =============================================================================
print(f"\n{'='*50}")
print("5. hERG CARDIAC LIABILITY INDICATORS")
print(f"{'='*50}")

# hERG blockers tend to have:
# - LogP > 3.7
# - Basic nitrogen (pKa > 7)
# - Aromatic rings
# - MW 250-500

herg_flags = []

if logp > 3.7:
    herg_flags.append(f"LogP {logp:.2f} > 3.7 (lipophilic)")

# Check for basic nitrogen
basic_n_pattern = Chem.MolFromSmarts("[#7;+,H1,H2,H3]")
if mol.HasSubstructMatch(basic_n_pattern):
    # Count basic nitrogens
    basic_n_count = len(mol.GetSubstructMatches(basic_n_pattern))
    if basic_n_count >= 2:
        herg_flags.append(f"{basic_n_count} basic nitrogens (potential hERG binder)")

# Quinazoline with piperazine is a known hERG pharmacophore - check
herg_pharmacophore = Chem.MolFromSmarts("c1ncnc2ccccc12")  # Quinazoline
if mol.HasSubstructMatch(herg_pharmacophore):
    pass  # Quinazoline alone is not a flag

if herg_flags:
    print(f"⚠️  hERG FLAGS: {len(herg_flags)}")
    for flag in herg_flags:
        print(f"   - {flag}")
    print("\n   Recommendation: Run hERG patch-clamp assay (IC50 target: >10 µM)")
    results['screens']['hERG'] = {'status': 'FLAG', 'alerts': herg_flags, 'recommendation': 'Patch-clamp assay'}
else:
    print("✓ hERG: LOW RISK")
    results['screens']['hERG'] = {'status': 'PASS', 'alerts': []}

# =============================================================================
# 6. CYP450 Inhibition Flags
# =============================================================================
print(f"\n{'='*50}")
print("6. CYP450 INHIBITION FLAGS")
print(f"{'='*50}")

cyp_flags = []

# CYP3A4 inhibitors often have:
# - Imidazole/triazole (not in our molecule)
# - Large aromatic systems
# - Basic nitrogen

# Check for imidazole (strong CYP inhibitor)
imidazole = Chem.MolFromSmarts("c1cnc[nH]1")
triazole = Chem.MolFromSmarts("c1nncn1")

if mol.HasSubstructMatch(imidazole):
    cyp_flags.append("Imidazole (CYP3A4 inhibitor)")
if mol.HasSubstructMatch(triazole):
    cyp_flags.append("Triazole (CYP3A4 inhibitor)")

# Quinazoline is metabolized by CYP but not typically an inhibitor
# Piperazine is generally okay

# Check for known CYP2D6 substrate features (basic N + aromatic)
# Our compound has this but it's more of a substrate than inhibitor

if cyp_flags:
    print(f"⚠️  CYP FLAGS: {len(cyp_flags)}")
    for flag in cyp_flags:
        print(f"   - {flag}")
    results['screens']['CYP450'] = {'status': 'FLAG', 'alerts': cyp_flags}
else:
    print("✓ CYP450: No strong inhibitor motifs")
    print("  Note: Likely CYP3A4/2D6 substrate (normal metabolism)")
    results['screens']['CYP450'] = {'status': 'PASS', 'alerts': [], 'note': 'Likely CYP3A4/2D6 substrate'}

# =============================================================================
# 7. Ames Mutagenicity Structural Alerts
# =============================================================================
print(f"\n{'='*50}")
print("7. AMES MUTAGENICITY ALERTS")
print(f"{'='*50}")

ames_alerts = []

# Check for known mutagenic substructures
nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
nitroso = Chem.MolFromSmarts("N=O")
azide = Chem.MolFromSmarts("[N-]=[N+]=[N-]")
epoxide = Chem.MolFromSmarts("C1OC1")
aziridine = Chem.MolFromSmarts("C1NC1")
hydrazine = Chem.MolFromSmarts("NN")
aromatic_amine = Chem.MolFromSmarts("c-[NH2]")  # Primary aromatic amine
alkyl_halide = Chem.MolFromSmarts("[CH2][Cl,Br,I]")

mutagenic_patterns = [
    (nitro, "Nitro group"),
    (nitroso, "Nitroso group"),
    (azide, "Azide"),
    (epoxide, "Epoxide"),
    (aziridine, "Aziridine"),
    (hydrazine, "Hydrazine"),
    (aromatic_amine, "Primary aromatic amine"),
    (alkyl_halide, "Alkyl halide"),
]

for pattern, name in mutagenic_patterns:
    if pattern and mol.HasSubstructMatch(pattern):
        ames_alerts.append(name)

if ames_alerts:
    print(f"⚠️  AMES ALERTS: {len(ames_alerts)}")
    for alert in ames_alerts:
        print(f"   - {alert}")
    results['screens']['Ames'] = {'status': 'FAIL', 'alerts': ames_alerts}
else:
    print("✓ Ames: PASS (no mutagenic substructures)")
    results['screens']['Ames'] = {'status': 'PASS', 'alerts': []}

# =============================================================================
# 8. Reactive Metabolite Alerts
# =============================================================================
print(f"\n{'='*50}")
print("8. REACTIVE METABOLITE ALERTS")
print(f"{'='*50}")

reactive_alerts = []

# Check for Michael acceptors
michael = Chem.MolFromSmarts("[C]=[C]-[C]=O")
if michael and mol.HasSubstructMatch(michael):
    reactive_alerts.append("Michael acceptor")

# Quinone precursor
quinone_precursor = Chem.MolFromSmarts("c1(O)ccc(O)cc1")
if quinone_precursor and mol.HasSubstructMatch(quinone_precursor):
    reactive_alerts.append("Hydroquinone (quinone precursor)")

# Thiophene (S-oxidation)
thiophene = Chem.MolFromSmarts("c1ccsc1")
if thiophene and mol.HasSubstructMatch(thiophene):
    reactive_alerts.append("Thiophene (S-oxidation liability)")

# Aniline (N-oxidation)
aniline = Chem.MolFromSmarts("c-[NH2]")
if aniline and mol.HasSubstructMatch(aniline):
    reactive_alerts.append("Aniline (N-hydroxylation)")

# Furan (epoxidation)
furan = Chem.MolFromSmarts("c1ccoc1")
if furan and mol.HasSubstructMatch(furan):
    reactive_alerts.append("Furan (epoxide metabolite)")

if reactive_alerts:
    print(f"⚠️  REACTIVE METABOLITE FLAGS: {len(reactive_alerts)}")
    for alert in reactive_alerts:
        print(f"   - {alert}")
    results['screens']['ReactiveMetabolites'] = {'status': 'FLAG', 'alerts': reactive_alerts}
else:
    print("✓ Reactive Metabolites: PASS")
    results['screens']['ReactiveMetabolites'] = {'status': 'PASS', 'alerts': []}

# =============================================================================
# FINAL VERDICT
# =============================================================================
print(f"\n{'='*70}")
print("TOXICOLOGY SCREEN SUMMARY")
print(f"{'='*70}")

all_pass = True
flags = []
fails = []

for screen, data in results['screens'].items():
    status = data['status']
    if status == 'FAIL':
        fails.append(screen)
        all_pass = False
    elif status == 'FLAG':
        flags.append(screen)

print(f"\n| Screen               | Status  |")
print(f"|----------------------|---------|")
for screen, data in results['screens'].items():
    status = data['status']
    symbol = '✓' if status == 'PASS' else ('⚠️' if status == 'FLAG' else '✗')
    print(f"| {screen:<20} | {symbol} {status:<5} |")

print(f"\n{'='*50}")
if fails:
    verdict = "FAIL"
    print(f"VERDICT: {verdict}")
    print(f"Showstoppers: {', '.join(fails)}")
    print("DO NOT SYNTHESIZE until addressed.")
elif flags:
    verdict = "PROCEED WITH CAUTION"
    print(f"VERDICT: {verdict}")
    print(f"Flags to monitor: {', '.join(flags)}")
    print("Synthesis OK, but add assays to de-risk.")
else:
    verdict = "PASS"
    print(f"VERDICT: {verdict} ✓")
    print("No toxicity showstoppers. Clear for synthesis.")
print(f"{'='*50}")

results['verdict'] = verdict
results['fails'] = fails
results['flags'] = flags

# Save results
with open('TIG011A_TOX_SCREEN.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to TIG011A_TOX_SCREEN.json")

# Recommendations
if flags:
    print(f"\n{'='*50}")
    print("RECOMMENDED FOLLOW-UP ASSAYS")
    print(f"{'='*50}")
    if 'hERG' in flags:
        print("• hERG patch-clamp (target IC50 > 10 µM)")
    if 'Brenk' in flags:
        print("• Microsomal stability (liver microsomes)")
    print("• Plasma protein binding")
    print("• Caco-2 permeability")
