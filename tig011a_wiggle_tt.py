#!/usr/bin/env python3
"""
TIG-011a Wiggle Test with TT Compression
=========================================
N-methyl piperazine variant - testing if methyl deepens the energy well.
"""

import numpy as np
import json
from datetime import datetime, timezone
import urllib.request
import os

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

print("=" * 70)
print("TIG-011a WIGGLE TEST WITH TT COMPRESSION")
print("N-methyl piperazine variant")
print("=" * 70)

# =============================================================================
# 1. Download PDB if not present
# =============================================================================

PDB_PATH = "6GJ8.pdb"

if not os.path.exists(PDB_PATH):
    print("Downloading 6GJ8.pdb...")
    url = "https://files.rcsb.org/download/6GJ8.pdb"
    urllib.request.urlretrieve(url, PDB_PATH)
    print("Downloaded.")

# =============================================================================
# 2. Load pocket atoms from PDB
# =============================================================================

def load_pocket_atoms(pdb_path, center, radius=15.0):
    """Load atoms within radius of center, INCLUDING cofactors."""
    atoms = []
    
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                if dist < radius:
                    element = line[76:78].strip()
                    if not element:
                        element = line[12:16].strip()[0]
                    resname = line[17:20].strip()
                    atoms.append({
                        'coord': np.array([x, y, z]),
                        'element': element,
                        'resname': resname
                    })
    
    return atoms

# LJ parameters (ε in kcal/mol, σ in Å)
LJ_PARAMS = {
    'C': (0.086, 3.4),
    'N': (0.170, 3.25),
    'O': (0.210, 2.96),
    'S': (0.250, 3.55),
    'H': (0.020, 2.5),
    'P': (0.200, 3.74),
    'MG': (0.150, 1.30),
}

# ASP-12 coordinates (the G12D mutation)
ASP12_COORD = np.array([4.46, 24.04, -4.60])

def compute_energy(mol_coords, pocket_atoms, asp12_coord):
    """Compute LJ + Coulombic energy for molecule placement."""
    total_energy = 0.0
    
    for mol_atom in mol_coords:
        mol_pos = mol_atom['coord']
        mol_elem = mol_atom['element']
        mol_eps, mol_sig = LJ_PARAMS.get(mol_elem, (0.1, 3.0))
        
        # LJ with pocket
        for pocket_atom in pocket_atoms:
            pocket_pos = pocket_atom['coord']
            pocket_elem = pocket_atom['element']
            pocket_eps, pocket_sig = LJ_PARAMS.get(pocket_elem, (0.1, 3.0))
            
            r = np.linalg.norm(mol_pos - pocket_pos)
            if r < 0.5:
                r = 0.5  # Prevent singularity
            
            # Lorentz-Berthelot combining rules
            eps = np.sqrt(mol_eps * pocket_eps)
            sig = (mol_sig + pocket_sig) / 2
            
            # LJ 6-12
            sr6 = (sig / r) ** 6
            sr12 = sr6 ** 2
            lj = 4 * eps * (sr12 - sr6)
            total_energy += lj
        
        # Coulombic with ASP-12 (salt bridge)
        if mol_elem == 'N':
            r_asp = np.linalg.norm(mol_pos - asp12_coord)
            if r_asp < 0.5:
                r_asp = 0.5
            # +1 on piperazine N, -1 on ASP carboxylate
            coulomb = -332.0 / (4.0 * r_asp)  # ε_r = 4 for protein
            total_energy += coulomb
    
    return total_energy

# =============================================================================
# 3. Generate TIG-011a molecule
# =============================================================================

SMILES_011A = "COc1ccc2ncnc(N3CCN(C)CC3)c2c1"  # N-methyl piperazine

print(f"\nTIG-011a SMILES: {SMILES_011A}")

mol = Chem.MolFromSmiles(SMILES_011A)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

# Properties
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)
hbd = Descriptors.NumHDonors(mol)
hba = Descriptors.NumHAcceptors(mol)

print(f"MW: {mw:.1f} | LogP: {logp:.2f} | TPSA: {tpsa:.1f}")
print(f"HBD: {hbd} | HBA: {hba}")

# Get conformer
conf = mol.GetConformer()
mol_atoms = []
piperazine_n_idx = None

for i, atom in enumerate(mol.GetAtoms()):
    pos = conf.GetAtomPosition(i)
    elem = atom.GetSymbol()
    mol_atoms.append({
        'coord': np.array([pos.x, pos.y, pos.z]),
        'element': elem,
        'idx': i
    })
    # Find the protonated N (the one NOT attached to methyl, still has H)
    if elem == 'N' and atom.GetTotalNumHs() > 0:
        piperazine_n_idx = i

if piperazine_n_idx is None:
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == 'N' and atom.IsInRing():
            piperazine_n_idx = i
            break

print(f"Piperazine NH index: {piperazine_n_idx}")

# =============================================================================
# 4. Load pocket
# =============================================================================

POCKET_CENTER = np.array([4.5, 20.0, -4.6])

pocket_atoms = load_pocket_atoms(PDB_PATH, POCKET_CENTER, radius=12.0)
print(f"\nPocket atoms loaded: {len(pocket_atoms)}")

# =============================================================================
# 5. Build energy grid (simplified for speed)
# =============================================================================

GRID_CENTER = np.array([3.34, 18.69, -3.55])  # Energy minimum from TIG-010
GRID_SIZE = 8.0
GRID_POINTS = 16  # Slightly smaller for speed

x_range = np.linspace(GRID_CENTER[0] - GRID_SIZE/2, GRID_CENTER[0] + GRID_SIZE/2, GRID_POINTS)
y_range = np.linspace(GRID_CENTER[1] - GRID_SIZE/2, GRID_CENTER[1] + GRID_SIZE/2, GRID_POINTS)
z_range = np.linspace(GRID_CENTER[2] - GRID_SIZE/2, GRID_CENTER[2] + GRID_SIZE/2, GRID_POINTS)

print(f"\nBuilding energy grid: {GRID_POINTS}³ = {GRID_POINTS**3} points")

mol_center = np.mean([a['coord'] for a in mol_atoms], axis=0)
piperazine_n_pos = mol_atoms[piperazine_n_idx]['coord']

energy_grid = np.zeros((GRID_POINTS, GRID_POINTS, GRID_POINTS))

for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        for k, z in enumerate(z_range):
            target = np.array([x, y, z])
            translation = target - piperazine_n_pos
            
            translated_atoms = []
            for atom in mol_atoms:
                translated_atoms.append({
                    'coord': atom['coord'] + translation,
                    'element': atom['element']
                })
            
            energy = compute_energy(translated_atoms, pocket_atoms, ASP12_COORD)
            energy_grid[i, j, k] = energy

print(f"Energy grid built. Range: [{energy_grid.min():.2f}, {energy_grid.max():.2f}] kcal/mol")

# =============================================================================
# 6. Find energy minimum
# =============================================================================

min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
min_energy = energy_grid[min_idx]
min_pos = np.array([x_range[min_idx[0]], y_range[min_idx[1]], z_range[min_idx[2]]])

dist_to_asp12 = np.linalg.norm(min_pos - ASP12_COORD)

print(f"\n{'='*50}")
print("ENERGY MINIMUM FOUND")
print(f"{'='*50}")
print(f"Position: ({min_pos[0]:.2f}, {min_pos[1]:.2f}, {min_pos[2]:.2f})")
print(f"Energy: {min_energy:.2f} kcal/mol")
print(f"Distance to ASP-12: {dist_to_asp12:.2f}Å")

# =============================================================================
# 7. Wiggle Test
# =============================================================================

print(f"\n{'='*50}")
print("WIGGLE TEST - PERTURBATION STABILITY")
print(f"{'='*50}")

def find_local_minimum(start_pos, grid, x_range, y_range, z_range):
    current = start_pos.copy()
    
    for _ in range(50):
        i = np.argmin(np.abs(x_range - current[0]))
        j = np.argmin(np.abs(y_range - current[1]))
        k = np.argmin(np.abs(z_range - current[2]))
        
        best_energy = grid[i, j, k]
        best_pos = current.copy()
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < len(x_range) and 0 <= nj < len(y_range) and 0 <= nk < len(z_range):
                        if grid[ni, nj, nk] < best_energy:
                            best_energy = grid[ni, nj, nk]
                            best_pos = np.array([x_range[ni], y_range[nj], z_range[nk]])
        
        if np.allclose(best_pos, current):
            break
        current = best_pos
    
    return current, best_energy

perturbations = [0.5, 1.0, 2.0, 5.0]
n_trials = 50

results = {}

for kick in perturbations:
    snap_back_count = 0
    final_distances = []
    
    for trial in range(n_trials):
        np.random.seed(trial * 100 + int(kick * 10))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        perturbed = min_pos + kick * direction
        relaxed_pos, relaxed_energy = find_local_minimum(perturbed, energy_grid, x_range, y_range, z_range)
        
        dist_from_min = np.linalg.norm(relaxed_pos - min_pos)
        dist_to_asp = np.linalg.norm(relaxed_pos - ASP12_COORD)
        final_distances.append(dist_to_asp)
        
        if dist_from_min < 1.0:
            snap_back_count += 1
    
    snap_back_pct = 100 * snap_back_count / n_trials
    mean_dist = np.mean(final_distances)
    std_dist = np.std(final_distances)
    
    results[kick] = {
        'snap_back_pct': snap_back_pct,
        'mean_asp_dist': mean_dist,
        'std_asp_dist': std_dist
    }
    
    print(f"Kick ±{kick}Å: {snap_back_pct:.0f}% snap back | Final: {mean_dist:.2f}±{std_dist:.2f}Å from ASP-12")

# =============================================================================
# 8. Verdict
# =============================================================================

thermal_snap = results[0.5]['snap_back_pct']
medium_snap = results[2.0]['snap_back_pct']

print(f"\n{'='*50}")
if thermal_snap >= 90 and medium_snap >= 80:
    verdict = "STABLE WELL"
    print(f"VERDICT: {verdict} ✓✓✓")
    print("N-methyl deepened the energy well!")
elif thermal_snap >= 90:
    verdict = "METASTABLE"
    print(f"VERDICT: {verdict}")
    print("Survives thermal, but well could be deeper")
else:
    verdict = "UNSTABLE"
    print(f"VERDICT: {verdict}")
    print("Does not survive thermal perturbation")
print(f"{'='*50}")

# =============================================================================
# 9. Comparison with TIG-010
# =============================================================================

print(f"\n{'='*50}")
print("COMPARISON: TIG-010 vs TIG-011a")
print(f"{'='*50}")

tig010_results = {
    0.5: {'snap_back_pct': 92, 'mean_asp_dist': 5.99},
    1.0: {'snap_back_pct': 82, 'mean_asp_dist': 5.98},
    2.0: {'snap_back_pct': 70, 'mean_asp_dist': 6.17},
    5.0: {'snap_back_pct': 36, 'mean_asp_dist': 7.85}
}

print(f"{'Kick':<8} {'TIG-010':<15} {'TIG-011a':<15} {'Delta':<10}")
print("-" * 48)
for kick in perturbations:
    t010 = tig010_results[kick]['snap_back_pct']
    t011 = results[kick]['snap_back_pct']
    delta = t011 - t010
    sign = "+" if delta > 0 else ""
    print(f"+/-{kick}A    {t010:.0f}%            {t011:.0f}%            {sign}{delta:.0f}%")

# =============================================================================
# 10. Save attestation
# =============================================================================

attestation = {
    'molecule': 'TIG-011a',
    'smiles': SMILES_011A,
    'modification': 'N-methyl piperazine',
    'properties': {
        'mw': round(mw, 1),
        'logp': round(logp, 2),
        'tpsa': round(tpsa, 1),
        'hbd': hbd,
        'hba': hba
    },
    'energy_minimum': {
        'position': [round(x, 2) for x in min_pos],
        'energy_kcal_mol': round(min_energy, 2),
        'distance_to_asp12': round(dist_to_asp12, 2)
    },
    'wiggle_test': {
        str(kick): {
            'snap_back_pct': round(r['snap_back_pct'], 1),
            'mean_asp_dist': round(r['mean_asp_dist'], 2),
            'std_asp_dist': round(r['std_asp_dist'], 2)
        }
        for kick, r in results.items()
    },
    'verdict': verdict,
    'comparison_to_tig010': {
        str(kick): {
            'tig010_pct': tig010_results[kick]['snap_back_pct'],
            'tig011a_pct': results[kick]['snap_back_pct'],
            'delta': results[kick]['snap_back_pct'] - tig010_results[kick]['snap_back_pct']
        }
        for kick in perturbations
    },
    'timestamp': datetime.now(timezone.utc).isoformat()
}

with open('TIG011A_WIGGLE_TT.json', 'w') as f:
    json.dump(attestation, f, indent=2)

print(f"\nAttestation saved to TIG011A_WIGGLE_TT.json")

# Final recommendation
print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")
improvement = results[2.0]['snap_back_pct'] - tig010_results[2.0]['snap_back_pct']
if verdict == "STABLE WELL":
    print("TIG-011a shows IMPROVED binding over TIG-010")
    print("-> Proceed with TIG-011a as lead compound")
elif improvement > 0:
    print(f"TIG-011a shows +{improvement:.0f}% improvement at +/-2A")
    print("-> Consider parallel synthesis of both")
else:
    print("TIG-011a shows NO improvement over TIG-010")
    print("-> Try alternative modifications (TIG-011b, TIG-011c)")
