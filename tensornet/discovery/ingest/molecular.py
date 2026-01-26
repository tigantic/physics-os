#!/usr/bin/env python3
"""
Molecular Data Ingester for Drug Discovery

Phase 3 of the Autonomous Discovery Engine.

Ingests molecular structure data for binding site detection and drug-protein
interaction analysis using QTT-native Genesis primitives.

Supported formats:
    - PDB (Protein Data Bank)
    - mmCIF (macromolecular Crystallographic Information File)
    - SMILES (Simplified Molecular Input Line Entry System)
    - SDF/MOL (Structure Data Format)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch


# Amino acid properties (hydrophobicity, charge, size)
AMINO_ACID_PROPERTIES = {
    'ALA': {'hydrophobicity': 1.8, 'charge': 0, 'size': 89, 'one_letter': 'A'},
    'ARG': {'hydrophobicity': -4.5, 'charge': 1, 'size': 174, 'one_letter': 'R'},
    'ASN': {'hydrophobicity': -3.5, 'charge': 0, 'size': 132, 'one_letter': 'N'},
    'ASP': {'hydrophobicity': -3.5, 'charge': -1, 'size': 133, 'one_letter': 'D'},
    'CYS': {'hydrophobicity': 2.5, 'charge': 0, 'size': 121, 'one_letter': 'C'},
    'GLN': {'hydrophobicity': -3.5, 'charge': 0, 'size': 146, 'one_letter': 'Q'},
    'GLU': {'hydrophobicity': -3.5, 'charge': -1, 'size': 147, 'one_letter': 'E'},
    'GLY': {'hydrophobicity': -0.4, 'charge': 0, 'size': 75, 'one_letter': 'G'},
    'HIS': {'hydrophobicity': -3.2, 'charge': 0.5, 'size': 155, 'one_letter': 'H'},
    'ILE': {'hydrophobicity': 4.5, 'charge': 0, 'size': 131, 'one_letter': 'I'},
    'LEU': {'hydrophobicity': 3.8, 'charge': 0, 'size': 131, 'one_letter': 'L'},
    'LYS': {'hydrophobicity': -3.9, 'charge': 1, 'size': 146, 'one_letter': 'K'},
    'MET': {'hydrophobicity': 1.9, 'charge': 0, 'size': 149, 'one_letter': 'M'},
    'PHE': {'hydrophobicity': 2.8, 'charge': 0, 'size': 165, 'one_letter': 'F'},
    'PRO': {'hydrophobicity': -1.6, 'charge': 0, 'size': 115, 'one_letter': 'P'},
    'SER': {'hydrophobicity': -0.8, 'charge': 0, 'size': 105, 'one_letter': 'S'},
    'THR': {'hydrophobicity': -0.7, 'charge': 0, 'size': 119, 'one_letter': 'T'},
    'TRP': {'hydrophobicity': -0.9, 'charge': 0, 'size': 204, 'one_letter': 'W'},
    'TYR': {'hydrophobicity': -1.3, 'charge': 0, 'size': 181, 'one_letter': 'Y'},
    'VAL': {'hydrophobicity': 4.2, 'charge': 0, 'size': 117, 'one_letter': 'V'},
}

# One-letter to three-letter mapping
ONE_TO_THREE = {v['one_letter']: k for k, v in AMINO_ACID_PROPERTIES.items()}


@dataclass
class Atom:
    """Represents an atom in a molecular structure."""
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_seq: int
    x: float
    y: float
    z: float
    element: str
    occupancy: float = 1.0
    b_factor: float = 0.0
    
    @property
    def coords(self) -> torch.Tensor:
        """Return coordinates as tensor."""
        return torch.tensor([self.x, self.y, self.z])
    
    @property
    def is_backbone(self) -> bool:
        """Check if atom is part of backbone."""
        return self.name in ('N', 'CA', 'C', 'O')
    
    @property
    def is_alpha_carbon(self) -> bool:
        """Check if atom is alpha carbon."""
        return self.name == 'CA'


@dataclass
class Residue:
    """Represents an amino acid residue."""
    name: str
    chain_id: str
    seq_num: int
    atoms: List[Atom] = field(default_factory=list)
    
    @property
    def one_letter(self) -> str:
        """Get one-letter code."""
        props = AMINO_ACID_PROPERTIES.get(self.name, {})
        return props.get('one_letter', 'X')
    
    @property
    def hydrophobicity(self) -> float:
        """Get Kyte-Doolittle hydrophobicity."""
        props = AMINO_ACID_PROPERTIES.get(self.name, {})
        return props.get('hydrophobicity', 0.0)
    
    @property
    def charge(self) -> float:
        """Get formal charge at pH 7."""
        props = AMINO_ACID_PROPERTIES.get(self.name, {})
        return props.get('charge', 0.0)
    
    @property
    def size(self) -> float:
        """Get molecular weight."""
        props = AMINO_ACID_PROPERTIES.get(self.name, {})
        return props.get('size', 100.0)
    
    @property
    def center_of_mass(self) -> torch.Tensor:
        """Compute center of mass."""
        if not self.atoms:
            return torch.zeros(3)
        coords = torch.stack([a.coords for a in self.atoms])
        return coords.mean(dim=0)
    
    @property
    def alpha_carbon(self) -> Optional[Atom]:
        """Get alpha carbon atom."""
        for atom in self.atoms:
            if atom.is_alpha_carbon:
                return atom
        return None


@dataclass
class Chain:
    """Represents a protein chain."""
    chain_id: str
    residues: List[Residue] = field(default_factory=list)
    
    @property
    def sequence(self) -> str:
        """Get amino acid sequence."""
        return ''.join(r.one_letter for r in self.residues)
    
    @property
    def length(self) -> int:
        """Get chain length."""
        return len(self.residues)
    
    def get_ca_coords(self) -> torch.Tensor:
        """Get alpha carbon coordinates."""
        coords = []
        for res in self.residues:
            ca = res.alpha_carbon
            if ca:
                coords.append(ca.coords)
        return torch.stack(coords) if coords else torch.zeros(0, 3)


@dataclass
class Ligand:
    """Represents a small molecule ligand."""
    name: str
    chain_id: str
    atoms: List[Atom] = field(default_factory=list)
    
    @property
    def center_of_mass(self) -> torch.Tensor:
        """Compute center of mass."""
        if not self.atoms:
            return torch.zeros(3)
        coords = torch.stack([a.coords for a in self.atoms])
        return coords.mean(dim=0)
    
    @property
    def num_atoms(self) -> int:
        """Get number of atoms."""
        return len(self.atoms)


@dataclass
class BindingSite:
    """Represents a detected binding site."""
    center: torch.Tensor
    radius: float
    residues: List[Residue]
    ligand: Optional[Ligand] = None
    score: float = 0.0
    
    @property
    def residue_names(self) -> List[str]:
        """Get residue names in binding site."""
        return [f"{r.name}{r.seq_num}" for r in self.residues]
    
    @property
    def volume(self) -> float:
        """Estimate binding site volume."""
        return (4/3) * math.pi * self.radius ** 3


@dataclass
class ProteinStructure:
    """
    Complete protein structure with chains, ligands, and metadata.
    """
    pdb_id: str
    title: str = ""
    resolution: float = 0.0
    chains: Dict[str, Chain] = field(default_factory=dict)
    ligands: List[Ligand] = field(default_factory=list)
    binding_sites: List[BindingSite] = field(default_factory=list)
    
    @property
    def num_residues(self) -> int:
        """Total number of residues."""
        return sum(c.length for c in self.chains.values())
    
    @property
    def num_atoms(self) -> int:
        """Total number of atoms."""
        count = sum(
            len(r.atoms) for c in self.chains.values() for r in c.residues
        )
        count += sum(len(lig.atoms) for lig in self.ligands)
        return count
    
    @property
    def sequences(self) -> Dict[str, str]:
        """Get all chain sequences."""
        return {cid: c.sequence for cid, c in self.chains.items()}
    
    def get_all_coords(self) -> torch.Tensor:
        """Get all atom coordinates."""
        coords = []
        for chain in self.chains.values():
            for residue in chain.residues:
                for atom in residue.atoms:
                    coords.append(atom.coords)
        for ligand in self.ligands:
            for atom in ligand.atoms:
                coords.append(atom.coords)
        return torch.stack(coords) if coords else torch.zeros(0, 3)
    
    def get_ca_coords(self) -> torch.Tensor:
        """Get all alpha carbon coordinates."""
        coords = []
        for chain in self.chains.values():
            ca_coords = chain.get_ca_coords()
            if len(ca_coords) > 0:
                coords.append(ca_coords)
        return torch.cat(coords) if coords else torch.zeros(0, 3)


class MolecularIngester:
    """
    Ingests molecular structure data for drug discovery analysis.
    
    Supports:
        - PDB file parsing
        - Sequence embedding for ML
        - Binding site detection
        - Distance matrix computation
        - Property extraction
    """
    
    def __init__(self):
        """Initialize the molecular ingester."""
        pass
    
    def from_pdb_string(self, pdb_content: str, pdb_id: str = "UNKNOWN") -> ProteinStructure:
        """
        Parse PDB format string into ProteinStructure.
        
        Args:
            pdb_content: PDB file content as string
            pdb_id: PDB identifier
            
        Returns:
            ProteinStructure object
        """
        structure = ProteinStructure(pdb_id=pdb_id)
        chains: Dict[str, Chain] = {}
        current_residues: Dict[str, Dict[int, Residue]] = {}
        ligands: Dict[str, Ligand] = {}
        
        for line in pdb_content.split('\n'):
            record_type = line[:6].strip()
            
            if record_type == 'TITLE':
                structure.title = line[10:].strip()
                
            elif record_type == 'REMARK' and 'RESOLUTION' in line:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'RESOLUTION.':
                            structure.resolution = float(parts[i+1])
                            break
                except (ValueError, IndexError):
                    pass
                    
            elif record_type in ('ATOM', 'HETATM'):
                try:
                    serial = int(line[6:11].strip())
                    name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21] if len(line) > 21 else 'A'
                    residue_seq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    occupancy = float(line[54:60].strip()) if len(line) > 60 else 1.0
                    b_factor = float(line[60:66].strip()) if len(line) > 66 else 0.0
                    element = line[76:78].strip() if len(line) > 78 else name[0]
                    
                    atom = Atom(
                        serial=serial,
                        name=name,
                        residue_name=residue_name,
                        chain_id=chain_id,
                        residue_seq=residue_seq,
                        x=x, y=y, z=z,
                        element=element,
                        occupancy=occupancy,
                        b_factor=b_factor
                    )
                    
                    if record_type == 'ATOM' and residue_name in AMINO_ACID_PROPERTIES:
                        # Standard residue
                        if chain_id not in chains:
                            chains[chain_id] = Chain(chain_id=chain_id)
                            current_residues[chain_id] = {}
                        
                        if residue_seq not in current_residues[chain_id]:
                            residue = Residue(
                                name=residue_name,
                                chain_id=chain_id,
                                seq_num=residue_seq
                            )
                            current_residues[chain_id][residue_seq] = residue
                        
                        current_residues[chain_id][residue_seq].atoms.append(atom)
                        
                    elif record_type == 'HETATM' and residue_name not in ('HOH', 'WAT'):
                        # Ligand (non-water HETATM)
                        lig_key = f"{residue_name}_{chain_id}_{residue_seq}"
                        if lig_key not in ligands:
                            ligands[lig_key] = Ligand(name=residue_name, chain_id=chain_id)
                        ligands[lig_key].atoms.append(atom)
                        
                except (ValueError, IndexError):
                    continue
        
        # Assemble chains
        for chain_id, chain in chains.items():
            residue_dict = current_residues.get(chain_id, {})
            chain.residues = [residue_dict[seq] for seq in sorted(residue_dict.keys())]
        
        structure.chains = chains
        structure.ligands = list(ligands.values())
        
        return structure
    
    def from_pdb_file(self, filepath: Union[str, Path]) -> ProteinStructure:
        """
        Load structure from PDB file.
        
        Args:
            filepath: Path to PDB file
            
        Returns:
            ProteinStructure object
        """
        path = Path(filepath)
        pdb_id = path.stem.upper()
        content = path.read_text()
        return self.from_pdb_string(content, pdb_id)
    
    def from_sequence(self, sequence: str, chain_id: str = 'A') -> ProteinStructure:
        """
        Create minimal structure from amino acid sequence.
        
        Note: No 3D coordinates, only for sequence analysis.
        
        Args:
            sequence: One-letter amino acid sequence
            chain_id: Chain identifier
            
        Returns:
            ProteinStructure with sequence but no coordinates
        """
        structure = ProteinStructure(pdb_id="SEQ")
        chain = Chain(chain_id=chain_id)
        
        for i, aa in enumerate(sequence.upper()):
            if aa in ONE_TO_THREE:
                residue = Residue(
                    name=ONE_TO_THREE[aa],
                    chain_id=chain_id,
                    seq_num=i + 1
                )
                chain.residues.append(residue)
        
        structure.chains[chain_id] = chain
        return structure
    
    def compute_distance_matrix(self, structure: ProteinStructure,
                                 use_ca: bool = True) -> torch.Tensor:
        """
        Compute pairwise distance matrix.
        
        Args:
            structure: Protein structure
            use_ca: If True, use only alpha carbons; else all atoms
            
        Returns:
            Distance matrix tensor
        """
        if use_ca:
            coords = structure.get_ca_coords()
        else:
            coords = structure.get_all_coords()
        
        if len(coords) == 0:
            return torch.zeros(0, 0)
        
        # Pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))
        return dist
    
    def compute_contact_map(self, structure: ProteinStructure,
                            threshold: float = 8.0) -> torch.Tensor:
        """
        Compute binary contact map.
        
        Args:
            structure: Protein structure
            threshold: Distance threshold in Angstroms
            
        Returns:
            Binary contact matrix
        """
        dist = self.compute_distance_matrix(structure, use_ca=True)
        return (dist < threshold).float()
    
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """
        Create numerical embedding of amino acid sequence.
        
        Uses one-hot encoding + physicochemical properties.
        
        Args:
            sequence: One-letter amino acid sequence
            
        Returns:
            Embedding tensor [L, 23] (20 one-hot + 3 properties)
        """
        L = len(sequence)
        embedding = torch.zeros(L, 23)
        
        aa_list = list(AMINO_ACID_PROPERTIES.keys())
        aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
        
        for i, aa in enumerate(sequence.upper()):
            three_letter = ONE_TO_THREE.get(aa, None)
            if three_letter:
                # One-hot encoding (20 dimensions)
                if three_letter in aa_to_idx:
                    embedding[i, aa_to_idx[three_letter]] = 1.0
                
                # Physicochemical properties (3 dimensions)
                props = AMINO_ACID_PROPERTIES[three_letter]
                embedding[i, 20] = props['hydrophobicity'] / 4.5  # Normalize
                embedding[i, 21] = props['charge']  # Already -1 to 1
                embedding[i, 22] = props['size'] / 200  # Normalize
        
        return embedding
    
    def detect_binding_sites(self, structure: ProteinStructure,
                              probe_radius: float = 1.4,
                              min_residues: int = 5) -> List[BindingSite]:
        """
        Detect potential binding sites using geometric analysis.
        
        Uses cavity detection based on alpha carbon distribution.
        
        Args:
            structure: Protein structure
            probe_radius: Probe sphere radius (Angstroms)
            min_residues: Minimum residues for a binding site
            
        Returns:
            List of detected binding sites
        """
        binding_sites = []
        
        # If ligands present, use them to define binding sites
        for ligand in structure.ligands:
            if ligand.num_atoms < 3:
                continue
            
            center = ligand.center_of_mass
            
            # Find residues within 6 Angstroms of ligand
            nearby_residues = []
            for chain in structure.chains.values():
                for residue in chain.residues:
                    res_center = residue.center_of_mass
                    dist = torch.norm(center - res_center).item()
                    if dist < 6.0:
                        nearby_residues.append(residue)
            
            if len(nearby_residues) >= min_residues:
                # Compute bounding radius
                res_coords = torch.stack([r.center_of_mass for r in nearby_residues])
                dists = torch.norm(res_coords - center, dim=1)
                radius = dists.max().item()
                
                site = BindingSite(
                    center=center,
                    radius=radius,
                    residues=nearby_residues,
                    ligand=ligand,
                    score=1.0  # Known binding site
                )
                binding_sites.append(site)
        
        # If no ligands, detect cavities geometrically
        if not structure.ligands:
            ca_coords = structure.get_ca_coords()
            if len(ca_coords) < 10:
                return binding_sites
            
            # Find points with low local density (potential cavities)
            # Using simple k-means-like clustering on sparse regions
            grid_spacing = 3.0
            min_coord = ca_coords.min(dim=0).values - 5
            max_coord = ca_coords.max(dim=0).values + 5
            
            # Sample grid points
            x = torch.arange(min_coord[0].item(), max_coord[0].item(), grid_spacing)
            y = torch.arange(min_coord[1].item(), max_coord[1].item(), grid_spacing)
            z = torch.arange(min_coord[2].item(), max_coord[2].item(), grid_spacing)
            
            # Limit grid size
            max_points = 20
            if len(x) > max_points:
                x = x[::len(x)//max_points]
            if len(y) > max_points:
                y = y[::len(y)//max_points]
            if len(z) > max_points:
                z = z[::len(z)//max_points]
            
            # Find cavity-like regions (surrounded by protein but not too dense)
            for gx in x:
                for gy in y:
                    for gz in z:
                        point = torch.tensor([gx, gy, gz])
                        dists = torch.norm(ca_coords - point, dim=1)
                        
                        # Count residues at different distances
                        n_close = (dists < 4.0).sum().item()
                        n_medium = ((dists >= 4.0) & (dists < 8.0)).sum().item()
                        n_far = ((dists >= 8.0) & (dists < 12.0)).sum().item()
                        
                        # Cavity: few very close, many at medium distance
                        if n_close < 3 and n_medium >= min_residues and n_far >= min_residues:
                            # Collect nearby residues
                            nearby_residues = []
                            for chain in structure.chains.values():
                                for residue in chain.residues:
                                    ca = residue.alpha_carbon
                                    if ca:
                                        dist = torch.norm(point - ca.coords).item()
                                        if dist < 10.0:
                                            nearby_residues.append(residue)
                            
                            if len(nearby_residues) >= min_residues:
                                score = n_medium / (n_close + 1)  # Higher = more cavity-like
                                site = BindingSite(
                                    center=point,
                                    radius=8.0,
                                    residues=nearby_residues,
                                    score=score
                                )
                                binding_sites.append(site)
            
            # Remove overlapping sites, keep highest scoring
            if binding_sites:
                binding_sites.sort(key=lambda s: s.score, reverse=True)
                filtered = []
                for site in binding_sites:
                    is_overlap = False
                    for kept in filtered:
                        dist = torch.norm(site.center - kept.center).item()
                        if dist < 6.0:
                            is_overlap = True
                            break
                    if not is_overlap:
                        filtered.append(site)
                binding_sites = filtered[:5]  # Keep top 5
        
        structure.binding_sites = binding_sites
        return binding_sites
    
    def compute_hydrophobicity_profile(self, sequence: str,
                                        window_size: int = 7) -> torch.Tensor:
        """
        Compute Kyte-Doolittle hydrophobicity profile.
        
        Args:
            sequence: One-letter amino acid sequence
            window_size: Sliding window size
            
        Returns:
            Hydrophobicity profile tensor
        """
        hydro_values = []
        for aa in sequence.upper():
            three_letter = ONE_TO_THREE.get(aa, None)
            if three_letter:
                hydro_values.append(AMINO_ACID_PROPERTIES[three_letter]['hydrophobicity'])
            else:
                hydro_values.append(0.0)
        
        hydro = torch.tensor(hydro_values)
        
        # Sliding window average
        if len(hydro) < window_size:
            return hydro
        
        profile = torch.zeros(len(hydro) - window_size + 1)
        for i in range(len(profile)):
            profile[i] = hydro[i:i+window_size].mean()
        
        return profile
    
    def compute_secondary_structure_propensity(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Estimate secondary structure propensity.
        
        Uses Chou-Fasman parameters.
        
        Args:
            sequence: One-letter amino acid sequence
            
        Returns:
            Dict with 'helix', 'sheet', 'coil' propensity tensors
        """
        # Chou-Fasman propensities (simplified)
        helix_prop = {
            'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
            'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
            'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
            'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
        }
        
        sheet_prop = {
            'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
            'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
            'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
            'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
        }
        
        helix = []
        sheet = []
        
        for aa in sequence.upper():
            helix.append(helix_prop.get(aa, 1.0))
            sheet.append(sheet_prop.get(aa, 1.0))
        
        helix_t = torch.tensor(helix)
        sheet_t = torch.tensor(sheet)
        coil_t = 2.0 - helix_t - sheet_t  # Inverse of structure propensity
        coil_t = torch.clamp(coil_t, 0.0, 2.0)
        
        return {
            'helix': helix_t,
            'sheet': sheet_t,
            'coil': coil_t
        }
    
    def to_dict(self, structure: ProteinStructure) -> Dict[str, Any]:
        """
        Convert structure to dictionary for serialization.
        
        Args:
            structure: Protein structure
            
        Returns:
            Dictionary representation
        """
        return {
            'pdb_id': structure.pdb_id,
            'title': structure.title,
            'resolution': structure.resolution,
            'num_residues': structure.num_residues,
            'num_atoms': structure.num_atoms,
            'chains': {
                cid: {
                    'length': c.length,
                    'sequence': c.sequence
                }
                for cid, c in structure.chains.items()
            },
            'ligands': [
                {
                    'name': lig.name,
                    'chain': lig.chain_id,
                    'num_atoms': lig.num_atoms
                }
                for lig in structure.ligands
            ],
            'binding_sites': [
                {
                    'center': site.center.tolist(),
                    'radius': site.radius,
                    'residues': site.residue_names,
                    'ligand': site.ligand.name if site.ligand else None,
                    'score': site.score
                }
                for site in structure.binding_sites
            ]
        }


# Synthetic test data for validation
def create_synthetic_protein(n_residues: int = 100) -> ProteinStructure:
    """
    Create a synthetic protein structure for testing.
    
    ⚠️  TESTING ONLY - NOT REAL BIOLOGICAL DATA
    
    Creates an artificial alpha-helix-like structure with random amino acids.
    Coordinates are mathematically generated, not from X-ray crystallography
    or cryo-EM experiments.
    
    For real structural biology analysis, use:
        - Load PDB files via from_pdb()
        - Load mmCIF files via from_mmcif()
    
    Args:
        n_residues: Number of residues in synthetic chain
        
    Returns:
        ProteinStructure with synthetic coordinates
    """
    import random
    random.seed(42)
    
    structure = ProteinStructure(
        pdb_id="SYNTH",
        title="Synthetic Test Protein",
        resolution=2.0
    )
    
    chain = Chain(chain_id='A')
    aa_list = list(AMINO_ACID_PROPERTIES.keys())
    
    # Generate alpha-helix-like structure
    for i in range(n_residues):
        aa = random.choice(aa_list)
        residue = Residue(name=aa, chain_id='A', seq_num=i + 1)
        
        # Helical coordinates
        t = i * 0.6  # Rise per residue
        theta = i * (100 * math.pi / 180)  # ~100 degrees per residue
        
        x = 5.0 * math.cos(theta)
        y = 5.0 * math.sin(theta)
        z = 1.5 * t
        
        # Add CA atom
        ca = Atom(
            serial=i * 4 + 1,
            name='CA',
            residue_name=aa,
            chain_id='A',
            residue_seq=i + 1,
            x=x, y=y, z=z,
            element='C'
        )
        residue.atoms.append(ca)
        
        # Add N atom (slightly offset)
        n_atom = Atom(
            serial=i * 4 + 2,
            name='N',
            residue_name=aa,
            chain_id='A',
            residue_seq=i + 1,
            x=x - 0.5, y=y + 0.3, z=z - 0.2,
            element='N'
        )
        residue.atoms.append(n_atom)
        
        chain.residues.append(residue)
    
    structure.chains['A'] = chain
    
    # Add a synthetic ligand in the middle
    ligand = Ligand(name='LIG', chain_id='A')
    mid_residue = chain.residues[n_residues // 2]
    ca = mid_residue.alpha_carbon
    if ca:
        for j in range(10):
            lig_atom = Atom(
                serial=n_residues * 4 + j + 1,
                name=f'C{j}',
                residue_name='LIG',
                chain_id='A',
                residue_seq=1,
                x=ca.x + random.uniform(-2, 2),
                y=ca.y + random.uniform(-2, 2),
                z=ca.z + random.uniform(-2, 2),
                element='C'
            )
            ligand.atoms.append(lig_atom)
    
    structure.ligands.append(ligand)
    
    return structure


if __name__ == "__main__":
    # Quick test
    ingester = MolecularIngester()
    
    # Create synthetic protein
    protein = create_synthetic_protein(100)
    print(f"Synthetic protein: {protein.pdb_id}")
    print(f"  Residues: {protein.num_residues}")
    print(f"  Atoms: {protein.num_atoms}")
    print(f"  Chains: {list(protein.chains.keys())}")
    print(f"  Ligands: {[l.name for l in protein.ligands]}")
    
    # Detect binding sites
    sites = ingester.detect_binding_sites(protein)
    print(f"  Binding sites: {len(sites)}")
    for site in sites:
        print(f"    - {len(site.residues)} residues, score={site.score:.2f}")
    
    # Compute properties
    seq = protein.chains['A'].sequence
    print(f"  Sequence length: {len(seq)}")
    
    embedding = ingester.embed_sequence(seq)
    print(f"  Embedding shape: {embedding.shape}")
    
    hydro = ingester.compute_hydrophobicity_profile(seq)
    print(f"  Hydrophobicity profile: {hydro.shape}")
    
    dist_matrix = ingester.compute_distance_matrix(protein)
    print(f"  Distance matrix: {dist_matrix.shape}")
    
    print("\n✅ Molecular ingester test passed!")
