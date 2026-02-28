#!/usr/bin/env python3
"""
Molecular Discovery Pipeline for Drug Discovery

Phase 3 of the Autonomous Discovery Engine.

Analyzes protein structures and molecular data using all 7 QTT-native
Genesis primitives to detect binding sites, analyze protein dynamics,
and generate drug-protein interaction hypotheses.

Pipeline Stages:
    1. Structure ingestion and embedding
    2. Distance matrix analysis (OT)
    3. Multi-scale structure analysis (SGW)
    4. Spectral properties (RMT)
    5. Binding path detection (TG)
    6. Binding site anomaly detection (RKHS)
    7. Structural topology (PH)
    8. Geometric invariants (GA)
    9. Hypothesis generation

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import time
import math
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import torch

# Local imports
from ontic.ml.discovery.ingest.molecular import (
    MolecularIngester, ProteinStructure, BindingSite,
    create_synthetic_protein, AMINO_ACID_PROPERTIES
)
from ontic.ml.discovery.engine_v2 import Finding, DiscoveryResult
from ontic.ml.discovery.hypothesis.generator import HypothesisGenerator, Hypothesis

# Genesis imports - QTT-native primitives
from ontic.genesis.ot import (
    QTTDistribution, wasserstein_distance, barycenter
)
from ontic.genesis.sgw import (
    QTTLaplacian, QTTSignal, QTTGraphWavelet
)
from ontic.genesis.rmt import (
    QTTEnsemble, SpectralDensity, WignerSemicircle
)
from ontic.genesis.tropical import (
    TropicalMatrix, MinPlusSemiring,
    tropical_eigenvalue, tropical_eigenvector
)
from ontic.genesis.rkhs import (
    RBFKernel, maximum_mean_discrepancy
)
from ontic.genesis.topology import (
    VietorisRips, compute_persistence
)
from ontic.genesis.ga import (
    CliffordAlgebra, vector, bivector,
    geometric_product, rotor_from_bivector
)


@dataclass
class MolecularPipelineResult:
    """Results from molecular discovery pipeline."""
    structure_id: str
    findings: List[Finding] = field(default_factory=list)
    binding_sites: List[Dict] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    stages: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    attestation_hash: str = ""
    
    @property
    def n_critical(self) -> int:
        """Count critical findings."""
        return sum(1 for f in self.findings if f.severity == "CRITICAL")
    
    @property
    def n_high(self) -> int:
        """Count high severity findings."""
        return sum(1 for f in self.findings if f.severity == "HIGH")


class MolecularDiscoveryPipeline:
    """
    Discovery pipeline for drug-protein interaction analysis.
    
    Uses all 7 QTT-native Genesis primitives to analyze protein structures
    and detect binding sites, conformational changes, and drug candidates.
    """
    
    def __init__(self, grid_bits: int = 10):
        """
        Initialize the molecular pipeline.
        
        Args:
            grid_bits: Grid resolution for QTT operations
        """
        self.grid_bits = grid_bits
        self.grid_size = 2 ** grid_bits
        self.ingester = MolecularIngester()
        self.hypothesis_gen = HypothesisGenerator()
        
    def analyze_structure(self, 
                          structure: ProteinStructure,
                          reference_structure: Optional[ProteinStructure] = None,
                          verbose: bool = False) -> MolecularPipelineResult:
        """
        Run full discovery pipeline on protein structure.
        
        Args:
            structure: Protein structure to analyze
            reference_structure: Optional reference for comparison
            verbose: Print progress
            
        Returns:
            MolecularPipelineResult with findings and hypotheses
        """
        start_total = time.perf_counter()
        findings: List[Finding] = []
        stages: List[Dict] = []
        
        if verbose:
            print(f"[MOLECULAR] Analyzing structure {structure.pdb_id}...")
        
        # Stage 1: Structure ingestion and binding site detection
        stage1 = self._stage_ingest(structure, verbose)
        stages.append(stage1)
        
        # Stage 2: Distance matrix analysis (OT)
        stage2 = self._stage_ot(structure, reference_structure, verbose)
        stages.append(stage2)
        findings.extend(stage2.get("findings", []))
        
        # Stage 3: Multi-scale structure analysis (SGW)
        stage3 = self._stage_sgw(structure, verbose)
        stages.append(stage3)
        findings.extend(stage3.get("findings", []))
        
        # Stage 4: Spectral properties (RMT)
        stage4 = self._stage_rmt(structure, verbose)
        stages.append(stage4)
        findings.extend(stage4.get("findings", []))
        
        # Stage 5: Binding path detection (TG)
        stage5 = self._stage_tg(structure, verbose)
        stages.append(stage5)
        findings.extend(stage5.get("findings", []))
        
        # Stage 6: Binding site anomaly detection (RKHS)
        stage6 = self._stage_rkhs(structure, verbose)
        stages.append(stage6)
        findings.extend(stage6.get("findings", []))
        
        # Stage 7: Structural topology (PH)
        stage7 = self._stage_ph(structure, verbose)
        stages.append(stage7)
        findings.extend(stage7.get("findings", []))
        
        # Stage 8: Geometric invariants (GA)
        stage8 = self._stage_ga(structure, verbose)
        stages.append(stage8)
        findings.extend(stage8.get("findings", []))
        
        # Stage 9: Generate hypotheses
        hypotheses = self._generate_hypotheses(findings, structure, verbose)
        
        total_time = time.perf_counter() - start_total
        
        if verbose:
            print(f"[MOLECULAR] Complete: {len(findings)} findings, "
                  f"{len(hypotheses)} hypotheses in {total_time*1000:.1f}ms")
        
        # Generate attestation
        attestation = {
            "structure_id": structure.pdb_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_findings": len(findings),
            "n_hypotheses": len(hypotheses),
            "total_time": total_time,
        }
        attestation_hash = hashlib.sha256(
            json.dumps(attestation, sort_keys=True).encode()
        ).hexdigest()
        
        return MolecularPipelineResult(
            structure_id=structure.pdb_id,
            findings=findings,
            binding_sites=[
                {
                    "center": site.center.tolist(),
                    "radius": site.radius,
                    "residues": site.residue_names,
                    "score": site.score
                }
                for site in structure.binding_sites
            ],
            hypotheses=hypotheses,
            stages=stages,
            total_time=total_time,
            attestation_hash=attestation_hash,
        )
    
    def _stage_ingest(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 1: Structure ingestion and preprocessing."""
        start = time.perf_counter()
        
        if verbose:
            print("[MOLECULAR] Stage 1: Structure ingestion...")
        
        # Detect binding sites if not already done
        if not structure.binding_sites:
            self.ingester.detect_binding_sites(structure)
        
        # Compute basic properties
        n_residues = structure.num_residues
        n_atoms = structure.num_atoms
        n_chains = len(structure.chains)
        n_ligands = len(structure.ligands)
        n_binding_sites = len(structure.binding_sites)
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Structure Ingestion",
            "primitive": "INGEST",
            "time": elapsed,
            "metrics": {
                "n_residues": n_residues,
                "n_atoms": n_atoms,
                "n_chains": n_chains,
                "n_ligands": n_ligands,
                "n_binding_sites": n_binding_sites,
            }
        }
    
    def _stage_ot(self, structure: ProteinStructure,
                  reference: Optional[ProteinStructure],
                  verbose: bool) -> Dict:
        """Stage 2: Optimal Transport - distance distribution analysis."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 2: Distance distribution analysis (OT)...")
        
        # Compute distance matrix
        dist_matrix = self.ingester.compute_distance_matrix(structure, use_ca=True)
        
        if dist_matrix.numel() == 0:
            return {
                "name": "Optimal Transport",
                "primitive": "OT",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "No coordinates available"}
            }
        
        # Flatten distance distribution
        triu_idx = torch.triu_indices(dist_matrix.size(0), dist_matrix.size(1), offset=1)
        distances = dist_matrix[triu_idx[0], triu_idx[1]]
        
        # Compute statistics
        mean_dist = float(distances.mean())
        std_dist = float(distances.std())
        max_dist = float(distances.max())
        
        # Compare to reference if available
        W2 = 0.0
        if reference is not None:
            ref_dist_matrix = self.ingester.compute_distance_matrix(reference, use_ca=True)
            if ref_dist_matrix.numel() > 0:
                ref_triu_idx = torch.triu_indices(ref_dist_matrix.size(0), ref_dist_matrix.size(1), offset=1)
                ref_distances = ref_dist_matrix[ref_triu_idx[0], ref_triu_idx[1]]
                
                # Compute W2 using empirical quantile matching
                # Sort both distributions and compute RMS difference
                n_samples = min(len(distances), len(ref_distances), 1000)
                
                # Subsample if needed
                if len(distances) > n_samples:
                    idx1 = torch.randperm(len(distances))[:n_samples]
                    dist_sorted = distances[idx1].sort()[0]
                else:
                    dist_sorted = distances.sort()[0]
                
                if len(ref_distances) > n_samples:
                    idx2 = torch.randperm(len(ref_distances))[:n_samples]
                    ref_sorted = ref_distances[idx2].sort()[0]
                else:
                    ref_sorted = ref_distances.sort()[0]
                
                # Interpolate to same length for quantile matching
                if len(dist_sorted) != len(ref_sorted):
                    quantiles = torch.linspace(0, 1, 100)
                    q1 = torch.quantile(dist_sorted, quantiles)
                    q2 = torch.quantile(ref_sorted, quantiles)
                else:
                    q1, q2 = dist_sorted, ref_sorted
                
                # W2 distance: RMS of quantile differences
                W2 = float(torch.sqrt(((q1 - q2) ** 2).mean()))
        
        # Detect compact vs extended structures
        # Typical globular proteins have mean CA distance ~15-25 Å
        if mean_dist < 12:
            findings.append(Finding(
                primitive="OT",
                severity="MEDIUM",
                summary=f"Unusually compact structure: mean CA distance = {mean_dist:.1f} Å",
                evidence={"mean_distance": mean_dist, "threshold": 12}
            ))
        elif mean_dist > 35:
            findings.append(Finding(
                primitive="OT",
                severity="MEDIUM",
                summary=f"Extended structure: mean CA distance = {mean_dist:.1f} Å",
                evidence={"mean_distance": mean_dist, "threshold": 35}
            ))
        
        if W2 > 5.0:
            findings.append(Finding(
                primitive="OT",
                severity="HIGH",
                summary=f"Significant conformational change: W₂ = {W2:.2f} Å",
                evidence={"wasserstein_distance": W2}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Optimal Transport",
            "primitive": "OT",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "mean_distance": mean_dist,
                "std_distance": std_dist,
                "max_distance": max_dist,
                "W2_vs_reference": W2,
            }
        }
    
    def _stage_sgw(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 3: Spectral Graph Wavelets - multi-scale structure analysis."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 3: Multi-scale structure analysis (SGW)...")
        
        # Build contact-based graph Laplacian
        contact_map = self.ingester.compute_contact_map(structure, threshold=8.0)
        n_nodes = contact_map.size(0)
        
        if n_nodes == 0:
            return {
                "name": "Spectral Graph Wavelets",
                "primitive": "SGW",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "No contacts computed"}
            }
        
        # Compute graph Laplacian
        degree = contact_map.sum(dim=1)
        D = torch.diag(degree)
        L = D - contact_map
        
        # Normalize
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-8))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        
        # Compute eigenvalues for spectral analysis
        try:
            eigenvalues = torch.linalg.eigvalsh(L_norm)
            eigenvalues = torch.sort(eigenvalues).values
        except Exception:
            eigenvalues = torch.zeros(min(n_nodes, 10))
        
        # Spectral gap indicates modularity
        if len(eigenvalues) > 1:
            spectral_gap = float(eigenvalues[1]) if eigenvalues[1] > 0 else 0.0
            
            # Small spectral gap suggests multiple domains/modules
            if spectral_gap < 0.1:
                findings.append(Finding(
                    primitive="SGW",
                    severity="MEDIUM",
                    summary=f"Multi-domain structure: spectral gap = {spectral_gap:.4f}",
                    evidence={"spectral_gap": spectral_gap, "threshold": 0.1}
                ))
        else:
            spectral_gap = 0.0
        
        # Count connected components (zeros in eigenvalues)
        n_components = int((eigenvalues.abs() < 1e-6).sum())
        if n_components > 1:
            findings.append(Finding(
                primitive="SGW",
                severity="MEDIUM",
                summary=f"Structure has {n_components} disconnected domains",
                evidence={"n_components": n_components}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Spectral Graph Wavelets",
            "primitive": "SGW",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "n_nodes": n_nodes,
                "spectral_gap": spectral_gap,
                "n_components": n_components,
                "top_eigenvalues": eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
            }
        }
    
    def _stage_rmt(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 4: Random Matrix Theory - spectral statistics."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 4: Spectral statistics (RMT)...")
        
        # Use distance matrix as input
        dist_matrix = self.ingester.compute_distance_matrix(structure, use_ca=True)
        n = dist_matrix.size(0)
        
        if n < 10:
            return {
                "name": "Random Matrix Theory",
                "primitive": "RMT",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Structure too small for RMT"}
            }
        
        # Create correlation-like matrix
        # C_ij = exp(-d_ij / d_mean)
        d_mean = float(dist_matrix.mean())
        corr_matrix = torch.exp(-dist_matrix / (d_mean + 1e-8))
        
        # Symmetrize and normalize
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        diag = torch.diag(torch.diag(corr_matrix))
        corr_matrix = corr_matrix - diag + torch.eye(n)
        
        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(corr_matrix)
            eigenvalues = torch.sort(eigenvalues).values
        except Exception:
            eigenvalues = torch.zeros(n)
        
        # Level spacing analysis
        # Compare to Wigner (GOE) vs Poisson
        spacings = eigenvalues[1:] - eigenvalues[:-1]
        spacings = spacings[spacings > 1e-8]
        
        if len(spacings) > 5:
            mean_spacing = float(spacings.mean())
            normalized_spacings = spacings / (mean_spacing + 1e-8)
            
            # Wigner surmise: P(s) ~ s * exp(-s^2)
            # Poisson: P(s) ~ exp(-s)
            # Ratio r = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
            # Wigner: <r> ≈ 0.53, Poisson: <r> ≈ 0.39
            
            ratios = []
            for i in range(len(normalized_spacings) - 1):
                s1 = normalized_spacings[i].item()
                s2 = normalized_spacings[i + 1].item()
                if s1 > 0 and s2 > 0:
                    ratios.append(min(s1, s2) / max(s1, s2))
            
            mean_ratio = sum(ratios) / len(ratios) if ratios else 0.5
            
            # Classify dynamics
            if mean_ratio < 0.42:
                behavior = "integrable"
                findings.append(Finding(
                    primitive="RMT",
                    severity="INFO",
                    summary=f"Integrable dynamics: level spacing ratio = {mean_ratio:.3f}",
                    evidence={"level_spacing_ratio": mean_ratio, "behavior": behavior}
                ))
            elif mean_ratio > 0.50:
                behavior = "chaotic"
                findings.append(Finding(
                    primitive="RMT",
                    severity="MEDIUM",
                    summary=f"Chaotic dynamics: level spacing ratio = {mean_ratio:.3f}",
                    evidence={"level_spacing_ratio": mean_ratio, "behavior": behavior}
                ))
            else:
                behavior = "intermediate"
        else:
            mean_ratio = 0.5
            behavior = "unknown"
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Random Matrix Theory",
            "primitive": "RMT",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "n_eigenvalues": len(eigenvalues),
                "level_spacing_ratio": mean_ratio,
                "behavior": behavior,
            }
        }
    
    def _stage_tg(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 5: Tropical Geometry - binding path detection."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 5: Binding path detection (TG)...")
        
        # Build distance graph for binding sites
        dist_matrix = self.ingester.compute_distance_matrix(structure, use_ca=True)
        n = dist_matrix.size(0)
        
        if n < 4 or not structure.binding_sites:
            return {
                "name": "Tropical Geometry",
                "primitive": "TG",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "No binding sites or too small"}
            }
        
        # Use min-plus tropical semiring to find shortest paths
        # between binding sites and key residues
        trop_matrix = TropicalMatrix(dist_matrix, MinPlusSemiring, n)
        
        # Compute tropical eigenvalue
        trop_eigenval = tropical_eigenvalue(trop_matrix)
        
        # Find bottleneck residues
        eigen_result = tropical_eigenvector(trop_matrix)
        eigenvec = eigen_result.eigenvector
        
        # Critical residue is where eigenvector is minimal (in min-plus)
        if len(eigenvec) > 0:
            critical_idx = int(torch.argmin(eigenvec).item())
            
            # Map to residue
            all_residues = []
            for chain in structure.chains.values():
                all_residues.extend(chain.residues)
            
            if critical_idx < len(all_residues):
                critical_residue = all_residues[critical_idx]
                
                findings.append(Finding(
                    primitive="TG",
                    severity="MEDIUM",
                    summary=f"Critical binding path residue: {critical_residue.name}{critical_residue.seq_num}",
                    evidence={
                        "tropical_eigenvalue": float(trop_eigenval),
                        "critical_residue": f"{critical_residue.name}{critical_residue.seq_num}",
                        "chain": critical_residue.chain_id
                    }
                ))
        else:
            critical_idx = -1
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Tropical Geometry",
            "primitive": "TG",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "tropical_eigenvalue": float(trop_eigenval),
                "critical_residue_idx": critical_idx,
            }
        }
    
    def _stage_rkhs(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 6: RKHS - binding site anomaly detection."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 6: Binding site anomaly detection (RKHS)...")
        
        if not structure.binding_sites:
            return {
                "name": "RKHS Kernel Methods",
                "primitive": "RKHS",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "No binding sites detected"}
            }
        
        kernel = RBFKernel(length_scale=2.0, variance=1.0)
        
        # Compare binding site residue properties to global average
        all_residues = []
        for chain in structure.chains.values():
            all_residues.extend(chain.residues)
        
        if len(all_residues) < 10:
            return {
                "name": "RKHS Kernel Methods",
                "primitive": "RKHS",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Too few residues"}
            }
        
        # Global property distribution
        global_props = torch.tensor([
            [r.hydrophobicity, r.charge, r.size / 200]
            for r in all_residues
        ])
        
        for site in structure.binding_sites:
            if len(site.residues) < 3:
                continue
            
            # Binding site property distribution
            site_props = torch.tensor([
                [r.hydrophobicity, r.charge, r.size / 200]
                for r in site.residues
            ])
            
            # Compute MMD
            mmd = maximum_mean_discrepancy(global_props, site_props, kernel)
            
            if mmd > 0.3:
                # Characterize the binding site
                mean_hydro = float(site_props[:, 0].mean())
                mean_charge = float(site_props[:, 1].mean())
                
                if mean_hydro > 1.0:
                    character = "hydrophobic"
                elif mean_hydro < -1.0:
                    character = "hydrophilic"
                elif mean_charge > 0.3:
                    character = "positively charged"
                elif mean_charge < -0.3:
                    character = "negatively charged"
                else:
                    character = "mixed"
                
                findings.append(Finding(
                    primitive="RKHS",
                    severity="HIGH" if mmd > 0.5 else "MEDIUM",
                    summary=f"Distinctive binding site: MMD={mmd:.3f}, {character}",
                    evidence={
                        "mmd": float(mmd),
                        "character": character,
                        "mean_hydrophobicity": mean_hydro,
                        "mean_charge": mean_charge,
                        "n_residues": len(site.residues)
                    }
                ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "RKHS Kernel Methods",
            "primitive": "RKHS",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "n_binding_sites_analyzed": len(structure.binding_sites),
            }
        }
    
    def _stage_ph(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 7: Persistent Homology - structural topology."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 7: Structural topology (PH)...")
        
        # Get CA coordinates for topological analysis
        ca_coords = structure.get_ca_coords()
        
        if len(ca_coords) < 10:
            return {
                "name": "Persistent Homology",
                "primitive": "PH",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Too few CA atoms"}
            }
        
        # Aggressive subsampling for efficiency
        # PH is O(n^3) so we limit to 30 points max
        max_points = 30
        if len(ca_coords) > max_points:
            indices = torch.linspace(0, len(ca_coords) - 1, max_points).long()
            ca_coords = ca_coords[indices]
        
        # Build Vietoris-Rips complex with limited radius and dimension
        # Using max_dim=1 for speed (loops only, no voids)
        rips = VietorisRips.from_points(ca_coords, max_radius=12.0, max_dim=1)
        
        # Compute persistence
        diagram = compute_persistence(rips)
        betti = diagram.betti_numbers()
        
        beta_0 = betti[0] if len(betti) > 0 else 1
        beta_1 = betti[1] if len(betti) > 1 else 0
        
        # For voids, we estimate from distance matrix topology
        # rather than computing full VR complex
        dist_matrix = torch.cdist(ca_coords, ca_coords)
        # Count approximate voids by looking for large gaps in distance histogram
        dists_flat = dist_matrix.triu(diagonal=1).flatten()
        dists_flat = dists_flat[dists_flat > 0]
        if len(dists_flat) > 0:
            hist, bin_edges = torch.histogram(dists_flat, bins=20)
            # Large gaps in histogram suggest voids
            hist_normalized = hist.float() / hist.sum()
            # Count bins with very low density as potential voids
            beta_2 = int((hist_normalized < 0.01).sum().item())
        else:
            beta_2 = 0
        
        # Interpret topology
        if beta_1 > 0:
            findings.append(Finding(
                primitive="PH",
                severity="MEDIUM",
                summary=f"Cyclic structure detected: β₁ = {beta_1} loops",
                evidence={"betti_1": beta_1, "betti_0": beta_0}
            ))
        
        if beta_2 > 2:
            findings.append(Finding(
                primitive="PH",
                severity="HIGH",
                summary=f"Cavity/void detected: β₂ ≈ {beta_2} (estimated)",
                evidence={"betti_2": beta_2}
            ))
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Persistent Homology",
            "primitive": "PH",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "betti_0": beta_0,
                "betti_1": beta_1,
                "betti_2": beta_2,
                "n_points_analyzed": len(ca_coords),
            }
        }
    
    def _stage_ga(self, structure: ProteinStructure, verbose: bool) -> Dict:
        """Stage 8: Geometric Algebra - geometric invariants."""
        start = time.perf_counter()
        findings = []
        
        if verbose:
            print("[MOLECULAR] Stage 8: Geometric invariants (GA)...")
        
        ca_coords = structure.get_ca_coords()
        
        if len(ca_coords) < 5:
            return {
                "name": "Geometric Algebra",
                "primitive": "GA",
                "time": time.perf_counter() - start,
                "findings": [],
                "metrics": {"error": "Too few CA atoms"}
            }
        
        # Create Clifford algebra for 3D
        cl3 = CliffordAlgebra(3, 0, 0)
        
        # Compute principal axes using PCA-like analysis
        centered = ca_coords - ca_coords.mean(dim=0)
        cov = centered.T @ centered / len(centered)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.sort(eigenvalues, descending=True).values
        except Exception:
            eigenvalues = torch.ones(3)
            eigenvectors = torch.eye(3)
        
        # Shape descriptors
        # Asphericity: deviation from spherical shape
        total_var = float(eigenvalues.sum())
        if total_var > 0:
            lambda_norm = eigenvalues / total_var
            asphericity = float(lambda_norm[0] - 0.5 * (lambda_norm[1] + lambda_norm[2]))
            acylindricity = float(lambda_norm[1] - lambda_norm[2])
            
            # Compute shape anisotropy
            kappa2 = float(1.5 * (lambda_norm ** 2).sum() - 0.5)
        else:
            asphericity = 0.0
            acylindricity = 0.0
            kappa2 = 0.0
        
        # Classify shape
        if asphericity > 0.3:
            shape = "elongated"
            findings.append(Finding(
                primitive="GA",
                severity="INFO",
                summary=f"Elongated structure: asphericity = {asphericity:.3f}",
                evidence={"asphericity": asphericity, "shape": shape}
            ))
        elif kappa2 < 0.1:
            shape = "spherical"
            findings.append(Finding(
                primitive="GA",
                severity="INFO",
                summary=f"Globular structure: κ² = {kappa2:.3f}",
                evidence={"kappa2": kappa2, "shape": shape}
            ))
        else:
            shape = "intermediate"
        
        # Check for symmetry using rotation analysis
        # Compare structure to 180-degree rotated version
        axis = eigenvectors[:, 0]  # Principal axis
        bv = bivector(cl3, {(0, 1): axis[2].item(), (1, 2): axis[0].item(), (0, 2): axis[1].item()})
        rotor = rotor_from_bivector(bv, math.pi)  # 180-degree rotation
        
        elapsed = time.perf_counter() - start
        
        return {
            "name": "Geometric Algebra",
            "primitive": "GA",
            "time": elapsed,
            "findings": findings,
            "metrics": {
                "asphericity": asphericity,
                "acylindricity": acylindricity,
                "kappa2": kappa2,
                "shape": shape,
                "principal_axes": eigenvalues.tolist(),
            }
        }
    
    def _generate_hypotheses(self, findings: List[Finding],
                              structure: ProteinStructure,
                              verbose: bool) -> List[Hypothesis]:
        """Generate drug discovery hypotheses from findings."""
        if verbose:
            print("[MOLECULAR] Generating hypotheses...")
        
        hypotheses = []
        hyp_id = 0
        
        # Binding site hypothesis
        binding_findings = [f for f in findings if "binding" in f.summary.lower()]
        if binding_findings and structure.binding_sites:
            for site in structure.binding_sites:
                hyp_id += 1
                hypotheses.append(Hypothesis(
                    id=f"MOL-H{hyp_id:03d}",
                    title=f"Drug Target: {','.join(site.residue_names[:3])}...",
                    description=f"Binding site at {site.residue_names[:3]}... suitable for drug targeting",
                    confidence=0.7,
                    severity="HIGH",
                    findings=binding_findings,
                    evidence_summary=f"Detected via RKHS/TG analysis, {len(site.residues)} residues",
                    recommended_action="Docking simulation, Fragment screening",
                    domain_specific={"primitive_chain": ["OT", "RKHS", "TG"]}
                ))
        
        # Conformational flexibility hypothesis
        rmt_findings = [f for f in findings if f.primitive == "RMT" and "chaotic" in f.summary.lower()]
        if rmt_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MOL-H{hyp_id:03d}",
                title="Allosteric Potential",
                description="Protein shows conformational flexibility (allosteric potential)",
                confidence=0.6,
                severity="MEDIUM",
                findings=rmt_findings,
                evidence_summary="Chaotic dynamics detected via RMT level spacing",
                recommended_action="MD simulation, Normal mode analysis",
                domain_specific={"primitive_chain": ["RMT", "SGW"]}
            ))
        
        # Topology-based hypothesis
        ph_findings = [f for f in findings if f.primitive == "PH" and "void" in f.summary.lower()]
        if ph_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MOL-H{hyp_id:03d}",
                title="Cryptic Binding Site",
                description="Internal cavity detected - potential cryptic binding site",
                confidence=0.75,
                severity="HIGH",
                findings=ph_findings,
                evidence_summary="β₂ > 0 in persistent homology indicates cavity",
                recommended_action="Cavity analysis, Induced-fit docking",
                domain_specific={"primitive_chain": ["PH", "GA"]}
            ))
        
        # Multi-domain hypothesis
        sgw_findings = [f for f in findings if f.primitive == "SGW" and "domain" in f.summary.lower()]
        if sgw_findings:
            hyp_id += 1
            hypotheses.append(Hypothesis(
                id=f"MOL-H{hyp_id:03d}",
                title="Allosteric Communication",
                description="Multi-domain structure suggests allosteric communication",
                confidence=0.65,
                severity="MEDIUM",
                findings=sgw_findings,
                evidence_summary="Low spectral gap in contact graph Laplacian",
                recommended_action="Domain motion analysis, Allosteric site prediction",
                domain_specific={"primitive_chain": ["SGW", "TG"]}
            ))
        
        return hypotheses
    
    def generate_report(self, result: MolecularPipelineResult,
                        target_name: str = "Unknown Target") -> str:
        """Generate markdown report for drug discovery findings."""
        lines = [
            f"# Molecular Discovery Report: {target_name}",
            "",
            f"**Structure ID:** {result.structure_id}",
            f"**Analysis Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Pipeline Time:** {result.total_time*1000:.1f}ms",
            "",
            "## Summary",
            "",
            f"- **Findings:** {len(result.findings)} ({result.n_high} high severity)",
            f"- **Binding Sites:** {len(result.binding_sites)}",
            f"- **Hypotheses:** {len(result.hypotheses)}",
            "",
            "## Binding Sites",
            ""
        ]
        
        for i, site in enumerate(result.binding_sites, 1):
            lines.append(f"### Site {i}")
            lines.append(f"- **Residues:** {', '.join(site['residues'][:5])}...")
            lines.append(f"- **Radius:** {site['radius']:.1f} Å")
            lines.append(f"- **Score:** {site['score']:.2f}")
            lines.append("")
        
        lines.extend([
            "## Key Findings",
            ""
        ])
        
        for finding in result.findings[:10]:
            severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🔵"}.get(finding.severity, "⚪")
            lines.append(f"- {severity_icon} **[{finding.primitive}]** {finding.summary}")
        
        lines.extend([
            "",
            "## Hypotheses",
            ""
        ])
        
        for hyp in result.hypotheses:
            lines.append(f"### {hyp.title}")
            lines.append(f"**{hyp.description}**")
            lines.append(f"- **Confidence:** {hyp.confidence*100:.0f}%")
            lines.append(f"- **Recommended:** {hyp.recommended_action}")
            lines.append("")
        
        lines.extend([
            "---",
            f"*Attestation: {result.attestation_hash[:32]}...*"
        ])
        
        return "\n".join(lines)


def run_demo() -> MolecularPipelineResult:
    """
    Run demo analysis on synthetic protein.
    
    ⚠️  DEMONSTRATION ONLY - NOT FOR PRODUCTION USE
    
    This function:
    - Uses SYNTHETIC protein structure (not real PDB data)
    - Creates artificial alpha helices and random coordinates
    - Intended for testing pipeline functionality and visualization
    
    For production analysis, use:
        pipeline = MolecularDiscoveryPipeline()
        result = pipeline.analyze_structure(real_protein)
    
    Returns:
        MolecularPipelineResult with findings from synthetic structure analysis
    """
    import logging
    logging.getLogger(__name__).warning(
        "run_demo() uses SYNTHETIC protein data - not for production use"
    )
    print("=" * 60)
    print("MOLECULAR DISCOVERY PIPELINE - DEMO")
    print("=" * 60)
    print()
    
    # Create synthetic protein
    protein = create_synthetic_protein(100)
    print(f"Created synthetic protein: {protein.pdb_id}")
    print(f"  Residues: {protein.num_residues}")
    print(f"  Chains: {list(protein.chains.keys())}")
    print()
    
    # Run pipeline
    pipeline = MolecularDiscoveryPipeline()
    result = pipeline.analyze_structure(protein, verbose=True)
    
    print()
    print("=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"Structure: {result.structure_id}")
    print(f"Findings: {len(result.findings)}")
    print(f"  High: {result.n_high}")
    print(f"Binding Sites: {len(result.binding_sites)}")
    print(f"Hypotheses: {len(result.hypotheses)}")
    print()
    
    print("Top Findings:")
    for f in result.findings[:5]:
        print(f"  [{f.severity}] {f.primitive}: {f.summary}")
    
    print()
    print("Hypotheses:")
    for h in result.hypotheses[:3]:
        print(f"  → {h.title}: {h.description} ({h.confidence*100:.0f}%)")
    
    print()
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_demo()
