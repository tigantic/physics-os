#!/usr/bin/env python3
"""
Proof Tests for Molecular Discovery Pipeline

Phase 3 validation: Tests for drug discovery analysis pipeline.

All tests use QTT-native Genesis primitives only.
"""

from __future__ import annotations
import sys
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import torch

# Local imports
from ontic.ml.discovery.ingest.molecular import (
    MolecularIngester, ProteinStructure,
    create_synthetic_protein
)
from ontic.ml.discovery.pipelines.molecular_pipeline import (
    MolecularDiscoveryPipeline, MolecularPipelineResult
)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    time_ms: float


def run_tests() -> Tuple[List[TestResult], bool]:
    """Run all molecular pipeline tests."""
    results: List[TestResult] = []
    
    # ===== Test 1: Ingester PDB Parsing =====
    start = time.perf_counter()
    try:
        ingester = MolecularIngester()
        protein = create_synthetic_protein(50)
        
        assert protein.num_residues == 50, f"Expected 50 residues, got {protein.num_residues}"
        assert protein.num_atoms > 50, f"Expected >50 atoms, got {protein.num_atoms}"
        assert len(protein.chains) == 1, f"Expected 1 chain, got {len(protein.chains)}"
        
        results.append(TestResult(
            name="Ingester PDB Parsing",
            passed=True,
            message=f"Parsed {protein.num_residues} residues, {protein.num_atoms} atoms",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Ingester PDB Parsing",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 2: Sequence Embedding =====
    start = time.perf_counter()
    try:
        ingester = MolecularIngester()
        sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 amino acids
        embedding = ingester.embed_sequence(sequence)
        
        assert embedding.shape == (20, 23), f"Expected (20, 23), got {embedding.shape}"
        assert not torch.isnan(embedding).any(), "Embedding contains NaN"
        
        # One-hot should sum to 1 for each residue
        one_hot_sum = embedding[:, :20].sum(dim=1)
        assert torch.allclose(one_hot_sum, torch.ones(20)), "One-hot encoding incorrect"
        
        results.append(TestResult(
            name="Sequence Embedding",
            passed=True,
            message=f"23-dim embedding for {len(sequence)} residues",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Sequence Embedding",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 3: Binding Site Detection =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(100)
        ingester = MolecularIngester()
        
        # Detect binding sites
        sites = ingester.detect_binding_sites(protein)
        
        assert len(sites) >= 1, "Expected at least 1 binding site"
        assert sites[0].score > 0, "Binding site score should be positive"
        assert len(sites[0].residues) >= 3, "Binding site should have 3+ residues"
        
        results.append(TestResult(
            name="Binding Site Detection",
            passed=True,
            message=f"Detected {len(sites)} sites, top score={sites[0].score:.2f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Binding Site Detection",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 4: Distance Matrix =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(30)
        ingester = MolecularIngester()
        
        dist_matrix = ingester.compute_distance_matrix(protein, use_ca=True)
        
        assert dist_matrix.shape == (30, 30), f"Expected (30,30), got {dist_matrix.shape}"
        assert torch.allclose(dist_matrix, dist_matrix.T, atol=1e-5), "Matrix not symmetric"
        assert (torch.diag(dist_matrix) < 0.01).all(), "Diagonal should be ~0"
        
        results.append(TestResult(
            name="Distance Matrix",
            passed=True,
            message=f"30×30 symmetric matrix, mean dist={dist_matrix.mean():.1f}Å",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Distance Matrix",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 5: Contact Map =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(30)
        ingester = MolecularIngester()
        
        contact_map = ingester.compute_contact_map(protein, threshold=8.0)
        
        assert contact_map.shape == (30, 30), f"Expected (30,30)"
        assert contact_map.dtype == torch.float32, "Should be float tensor"
        assert (contact_map >= 0).all() and (contact_map <= 1).all(), "Should be binary"
        
        results.append(TestResult(
            name="Contact Map",
            passed=True,
            message=f"Contact density: {contact_map.mean()*100:.1f}%",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Contact Map",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 6: Full Pipeline Execution =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        assert result.structure_id == "SYNTH", f"Wrong structure ID"
        assert len(result.stages) == 8, f"Expected 8 stages, got {len(result.stages)}"
        assert result.total_time > 0, "Time should be positive"
        
        results.append(TestResult(
            name="Full Pipeline Execution",
            passed=True,
            message=f"8 stages in {result.total_time*1000:.0f}ms",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Full Pipeline Execution",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 7: Finding Generation =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(100)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        assert len(result.findings) >= 1, "Should generate at least 1 finding"
        
        # Check finding structure
        for f in result.findings:
            assert hasattr(f, "primitive"), "Finding missing primitive"
            assert hasattr(f, "severity"), "Finding missing severity"
            assert hasattr(f, "summary"), "Finding missing summary"
            assert f.severity in ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        results.append(TestResult(
            name="Finding Generation",
            passed=True,
            message=f"{len(result.findings)} findings from 7 primitives",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Finding Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 8: Hypothesis Generation =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(100)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        # Should generate at least 1 hypothesis from binding site
        assert len(result.hypotheses) >= 1, "Should generate hypotheses"
        
        for h in result.hypotheses:
            assert hasattr(h, "title"), "Hypothesis missing title"
            assert hasattr(h, "confidence"), "Hypothesis missing confidence"
            assert 0 <= h.confidence <= 1, f"Confidence out of range: {h.confidence}"
        
        results.append(TestResult(
            name="Hypothesis Generation",
            passed=True,
            message=f"{len(result.hypotheses)} drug discovery hypotheses",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Hypothesis Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 9: Report Generation =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        report = pipeline.generate_report(result, "Test Kinase")
        
        assert "Molecular Discovery Report" in report, "Report missing header"
        assert "Test Kinase" in report, "Report missing target name"
        assert "Binding Sites" in report, "Report missing binding sites section"
        assert len(report) > 500, f"Report too short: {len(report)} chars"
        
        results.append(TestResult(
            name="Report Generation",
            passed=True,
            message=f"Markdown report: {len(report)} chars",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Report Generation",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 10: Attestation Hash =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(30)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        assert result.attestation_hash, "Missing attestation hash"
        assert len(result.attestation_hash) == 64, "Hash should be 64 hex chars"
        
        # Running again should give different hash (time-dependent)
        result2 = pipeline.analyze_structure(protein, verbose=False)
        assert result2.attestation_hash != result.attestation_hash, "Attestations should differ"
        
        results.append(TestResult(
            name="Attestation Hash",
            passed=True,
            message=f"SHA-256 attestation: {result.attestation_hash[:16]}...",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="Attestation Hash",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 11: OT Stage (Distance Analysis) =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        # Find OT stage
        ot_stage = next((s for s in result.stages if s["primitive"] == "OT"), None)
        assert ot_stage is not None, "OT stage not found"
        assert "mean_distance" in ot_stage["metrics"], "Missing mean_distance metric"
        
        results.append(TestResult(
            name="OT Stage (Distance)",
            passed=True,
            message=f"Mean CA distance: {ot_stage['metrics']['mean_distance']:.1f}Å",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="OT Stage (Distance)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 12: RMT Stage (Spectral) =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        rmt_stage = next((s for s in result.stages if s["primitive"] == "RMT"), None)
        assert rmt_stage is not None, "RMT stage not found"
        assert "level_spacing_ratio" in rmt_stage["metrics"], "Missing level spacing"
        assert "behavior" in rmt_stage["metrics"], "Missing behavior classification"
        
        results.append(TestResult(
            name="RMT Stage (Spectral)",
            passed=True,
            message=f"Dynamics: {rmt_stage['metrics']['behavior']}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="RMT Stage (Spectral)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 13: PH Stage (Topology) =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        ph_stage = next((s for s in result.stages if s["primitive"] == "PH"), None)
        assert ph_stage is not None, "PH stage not found"
        assert "betti_0" in ph_stage["metrics"], "Missing β₀"
        assert "betti_1" in ph_stage["metrics"], "Missing β₁"
        
        results.append(TestResult(
            name="PH Stage (Topology)",
            passed=True,
            message=f"Betti: β₀={ph_stage['metrics']['betti_0']}, β₁={ph_stage['metrics']['betti_1']}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="PH Stage (Topology)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 14: GA Stage (Geometry) =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        ga_stage = next((s for s in result.stages if s["primitive"] == "GA"), None)
        assert ga_stage is not None, "GA stage not found"
        assert "shape" in ga_stage["metrics"], "Missing shape classification"
        assert "asphericity" in ga_stage["metrics"], "Missing asphericity"
        
        results.append(TestResult(
            name="GA Stage (Geometry)",
            passed=True,
            message=f"Shape: {ga_stage['metrics']['shape']}, κ²={ga_stage['metrics']['kappa2']:.3f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="GA Stage (Geometry)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # ===== Test 15: TG Stage (Tropical) =====
    start = time.perf_counter()
    try:
        protein = create_synthetic_protein(50)
        pipeline = MolecularDiscoveryPipeline()
        
        result = pipeline.analyze_structure(protein, verbose=False)
        
        tg_stage = next((s for s in result.stages if s["primitive"] == "TG"), None)
        assert tg_stage is not None, "TG stage not found"
        assert "tropical_eigenvalue" in tg_stage["metrics"], "Missing tropical eigenvalue"
        
        results.append(TestResult(
            name="TG Stage (Tropical)",
            passed=True,
            message=f"Tropical eigenvalue: {tg_stage['metrics']['tropical_eigenvalue']:.3f}",
            time_ms=(time.perf_counter() - start) * 1000
        ))
    except Exception as e:
        results.append(TestResult(
            name="TG Stage (Tropical)",
            passed=False,
            message=str(e),
            time_ms=(time.perf_counter() - start) * 1000
        ))
    
    # Compute overall status
    all_passed = all(r.passed for r in results)
    return results, all_passed


def main():
    """Run molecular pipeline proof tests."""
    print("=" * 60)
    print("MOLECULAR PIPELINE PROOF TESTS")
    print("Phase 3: Drug Discovery")
    print("=" * 60)
    print()
    
    results, all_passed = run_tests()
    
    # Print results
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"{status} | {r.name}")
        print(f"       {r.message}")
        print(f"       ({r.time_ms:.1f}ms)")
        print()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("=" * 60)
    print(f"MOLECULAR PIPELINE: {passed}/{total} tests passed")
    if all_passed:
        print("✅ Phase 3 validation COMPLETE")
    else:
        print("❌ Some tests failed - review above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
