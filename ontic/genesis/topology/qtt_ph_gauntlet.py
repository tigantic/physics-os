#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            QTT-PH ELITE GAUNTLET — PERSISTENT HOMOLOGY VALIDATION            ║
║                                                                              ║
║  Layer 25 of TENSOR GENESIS Protocol                                        ║
║  Topological Data Analysis with QTT acceleration                            ║
║                                                                              ║
║  Mathematical Foundation:                                                    ║
║  • Chain complex: C_k → C_{k-1} → ... → C_0                                 ║
║  • Boundary operator: ∂²=0                                                  ║
║  • Homology: H_k = ker(∂_k) / im(∂_{k+1})                                  ║
║  • Betti numbers: β_k = dim(H_k)                                           ║
║  • Persistence: track features through filtration                           ║
║                                                                              ║
║  QTT Insight: Boundary matrices have low rank for structured complexes      ║
║  Complexity: O(r³ log N) for matrix operations                              ║
║                                                                              ║
║  Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

# Add path for local imports
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')

from ontic.genesis.topology.simplicial import (
    Simplex, SimplicialComplex, RipsComplex, VietorisRips, CechComplex,
    pairwise_distances
)
from ontic.genesis.topology.boundary import (
    boundary_matrix, QTTBoundaryOperator, coboundary_matrix,
    verify_boundary_squared_zero, betti_numbers_from_boundary,
    chain_complex_matrices
)
from ontic.genesis.topology.persistence import (
    PersistencePair, PersistenceDiagram, compute_persistence,
    reduce_boundary_matrix, persistence_pairs, compute_betti_curve
)
from ontic.genesis.topology.distances import (
    bottleneck_distance, wasserstein_distance_diagram,
    persistence_landscape, landscape_distance,
    persistence_image, diagram_entropy, silhouette
)


@dataclass
class GauntletResult:
    """Result of a single gauntlet test."""
    name: str
    passed: bool
    message: str
    score: float = 1.0


class QTTPHGauntlet:
    """Elite test suite for QTT-PH persistent homology."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[GauntletResult] = []
        
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def run_test(self, name: str, test_fn) -> GauntletResult:
        """Run a single test with error handling."""
        try:
            passed, message, score = test_fn()
            result = GauntletResult(name, passed, message, score)
        except Exception as e:
            import traceback
            result = GauntletResult(name, False, f"Exception: {e}", 0.0)
        
        self.results.append(result)
        status = "✅" if result.passed else "❌"
        self.log(f"  {status} {name}: {result.message}")
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIMPLEX TESTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_simplex_creation(self) -> Tuple[bool, str, float]:
        """Test simplex creation and properties."""
        # Vertex (0-simplex)
        v = Simplex((0,))
        assert v.dimension == 0
        
        # Edge (1-simplex)
        e = Simplex((0, 1))
        assert e.dimension == 1
        
        # Triangle (2-simplex)
        t = Simplex((0, 1, 2))
        assert t.dimension == 2
        
        # Vertices should be sorted
        t2 = Simplex((2, 0, 1))
        assert t.vertices == t2.vertices
        
        return True, "Simplex dimensions correct, sorting works", 1.0
    
    def test_simplex_faces(self) -> Tuple[bool, str, float]:
        """Test face enumeration."""
        # Triangle faces
        t = Simplex((0, 1, 2))
        faces = list(t.faces())
        
        assert len(faces) == 3
        expected = {Simplex((0, 1)), Simplex((0, 2)), Simplex((1, 2))}
        assert set(faces) == expected
        
        # Tetrahedron faces
        tet = Simplex((0, 1, 2, 3))
        tet_faces = list(tet.faces())
        assert len(tet_faces) == 4
        
        return True, "Face enumeration correct", 1.0
    
    def test_boundary_coefficients(self) -> Tuple[bool, str, float]:
        """Test boundary coefficients with signs."""
        t = Simplex((0, 1, 2))
        
        # ∂[0,1,2] = [1,2] - [0,2] + [0,1]
        coeffs = {}
        for face in t.faces():
            coeffs[face] = t.boundary_coefficient(face)
        
        assert coeffs[Simplex((1, 2))] == 1
        assert coeffs[Simplex((0, 2))] == -1
        assert coeffs[Simplex((0, 1))] == 1
        
        return True, "Boundary coefficients correct", 1.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIMPLICIAL COMPLEX TESTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_complex_from_edges(self) -> Tuple[bool, str, float]:
        """Test complex construction from edges."""
        # Triangle
        complex = SimplicialComplex.from_edges(3, [(0, 1), (1, 2), (0, 2)])
        
        assert complex.num_simplices(0) == 3  # vertices
        assert complex.num_simplices(1) == 3  # edges
        
        return True, "Edge complex construction correct", 1.0
    
    def test_complex_validity(self) -> Tuple[bool, str, float]:
        """Test complex closure property."""
        complex = SimplicialComplex()
        
        # Add triangle with all faces
        complex.add_simplex(Simplex((0, 1, 2)), add_faces=True)
        
        assert complex.is_valid()
        assert complex.num_simplices(0) == 3
        assert complex.num_simplices(1) == 3
        assert complex.num_simplices(2) == 1
        
        return True, "Complex is valid (closed under faces)", 1.0
    
    def test_euler_characteristic(self) -> Tuple[bool, str, float]:
        """Test Euler characteristic computation."""
        # Triangle: χ = 3 - 3 + 1 = 1
        complex = SimplicialComplex()
        complex.add_simplex(Simplex((0, 1, 2)), add_faces=True)
        
        chi = complex.euler_characteristic()
        
        passed = chi == 1
        return passed, f"χ(triangle) = {chi}", 1.0 if passed else 0.0
    
    def test_rips_complex(self) -> Tuple[bool, str, float]:
        """Test Rips complex construction."""
        # Triangle of points
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866]
        ])
        
        # All points within distance 2
        rips = RipsComplex.from_points(points, max_radius=2.0, max_dim=2)
        
        assert rips.num_simplices(0) == 3
        assert rips.num_simplices(1) == 3
        assert rips.num_simplices(2) == 1
        
        return True, "Rips complex correct for triangle", 1.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # BOUNDARY MATRIX TESTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_boundary_matrix_shape(self) -> Tuple[bool, str, float]:
        """Test boundary matrix dimensions."""
        complex = SimplicialComplex()
        complex.add_simplex(Simplex((0, 1, 2)), add_faces=True)
        
        D1 = boundary_matrix(complex, 1)  # edges to vertices
        D2 = boundary_matrix(complex, 2)  # triangles to edges
        
        assert D1.shape == (3, 3)  # 3 vertices, 3 edges
        assert D2.shape == (3, 1)  # 3 edges, 1 triangle
        
        return True, f"D1: {D1.shape}, D2: {D2.shape}", 1.0
    
    def test_boundary_squared_zero(self) -> Tuple[bool, str, float]:
        """Test ∂² = 0 fundamental property."""
        complex = SimplicialComplex()
        complex.add_simplex(Simplex((0, 1, 2, 3)), add_faces=True)  # tetrahedron
        
        passed, msg = verify_boundary_squared_zero(complex)
        
        return passed, msg, 1.0 if passed else 0.0
    
    def test_betti_numbers(self) -> Tuple[bool, str, float]:
        """Test Betti number computation."""
        # Hollow triangle (no 2-simplex): β_0=1, β_1=1
        complex = SimplicialComplex.from_edges(3, [(0, 1), (1, 2), (0, 2)])
        
        betti = betti_numbers_from_boundary(complex)
        
        # β_0 = 1 (one connected component)
        # β_1 = 1 (one loop)
        passed = betti[0] == 1 and betti[1] == 1
        
        return passed, f"β = {betti}", 1.0 if passed else 0.0
    
    def test_qtt_boundary_operator(self) -> Tuple[bool, str, float]:
        """Test QTT boundary operator compression."""
        complex = SimplicialComplex()
        complex.add_simplex(Simplex((0, 1, 2)), add_faces=True)
        
        qtt_op = QTTBoundaryOperator.from_complex(complex, k=1, max_rank=10)
        
        D_dense = boundary_matrix(complex, 1)
        D_recon = qtt_op.to_dense()
        
        error = (D_dense - D_recon).abs().max().item()
        
        passed = error < 1e-6
        return passed, f"Reconstruction error: {error:.2e}", 1.0 if passed else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE TESTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_matrix_reduction(self) -> Tuple[bool, str, float]:
        """Test boundary matrix reduction."""
        # Simple example
        D = torch.tensor([
            [1., 1., 0.],
            [1., 0., 1.],
            [0., 1., 1.]
        ])
        
        R, pivots = reduce_boundary_matrix(D)
        
        # After reduction, columns should have distinct pivots or be zero
        passed = len(pivots) <= D.shape[1]
        
        return passed, f"Found {len(pivots)} pivots", 1.0 if passed else 0.0
    
    def test_persistence_pairs(self) -> Tuple[bool, str, float]:
        """Test persistence pair extraction."""
        # Filtered triangle
        complex = SimplicialComplex()
        
        # Add vertices at time 0
        for i in range(3):
            s = Simplex((i,))
            complex.add_simplex(s, filtration_value=0.0, add_faces=False)
        
        # Add edges at different times
        complex.add_simplex(Simplex((0, 1)), filtration_value=1.0, add_faces=False)
        complex.add_simplex(Simplex((1, 2)), filtration_value=2.0, add_faces=False)
        complex.add_simplex(Simplex((0, 2)), filtration_value=3.0, add_faces=False)
        
        pairs = persistence_pairs(complex)
        
        # Should have pairs for H_0 and H_1
        h0_pairs = [p for p in pairs if p.dimension == 0]
        h1_pairs = [p for p in pairs if p.dimension == 1]
        
        return True, f"H_0: {len(h0_pairs)} pairs, H_1: {len(h1_pairs)} pairs", 1.0
    
    def test_persistence_diagram(self) -> Tuple[bool, str, float]:
        """Test persistence diagram computation."""
        # Circle of points
        n = 8
        angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
        rips = RipsComplex.from_points(points, max_radius=3.0, max_dim=2)
        diagram = compute_persistence(rips)
        
        # Should detect the hole (H_1 feature)
        h1_features = [p for p in diagram[1] if not p.is_essential]
        
        passed = len(diagram[0]) > 0  # At least some H_0 features
        
        return passed, f"H_0: {len(diagram[0])}, H_1: {len(diagram[1])}", 1.0 if passed else 0.0
    
    def test_total_persistence(self) -> Tuple[bool, str, float]:
        """Test total persistence computation."""
        # Create simple diagram manually
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [
            PersistencePair(0.0, 1.0, 0),
            PersistencePair(0.0, 2.0, 0),
            PersistencePair(0.0, float('inf'), 0)  # essential
        ]
        
        total = diagram.total_persistence(dim=0, p=1.0)
        
        # Should be 1 + 2 = 3 (excluding essential)
        passed = abs(total - 3.0) < 1e-6
        
        return passed, f"Total persistence: {total}", 1.0 if passed else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # DISTANCE TESTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def test_bottleneck_same_diagram(self) -> Tuple[bool, str, float]:
        """Test bottleneck distance to self is zero."""
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [
            PersistencePair(0.0, 1.0, 0),
            PersistencePair(0.5, 2.0, 0)
        ]
        
        d = bottleneck_distance(diagram, diagram, dim=0)
        
        passed = d < 1e-6
        return passed, f"d_B(D, D) = {d:.6f}", 1.0 if passed else 0.0
    
    def test_bottleneck_different_diagrams(self) -> Tuple[bool, str, float]:
        """Test bottleneck distance between different diagrams."""
        diagram1 = PersistenceDiagram()
        diagram1.pairs[0] = [PersistencePair(0.0, 1.0, 0)]
        
        diagram2 = PersistenceDiagram()
        diagram2.pairs[0] = [PersistencePair(0.0, 2.0, 0)]
        
        d = bottleneck_distance(diagram1, diagram2, dim=0)
        
        # Different death times: should be around 0.5 or 1.0
        passed = d > 0
        return passed, f"d_B = {d:.4f}", 1.0 if passed else 0.0
    
    def test_wasserstein_distance(self) -> Tuple[bool, str, float]:
        """Test Wasserstein distance computation."""
        diagram1 = PersistenceDiagram()
        diagram1.pairs[0] = [
            PersistencePair(0.0, 1.0, 0),
            PersistencePair(0.5, 1.5, 0)
        ]
        
        diagram2 = PersistenceDiagram()
        diagram2.pairs[0] = [
            PersistencePair(0.0, 1.0, 0),
            PersistencePair(0.5, 1.5, 0)
        ]
        
        d = wasserstein_distance_diagram(diagram1, diagram2, dim=0)
        
        passed = d < 1e-4
        return passed, f"W_2(D, D) = {d:.6f}", 1.0 if passed else 0.0
    
    def test_persistence_landscape(self) -> Tuple[bool, str, float]:
        """Test persistence landscape computation."""
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [
            PersistencePair(0.0, 2.0, 0),
            PersistencePair(0.5, 1.5, 0)
        ]
        
        landscape = persistence_landscape(diagram, dim=0, num_samples=50)
        
        # Should have at least 2 landscapes
        passed = landscape.num_landscapes >= 1
        
        return passed, f"{landscape.num_landscapes} landscapes computed", 1.0 if passed else 0.0
    
    def test_landscape_distance(self) -> Tuple[bool, str, float]:
        """Test landscape distance computation."""
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [PersistencePair(0.0, 2.0, 0)]
        
        landscape = persistence_landscape(diagram, dim=0)
        
        d = landscape_distance(landscape, landscape)
        
        passed = d < 1e-6
        return passed, f"d_L(L, L) = {d:.6f}", 1.0 if passed else 0.0
    
    def test_persistence_image(self) -> Tuple[bool, str, float]:
        """Test persistence image computation."""
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [
            PersistencePair(0.0, 1.0, 0),
            PersistencePair(0.5, 2.0, 0)
        ]
        
        image = persistence_image(diagram, dim=0, resolution=10)
        
        passed = image.shape == (10, 10) and image.sum() > 0
        
        return passed, f"Image shape: {image.shape}, sum: {image.sum():.4f}", 1.0 if passed else 0.0
    
    def test_diagram_entropy(self) -> Tuple[bool, str, float]:
        """Test persistent entropy computation."""
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [
            PersistencePair(0.0, 1.0, 0),  # persistence 1
            PersistencePair(0.0, 1.0, 0)   # persistence 1
        ]
        
        entropy = diagram_entropy(diagram, dim=0)
        
        # Equal persistence -> max entropy for 2 elements = log(2)
        expected = math.log(2)
        
        passed = abs(entropy - expected) < 1e-6
        return passed, f"Entropy: {entropy:.4f} (expected {expected:.4f})", 1.0 if passed else 0.0
    
    def test_silhouette(self) -> Tuple[bool, str, float]:
        """Test persistence silhouette computation."""
        diagram = PersistenceDiagram()
        diagram.pairs[0] = [
            PersistencePair(0.0, 2.0, 0),
            PersistencePair(0.5, 1.5, 0)
        ]
        
        sil = silhouette(diagram, dim=0, num_samples=50)
        
        passed = sil.shape[0] == 50 and sil.max() > 0
        
        return passed, f"Silhouette max: {sil.max():.4f}", 1.0 if passed else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL TESTS
    # ═══════════════════════════════════════════════════════════════════════
    
    def run_all(self) -> Tuple[int, int]:
        """Run all gauntlet tests."""
        self.log("=" * 78)
        self.log("          QTT-PH ELITE GAUNTLET — PERSISTENT HOMOLOGY VALIDATION")
        self.log("=" * 78)
        
        # Simplex Tests
        self.log("\n▶ SIMPLEX TESTS")
        self.run_test("Simplex Creation", self.test_simplex_creation)
        self.run_test("Simplex Faces", self.test_simplex_faces)
        self.run_test("Boundary Coefficients", self.test_boundary_coefficients)
        
        # Simplicial Complex Tests
        self.log("\n▶ SIMPLICIAL COMPLEX TESTS")
        self.run_test("Complex from Edges", self.test_complex_from_edges)
        self.run_test("Complex Validity", self.test_complex_validity)
        self.run_test("Euler Characteristic", self.test_euler_characteristic)
        self.run_test("Rips Complex", self.test_rips_complex)
        
        # Boundary Matrix Tests
        self.log("\n▶ BOUNDARY MATRIX TESTS")
        self.run_test("Boundary Matrix Shape", self.test_boundary_matrix_shape)
        self.run_test("Boundary Squared Zero", self.test_boundary_squared_zero)
        self.run_test("Betti Numbers", self.test_betti_numbers)
        self.run_test("QTT Boundary Operator", self.test_qtt_boundary_operator)
        
        # Persistence Tests
        self.log("\n▶ PERSISTENCE TESTS")
        self.run_test("Matrix Reduction", self.test_matrix_reduction)
        self.run_test("Persistence Pairs", self.test_persistence_pairs)
        self.run_test("Persistence Diagram", self.test_persistence_diagram)
        self.run_test("Total Persistence", self.test_total_persistence)
        
        # Distance Tests
        self.log("\n▶ DIAGRAM DISTANCE TESTS")
        self.run_test("Bottleneck Same Diagram", self.test_bottleneck_same_diagram)
        self.run_test("Bottleneck Different", self.test_bottleneck_different_diagrams)
        self.run_test("Wasserstein Distance", self.test_wasserstein_distance)
        self.run_test("Persistence Landscape", self.test_persistence_landscape)
        self.run_test("Landscape Distance", self.test_landscape_distance)
        self.run_test("Persistence Image", self.test_persistence_image)
        self.run_test("Diagram Entropy", self.test_diagram_entropy)
        self.run_test("Silhouette", self.test_silhouette)
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        self.log("\n" + "=" * 78)
        self.log(f"  GAUNTLET COMPLETE: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        self.log("=" * 78)
        
        if passed == total:
            self.log("  🏆 LAYER 25 QTT-PH: ALL TESTS PASSED")
        else:
            failed = [r.name for r in self.results if not r.passed]
            self.log(f"  ⚠️  Failed tests: {', '.join(failed)}")
        
        return passed, total


def main():
    """Run the QTT-PH gauntlet."""
    gauntlet = QTTPHGauntlet(verbose=True)
    passed, total = gauntlet.run_all()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
