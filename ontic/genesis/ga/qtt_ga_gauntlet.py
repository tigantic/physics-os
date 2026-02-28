#!/usr/bin/env python3
"""
QTT-GA ELITE GAUNTLET — Geometric Algebra Validation

Layer 26 of TENSOR GENESIS Protocol

Tests all geometric algebra operations:
1. Multivector creation and arithmetic
2. Geometric, inner, outer products
3. Operations (reverse, dual, magnitude)
4. Rotors and rotations
5. Conformal GA
6. QTT compression

Constitutional Reference: TENSOR_GENESIS.md
Target: 100% pass rate

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

import sys
import math
import torch
from typing import Tuple, List

# Test results tracking
PASSED = 0
FAILED = 0
FAILED_TESTS = []


def test(name: str, condition: bool, detail: str = ""):
    """Record test result."""
    global PASSED, FAILED, FAILED_TESTS
    if condition:
        PASSED += 1
        print(f"  ✅ {name}: {detail}")
    else:
        FAILED += 1
        FAILED_TESTS.append(name)
        print(f"  ❌ {name}: {detail}")


def section(name: str):
    """Print section header."""
    print(f"\n▶ {name}")


# =============================================================================
# TESTS
# =============================================================================

def test_multivector_creation():
    """Test multivector creation and basic properties."""
    section("MULTIVECTOR TESTS")
    
    from ontic.genesis.ga import CliffordAlgebra, Multivector, scalar, vector, bivector
    
    # Test algebra creation
    vga = CliffordAlgebra(p=3, q=0, r=0)
    test("Algebra Creation", vga.n == 3 and vga.dim == 8, 
         f"Cl(3,0,0): n=3, dim=8")
    
    # Test scalar
    s = scalar(vga, 2.5)
    test("Scalar Creation", abs(s.coeffs[0] - 2.5) < 1e-10 and s.is_scalar,
         f"scalar=2.5, is_scalar={s.is_scalar}")
    
    # Test vector
    v = vector(vga, [1.0, 2.0, 3.0])
    test("Vector Creation", v.is_vector and abs(v.coeffs[1] - 1.0) < 1e-10,
         f"v = e1 + 2e2 + 3e3")
    
    # Test bivector
    B = bivector(vga, {(0, 1): 1.0})  # e12
    test("Bivector Creation", B.is_bivector and abs(B.coeffs[3] - 1.0) < 1e-10,
         f"B = e12")
    
    # Test arithmetic
    v2 = vector(vga, [1.0, 1.0, 1.0])
    v_sum = v + v2
    test("Vector Addition", abs(v_sum.coeffs[1] - 2.0) < 1e-10,
         f"(1+1)e1 = 2e1")
    
    v_scaled = v * 2.0
    test("Scalar Multiplication", abs(v_scaled.coeffs[1] - 2.0) < 1e-10,
         f"2 * e1 = 2e1")


def test_geometric_product():
    """Test the geometric product."""
    section("GEOMETRIC PRODUCT TESTS")
    
    from ontic.genesis.ga import (
        CliffordAlgebra, vector, geometric_product, scalar
    )
    
    vga = CliffordAlgebra(p=3, q=0, r=0)
    
    # e1 * e1 = 1
    e1 = vector(vga, [1.0, 0.0, 0.0])
    e1_sq = geometric_product(e1, e1)
    test("e1² = 1", e1_sq.is_scalar and abs(e1_sq.coeffs[0] - 1.0) < 1e-10,
         f"e1² = {e1_sq.coeffs[0]:.4f}")
    
    # e1 * e2 = e12 (bivector)
    e2 = vector(vga, [0.0, 1.0, 0.0])
    e12 = geometric_product(e1, e2)
    test("e1*e2 = e12", e12.is_bivector,
         f"e1*e2 is bivector")
    
    # e2 * e1 = -e12 (anti-commutative)
    e21 = geometric_product(e2, e1)
    test("e2*e1 = -e12", abs(e21.coeffs[3] + e12.coeffs[3]) < 1e-10,
         f"Anti-commutative: e21 = {e21.coeffs[3]:.4f}")
    
    # (e1 + e2)² = e1² + e2² + e1*e2 + e2*e1 = 2
    v = e1 + e2
    v_sq = geometric_product(v, v)
    test("(e1+e2)² = 2", abs(v_sq.coeffs[0] - 2.0) < 1e-10,
         f"(e1+e2)² = {v_sq.coeffs[0]:.4f}")


def test_inner_outer_products():
    """Test inner and outer products."""
    section("INNER/OUTER PRODUCT TESTS")
    
    from ontic.genesis.ga import (
        CliffordAlgebra, vector, bivector,
        inner_product, outer_product
    )
    
    vga = CliffordAlgebra(p=3, q=0, r=0)
    
    e1 = vector(vga, [1.0, 0.0, 0.0])
    e2 = vector(vga, [0.0, 1.0, 0.0])
    e3 = vector(vga, [0.0, 0.0, 1.0])
    
    # Inner product of orthogonal vectors = 0
    dot = inner_product(e1, e2)
    test("e1·e2 = 0", abs(dot.coeffs[0]) < 1e-10,
         f"Orthogonal inner product = {dot.coeffs[0]:.4f}")
    
    # Inner product with self = 1
    dot_self = inner_product(e1, e1)
    test("e1·e1 = 1", abs(dot_self.coeffs[0] - 1.0) < 1e-10,
         f"Self inner product = {dot_self.coeffs[0]:.4f}")
    
    # Outer product e1 ∧ e2 = e12
    wedge = outer_product(e1, e2)
    test("e1∧e2 = e12", wedge.is_bivector,
         f"Outer product is bivector")
    
    # Outer product e1 ∧ e1 = 0
    wedge_self = outer_product(e1, e1)
    test("e1∧e1 = 0", wedge_self.norm() < 1e-10,
         f"Self outer product = {wedge_self.norm():.6f}")
    
    # e1 ∧ e2 ∧ e3 = pseudoscalar
    temp = outer_product(e1, e2)
    trivector = outer_product(temp, e3)
    test("e1∧e2∧e3 = I", trivector.grades == [3],
         f"Triple wedge is grade-3")


def test_operations():
    """Test multivector operations."""
    section("OPERATION TESTS")
    
    from ontic.genesis.ga import (
        CliffordAlgebra, vector, bivector, scalar,
        reverse, grade_involution, magnitude, normalize, inverse,
        grade_projection, dual, geometric_product
    )
    
    vga = CliffordAlgebra(p=3, q=0, r=0)
    
    # Reverse of vector = vector (grade 1: sign = (-1)^0 = 1)
    v = vector(vga, [1.0, 2.0, 3.0])
    v_rev = reverse(v)
    test("Reverse of vector", torch.allclose(v.coeffs, v_rev.coeffs),
         f"ṽ = v for grade 1")
    
    # Reverse of bivector = -bivector (grade 2: sign = (-1)^1 = -1)
    B = bivector(vga, {(0, 1): 1.0})
    B_rev = reverse(B)
    test("Reverse of bivector", abs(B_rev.coeffs[3] + 1.0) < 1e-10,
         f"B̃ = -B for grade 2")
    
    # Grade involution
    v_hat = grade_involution(v)
    test("Grade involution of vector", torch.allclose(v_hat.coeffs, -v.coeffs),
         f"v̂ = -v for grade 1")
    
    # Magnitude
    v_mag = magnitude(v)
    expected_mag = math.sqrt(1 + 4 + 9)
    test("Vector magnitude", abs(v_mag - expected_mag) < 1e-6,
         f"|v| = {v_mag:.4f} (expected {expected_mag:.4f})")
    
    # Normalize
    v_unit = normalize(v)
    test("Normalize", abs(magnitude(v_unit) - 1.0) < 1e-6,
         f"|v̂| = {magnitude(v_unit):.4f}")
    
    # Inverse
    v_inv = inverse(v)
    v_v_inv = geometric_product(v, v_inv)
    test("Inverse v*v⁻¹=1", abs(v_v_inv.coeffs[0] - 1.0) < 1e-10,
         f"v*v⁻¹ scalar part = {v_v_inv.coeffs[0]:.4f}")
    
    # Grade projection
    mixed = v + scalar(vga, 5.0)
    grade1 = grade_projection(mixed, 1)
    test("Grade projection", torch.allclose(grade1.coeffs, v.coeffs),
         f"<v+5>_1 = v")


def test_rotors():
    """Test rotor operations."""
    section("ROTOR TESTS")
    
    from ontic.genesis.ga import (
        CliffordAlgebra, vector, bivector,
        rotor_from_bivector, rotor_from_angle_plane, rotor_from_vectors,
        apply_rotor, rotor_log, compose_rotors, magnitude
    )
    
    vga = CliffordAlgebra(p=3, q=0, r=0)
    
    # Create rotor for 90-degree rotation in xy-plane
    B = bivector(vga, {(0, 1): 1.0})  # e12
    R = rotor_from_angle_plane(math.pi / 2, B)
    
    # Rotor should be even and unit magnitude
    test("Rotor is even", R.is_even, f"Rotor grades: {R.grades}")
    test("Rotor unit magnitude", abs(magnitude(R) - 1.0) < 1e-10,
         f"|R| = {magnitude(R):.4f}")
    
    # Rotate e1 by 90° in xy-plane → should get e2
    e1 = vector(vga, [1.0, 0.0, 0.0])
    e1_rot = apply_rotor(R, e1)
    expected = vector(vga, [0.0, 1.0, 0.0])
    
    error = (e1_rot - expected).norm()
    test("90° rotation e1→e2", error < 1e-10,
         f"Rotated: [{e1_rot.coeffs[1]:.4f}, {e1_rot.coeffs[2]:.4f}, {e1_rot.coeffs[4]:.4f}]")
    
    # Rotor from two vectors
    e2 = vector(vga, [0.0, 1.0, 0.0])
    R2 = rotor_from_vectors(e1, e2)
    e1_rot2 = apply_rotor(R2, e1)
    
    # Should rotate to e2 direction
    test("Rotor from vectors", abs(e1_rot2.coeffs[2] - 1.0) < 1e-6,
         f"e1 → e2 via rotor")
    
    # Rotor log
    log_R = rotor_log(R)
    test("Rotor log is bivector", log_R.is_bivector,
         f"log(R) grades: {log_R.grades}")
    
    # Compose rotors (two 90° rotations = 180° rotation)
    R_composed = compose_rotors(R, R)
    e1_180 = apply_rotor(R_composed, e1)
    expected_180 = vector(vga, [-1.0, 0.0, 0.0])
    
    error_180 = (e1_180 - expected_180).norm()
    test("Composed rotation", error_180 < 1e-10,
         f"180° rotation: e1 → -e1")


def test_conformal_ga():
    """Test conformal geometric algebra."""
    section("CONFORMAL GA TESTS")
    
    from ontic.genesis.ga import (
        ConformalGA, point_to_cga, cga_to_point, 
        cga_sphere, cga_plane, distance_point_to_point, inner_product
    )
    
    cga = ConformalGA()
    
    # Test CGA algebra setup
    test("CGA dimension", cga.algebra.dim == 32, f"Cl(4,1,0) dim = {cga.algebra.dim}")
    
    # Null vectors
    e0 = cga.e_origin()
    einf = cga.e_infinity()
    
    # e0 · einf = -1
    e0_einf = inner_product(e0, einf)
    test("e₀·e∞ = -1", abs(e0_einf.coeffs[0] + 1.0) < 1e-10,
         f"e₀·e∞ = {e0_einf.coeffs[0]:.4f}")
    
    # Point embedding and extraction
    p = [1.0, 2.0, 3.0]
    P = point_to_cga(cga, p)
    p_back = cga_to_point(cga, P)
    
    error = sum((a - b)**2 for a, b in zip(p, p_back)) ** 0.5
    test("Point roundtrip", error < 1e-10,
         f"[1,2,3] → CGA → [{p_back[0]:.4f}, {p_back[1]:.4f}, {p_back[2]:.4f}]")
    
    # Distance between points
    p1 = [0.0, 0.0, 0.0]
    p2 = [3.0, 4.0, 0.0]
    P1 = point_to_cga(cga, p1)
    P2 = point_to_cga(cga, p2)
    
    dist = distance_point_to_point(cga, P1, P2)
    test("CGA distance", abs(dist - 5.0) < 1e-10,
         f"d([0,0,0], [3,4,0]) = {dist:.4f}")
    
    # Sphere creation
    S = cga_sphere(cga, center=[0, 0, 0], radius=1.0)
    test("Sphere creation", S is not None and S.norm() > 0,
         f"Unit sphere created")
    
    # Plane creation
    pi = cga_plane(cga, normal=[0, 0, 1], distance=0.0)
    test("Plane creation", pi is not None,
         f"xy-plane created")


def test_qtt_multivector():
    """Test QTT-compressed multivectors."""
    section("QTT MULTIVECTOR TESTS")
    
    from ontic.genesis.ga import (
        QTTMultivector, qtt_geometric_product, qtt_grade_projection
    )
    
    # Small test: n=4 (16 components)
    n = 4
    
    # Create scalar
    s = QTTMultivector.scalar(n, 3.0, p=4)
    s_dense = s.to_dense()
    test("QTT scalar", abs(s_dense[0] - 3.0) < 1e-10 and s_dense[1:].abs().max() < 1e-10,
         f"Scalar at index 0 = {s_dense[0]:.4f}")
    
    # Create basis blade
    blade = QTTMultivector.basis_blade(n, blade_index=5, coefficient=2.0, p=4)
    blade_dense = blade.to_dense()
    test("QTT basis blade", abs(blade_dense[5] - 2.0) < 1e-10,
         f"Blade 5 coefficient = {blade_dense[5]:.4f}")
    
    # Get coefficient
    coeff = blade.get_coefficient(5)
    test("Get coefficient", abs(coeff - 2.0) < 1e-10,
         f"get_coefficient(5) = {coeff:.4f}")
    
    # QTT norm
    random_mv = QTTMultivector.random(n, rank=5, p=4)
    qtt_norm = random_mv.norm()
    dense_norm = random_mv.to_dense().norm().item()
    test("QTT norm", abs(qtt_norm - dense_norm) / (dense_norm + 1e-10) < 1e-6,
         f"QTT norm = {qtt_norm:.4f}, dense norm = {dense_norm:.4f}")
    
    # From dense and back
    original = torch.randn(16)
    qtt = QTTMultivector.from_dense(original, p=4, max_rank=50)
    reconstructed = qtt.to_dense()
    error = (original - reconstructed).norm() / original.norm()
    test("Dense roundtrip", error < 1e-6,
         f"Relative error = {error:.2e}")
    
    # QTT geometric product (small n)
    a = QTTMultivector.basis_blade(4, 1, 1.0, p=4)  # e1
    b = QTTMultivector.basis_blade(4, 2, 1.0, p=4)  # e2
    ab = qtt_geometric_product(a, b, max_rank=50)
    ab_dense = ab.to_dense()
    
    # e1 * e2 = e12 (blade index 3 = 0b11)
    test("QTT geometric product", abs(ab_dense[3] - 1.0) < 1e-10,
         f"e1*e2 coefficient at e12 = {ab_dense[3]:.4f}")
    
    # Grade projection
    # Create 1 + e1 + e12
    mv = QTTMultivector.scalar(4, 1.0, p=4)
    mv_dense = mv.to_dense()
    mv_dense[1] = 1.0  # e1
    mv_dense[3] = 1.0  # e12
    mv = QTTMultivector.from_dense(mv_dense, p=4)
    
    grade1 = qtt_grade_projection(mv, 1)
    grade1_dense = grade1.to_dense()
    
    test("QTT grade projection", 
         abs(grade1_dense[0]) < 1e-6 and  # no scalar
         abs(grade1_dense[1] - 1.0) < 1e-6 and  # e1
         abs(grade1_dense[3]) < 1e-6,  # no e12
         f"<1+e1+e12>_1 = e1")


def test_large_qtt():
    """Test QTT with larger algebras."""
    section("LARGE QTT TESTS")
    
    from ontic.genesis.ga import QTTMultivector
    from ontic.genesis.ga.qtt_multivector import qtt_add, qtt_scale, qtt_inner_product
    
    # n=10: 1024 components
    n = 10
    
    # Random multivectors
    a = QTTMultivector.random(n, rank=5, p=n)
    b = QTTMultivector.random(n, rank=5, p=n)
    
    # Addition
    c = qtt_add(a, b)
    
    # Verify by computing inner products
    # <a+b, a+b> = <a,a> + 2<a,b> + <b,b>
    aa = qtt_inner_product(a, a)
    bb = qtt_inner_product(b, b)
    ab = qtt_inner_product(a, b)
    cc = qtt_inner_product(c, c)
    
    expected_cc = aa + 2*ab + bb
    test("QTT addition consistency", abs(cc - expected_cc) / (abs(expected_cc) + 1e-10) < 0.1,
         f"<a+b,a+b> = {cc:.4f}, expected {expected_cc:.4f}")
    
    # Scaling
    d = qtt_scale(a, 2.0)
    dd = qtt_inner_product(d, d)
    expected_dd = 4 * aa
    
    test("QTT scaling", abs(dd - expected_dd) / (abs(expected_dd) + 1e-10) < 1e-6,
         f"<2a,2a> = {dd:.4f}, expected {expected_dd:.4f}")
    
    # Memory efficiency
    dense_size = 2**n  # 1024 floats
    qtt_size = sum(c.numel() for c in a.cores)
    compression = dense_size / qtt_size
    
    test("Compression ratio", compression > 1,
         f"Dense: {dense_size}, QTT: {qtt_size}, ratio: {compression:.2f}x")
    
    # Very large algebra (n=20, 1M components)
    n_large = 20
    large_mv = QTTMultivector.random(n_large, rank=10, p=n_large)
    qtt_size_large = sum(c.numel() for c in large_mv.cores)
    
    test("Large algebra storage", qtt_size_large < 10000,
         f"Cl(20): 2^20 = 1M components, QTT uses {qtt_size_large} floats")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("=" * 78)
    print("          QTT-GA ELITE GAUNTLET — GEOMETRIC ALGEBRA VALIDATION")
    print("=" * 78)
    
    try:
        test_multivector_creation()
        test_geometric_product()
        test_inner_outer_products()
        test_operations()
        test_rotors()
        test_conformal_ga()
        test_qtt_multivector()
        test_large_qtt()
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    total = PASSED + FAILED
    pct = 100 * PASSED / total if total > 0 else 0
    
    print("\n" + "=" * 78)
    print(f"  GAUNTLET COMPLETE: {PASSED}/{total} tests passed ({pct:.1f}%)")
    print("=" * 78)
    
    if FAILED > 0:
        print(f"  ⚠️  Failed tests: {', '.join(FAILED_TESTS)}")
        print()
        return 1
    else:
        print("  🏆 LAYER 26 QTT-GA: ALL TESTS PASSED")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
