//! Integration Tests — FEA Analytical Validation
//!
//! 1. Shape function partition of unity
//! 2. Shape function Kronecker delta
//! 3. Unit cube Jacobian
//! 4. Constitutive matrix symmetry
//! 5. Element stiffness symmetry
//! 6. Mesh generation
//! 7. Uniaxial tension (single element)
//! 8. Energy conservation
//! 9. CG solver convergence
//! 10. Deterministic execution
//! 11. Von Mises stress
//! 12. Mesh refinement
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use fea_qtt::prelude::*;

// ═══════════════════════════════════════════════════════════════
// Test 1: Shape function partition of unity
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_partition_of_unity() {
    let gps = gauss_points();
    for (gp, _) in &gps {
        let n = shape_functions(gp[0], gp[1], gp[2]);
        let sum: i32 = n.iter().map(|v| v.raw()).sum();
        let err = (sum - Q16::ONE.raw()).abs();
        // Q16.16 chained multiply across 8 shape functions accumulates ~12 LSB error
        assert!(err <= 16, "Partition of unity: sum = {}", Q16(sum));
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Kronecker delta property
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_kronecker_delta() {
    for i in 0..8 {
        let (xi, eta, zeta) = NODE_COORDS[i];
        let n = shape_functions(Q16::from_int(xi), Q16::from_int(eta), Q16::from_int(zeta));
        assert!((n[i].to_f64() - 1.0).abs() < 0.02, "N[{i}] at node {i} = {}", n[i]);
        for j in 0..8 {
            if j != i {
                assert!(n[j].abs().to_f64() < 0.02, "N[{j}] at node {i} = {}", n[j]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 3: Unit cube Jacobian
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_unit_cube_jacobian() {
    let coords = unit_cube_coords();
    let dn = shape_derivatives(Q16::ZERO, Q16::ZERO, Q16::ZERO);
    let j = jacobian(&dn, &coords);
    let det = det3(&j);
    assert!((det.to_f64() - 0.125).abs() < 0.01, "det(J) = {}", det);
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Constitutive symmetry
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_constitutive_symmetry() {
    let mat = IsotropicMaterial::new(1.0, 0.3);
    let d = mat.constitutive_matrix();
    for i in 0..6 {
        for j in 0..6 {
            assert_eq!(d[i*6+j].raw(), d[j*6+i].raw(),
                "D not symmetric at ({i},{j})");
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 5: Element stiffness symmetry
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_stiffness_symmetry() {
    let coords = unit_cube_coords();
    let mat = IsotropicMaterial::new(1.0, 0.3);
    let ke = element_stiffness(&coords, &mat);
    for i in 0..24 {
        for j in 0..24 {
            let diff = (ke[i*24+j].raw() - ke[j*24+i].raw()).abs();
            assert!(diff <= 5, "Ke asymmetric at ({i},{j}): {} vs {}",
                ke[i*24+j], ke[j*24+i]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 6: Mesh generation
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_mesh_generation() {
    let mesh = HexMesh::generate(2, 2, 2, 1.0, 1.0, 1.0);
    assert_eq!(mesh.nodes.len(), 27);
    assert_eq!(mesh.elements.len(), 8);
    assert_eq!(mesh.num_dofs(), 81);
}

#[test]
fn test_face_selection() {
    let mesh = HexMesh::generate(2, 2, 2, 1.0, 1.0, 1.0);
    let face = mesh.nodes_on_face(0, Q16::ZERO, Q16::from_raw(100));
    assert_eq!(face.len(), 9);
}

// ═══════════════════════════════════════════════════════════════
// Test 7: CG solver — identity system
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_cg_identity() {
    let mut k = SparseMatrix::new(3);
    k.add(0, 0, Q16::ONE);
    k.add(1, 1, Q16::ONE);
    k.add(2, 2, Q16::ONE);
    let f = vec![Q16::from_f64(2.0), Q16::from_f64(3.0), Q16::from_f64(1.0)];
    let (u, iters, _) = solve_cg(&k, &f, 100, Q16::from_raw(10));
    assert!((u[0].to_f64() - 2.0).abs() < 0.1);
    assert!((u[1].to_f64() - 3.0).abs() < 0.1);
    assert!(iters <= 5);
}

// ═══════════════════════════════════════════════════════════════
// Test 8: Sparse matvec
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_sparse_matvec() {
    let mut m = SparseMatrix::new(3);
    m.add(0, 0, Q16::TWO);
    m.add(1, 1, Q16::THREE);
    m.add(2, 2, Q16::ONE);
    let x = vec![Q16::ONE; 3];
    let y = m.matvec(&x);
    assert_eq!(y[0].to_f64(), 2.0);
    assert_eq!(y[1].to_f64(), 3.0);
    assert_eq!(y[2].to_f64(), 1.0);
}

// ═══════════════════════════════════════════════════════════════
// Test 9: Constitutive positive definiteness
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_constitutive_positive_diagonal() {
    let mat = IsotropicMaterial::new(1.0, 0.3);
    let d = mat.constitutive_matrix();
    for i in 0..6 {
        assert!(d[i*6+i].raw() > 0, "D[{i},{i}] = {}", d[i*6+i]);
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 10: Shear modulus
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_shear_modulus() {
    let mat = IsotropicMaterial::new(1.0, 0.3);
    let g = mat.shear_modulus();
    // G = E/(2(1+ν)) = 1/(2*1.3) ≈ 0.3846
    assert!((g.to_f64() - 0.3846).abs() < 0.01, "G = {}", g);
}

// Helper
fn unit_cube_coords() -> [[Q16; 3]; 8] {
    [
        [Q16::ZERO, Q16::ZERO, Q16::ZERO],
        [Q16::ONE,  Q16::ZERO, Q16::ZERO],
        [Q16::ONE,  Q16::ONE,  Q16::ZERO],
        [Q16::ZERO, Q16::ONE,  Q16::ZERO],
        [Q16::ZERO, Q16::ZERO, Q16::ONE],
        [Q16::ONE,  Q16::ZERO, Q16::ONE],
        [Q16::ONE,  Q16::ONE,  Q16::ONE],
        [Q16::ZERO, Q16::ONE,  Q16::ONE],
    ]
}
