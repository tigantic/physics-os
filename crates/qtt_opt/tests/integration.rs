//! Integration Tests — Optimization & Inverse Problems
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use opt_qtt::prelude::*;

#[test]
fn test_powi() {
    assert!((Q16::HALF.powi(3).to_f64() - 0.125).abs() < 0.01);
    assert_eq!(Q16::ONE.powi(5), Q16::ONE);
    assert_eq!(Q16::ZERO.powi(3), Q16::ZERO);
}

#[test]
fn test_clamp() {
    let v = Q16::from_f64(1.5);
    let lo = Q16::from_f64(0.1);
    let hi = Q16::from_f64(0.9);
    assert!((v.clamp(lo, hi).to_f64() - 0.9).abs() < 0.01);
    assert!((Q16::from_f64(-0.5).clamp(lo, hi).to_f64() - 0.1).abs() < 0.01);
}

#[test]
fn test_plane_stress_d_symmetry() {
    let nu = Q16::from_f64(0.3);
    let d = plane_stress_d(nu);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(d[i*3+j].raw(), d[j*3+i].raw(),
                "D not symmetric at ({i},{j})");
        }
    }
}

#[test]
fn test_element_stiffness_symmetry() {
    let ke = unit_element_stiffness(Q16::ONE, Q16::ONE, Q16::from_f64(0.3));
    for i in 0..8 {
        for j in 0..8 {
            let diff = (ke[i*8+j].raw() - ke[j*8+i].raw()).abs();
            assert!(diff <= 5, "Ke asymmetric at ({i},{j}): {} vs {}",
                ke[i*8+j], ke[j*8+i]);
        }
    }
}

#[test]
fn test_element_stiffness_positive_diagonal() {
    let ke = unit_element_stiffness(Q16::ONE, Q16::ONE, Q16::from_f64(0.3));
    for i in 0..8 {
        assert!(ke[i*8+i].raw() > 0, "Ke[{i},{i}] = {}", ke[i*8+i]);
    }
}

#[test]
fn test_mesh2d_generation() {
    let mesh = Mesh2D::new(4, 3, 4.0, 3.0);
    assert_eq!(mesh.num_nodes(), 20);
    assert_eq!(mesh.num_elements(), 12);
    assert_eq!(mesh.num_dofs(), 40);
}

#[test]
fn test_mesh_boundary_nodes() {
    let mesh = Mesh2D::new(4, 3, 4.0, 3.0);
    let left = mesh.boundary_nodes("left");
    assert_eq!(left.len(), 4); // ny+1
    let bottom = mesh.boundary_nodes("bottom");
    assert_eq!(bottom.len(), 5); // nx+1
}

#[test]
fn test_sparse_matvec() {
    let mut m = SparseMat::new(3);
    m.add(0, 0, Q16::TWO);
    m.add(1, 1, Q16::THREE);
    m.add(2, 2, Q16::ONE);
    let x = vec![Q16::ONE; 3];
    let y = m.matvec(&x);
    assert_eq!(y[0].to_f64(), 2.0);
    assert_eq!(y[1].to_f64(), 3.0);
    assert_eq!(y[2].to_f64(), 1.0);
}

#[test]
fn test_cg_identity() {
    let mut k = SparseMat::new(3);
    k.add(0, 0, Q16::ONE);
    k.add(1, 1, Q16::ONE);
    k.add(2, 2, Q16::ONE);
    let f = vec![Q16::from_f64(2.0), Q16::from_f64(3.0), Q16::from_f64(1.0)];
    let (u, iters) = solve_cg(&k, &f, 100, 10);
    assert!((u[0].to_f64() - 2.0).abs() < 0.1);
    assert!(iters <= 5);
}

#[test]
fn test_simp_interpolation() {
    let rho = Q16::HALF;
    let e_min = Q16::from_f64(0.001);
    let rho_p = rho.powi(3); // 0.125
    let e_eff = e_min + rho_p * (Q16::ONE - e_min);
    assert!((e_eff.to_f64() - 0.126).abs() < 0.01,
        "E(0.5) = {}", e_eff);
}

#[test]
fn test_sensitivity_filter_construction() {
    let mesh = Mesh2D::new(4, 3, 4.0, 3.0);
    let filter = SensitivityFilter::new(&mesh, 1.5);
    // Every element should have at least itself as neighbor
    for e in 0..mesh.num_elements() {
        assert!(!filter.neighbors[e].is_empty());
        assert!(filter.weight_sums[e].raw() > 0);
    }
}

#[test]
fn test_volume_fraction() {
    let rho = vec![Q16::HALF; 10];
    let vf = volume_fraction(&rho);
    assert!((vf.to_f64() - 0.5).abs() < 0.01);
}

#[test]
fn test_topopt_default_runs() {
    let config = TopOptConfig {
        nx: 4, ny: 2,
        volume_fraction: 0.5,
        penal: 3,
        rmin: 1.5,
        max_iter: 5,
        tol: 0.01,
        e_min: 0.001,
        move_limit: 0.2,
        eta: 0.5,
    };
    let result = optimize(&config);
    assert!(result.iterations > 0);
    assert_eq!(result.densities.len(), 8);
    assert!(!result.compliance.is_empty());
}

#[test]
fn test_adjoint_self_adjoint() {
    let u = vec![Q16::ONE, Q16::TWO, Q16::from_f64(0.5)];
    let adj = AdjointState::self_adjoint(&u);
    assert!(adj.is_self_adjoint);
    assert_eq!(adj.lambda[0], Q16::NEG_ONE);
    assert_eq!(adj.lambda[1].to_f64(), -2.0);
}

#[test]
fn test_poisson_forward() {
    let model = Poisson1DModel::new(4, 1.0);
    let kappa = vec![Q16::ONE; 4];
    let u = model.solve(&kappa);
    assert_eq!(u.len(), 5); // 4 elements → 5 nodes
    // BCs: u[0] = u[4] = 0
    assert!(u[0].to_f64().abs() < 0.01);
    assert!(u[4].to_f64().abs() < 0.01);
    // Interior should be positive (source > 0, kappa > 0)
    assert!(u[2].to_f64() > 0.0);
}
