//! 8-Node Hexahedral Element (Hex8)
//!
//! Isoparametric element with trilinear shape functions.
//! Natural coordinates (ξ, η, ζ) ∈ [-1, 1]³.
//!
//! Shape functions:
//!   N_i = ⅛(1 + ξ_i·ξ)(1 + η_i·η)(1 + ζ_i·ζ)
//!
//! Node ordering (standard right-hand):
//!   Bottom: 0(-,-,-) 1(+,-,-) 2(+,+,-) 3(-,+,-)
//!   Top:    4(-,-,+) 5(+,-,+) 6(+,+,+) 7(-,+,+)
//!
//! Integration: 2×2×2 Gauss quadrature (8 points).
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::material::IsotropicMaterial;

/// Node natural coordinates for Hex8 [-1,+1].
pub const NODE_COORDS: [(i32, i32, i32); 8] = [
    (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
    (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
];

/// Gauss point coordinate: 1/√3.
const GP: f64 = 0.5773502691896258;

/// 2×2×2 Gauss points (ξ, η, ζ) and weights.
pub fn gauss_points() -> Vec<([Q16; 3], Q16)> {
    let g = Q16::from_f64(GP);
    let mg = Q16::from_f64(-GP);
    let w = Q16::ONE; // weight = 1.0 for each point in 2×2×2

    vec![
        ([mg, mg, mg], w), ([g, mg, mg], w),
        ([mg,  g, mg], w), ([g,  g, mg], w),
        ([mg, mg,  g], w), ([g, mg,  g], w),
        ([mg,  g,  g], w), ([g,  g,  g], w),
    ]
}

/// Evaluate shape functions at natural coordinates (ξ, η, ζ).
/// Returns N[8].
pub fn shape_functions(xi: Q16, eta: Q16, zeta: Q16) -> [Q16; 8] {
    let eighth = Q16::from_ratio(1, 8);
    let mut n = [Q16::ZERO; 8];

    for i in 0..8 {
        let (xi_i, eta_i, zeta_i) = NODE_COORDS[i];
        let xi_q = Q16::from_int(xi_i);
        let eta_q = Q16::from_int(eta_i);
        let zeta_q = Q16::from_int(zeta_i);

        n[i] = eighth * (Q16::ONE + xi_q * xi) * (Q16::ONE + eta_q * eta) * (Q16::ONE + zeta_q * zeta);
    }
    n
}

/// Shape function derivatives dN/dξ, dN/dη, dN/dζ at (ξ,η,ζ).
/// Returns dN[3][8]: [dN/dξ, dN/dη, dN/dζ].
pub fn shape_derivatives(xi: Q16, eta: Q16, zeta: Q16) -> [[Q16; 8]; 3] {
    let eighth = Q16::from_ratio(1, 8);
    let mut dn = [[Q16::ZERO; 8]; 3];

    for i in 0..8 {
        let (xi_i, eta_i, zeta_i) = NODE_COORDS[i];
        let xi_q = Q16::from_int(xi_i);
        let eta_q = Q16::from_int(eta_i);
        let zeta_q = Q16::from_int(zeta_i);

        let a = Q16::ONE + xi_q * xi;
        let b = Q16::ONE + eta_q * eta;
        let c = Q16::ONE + zeta_q * zeta;

        dn[0][i] = eighth * xi_q * b * c;      // dN/dξ
        dn[1][i] = eighth * a * eta_q * c;      // dN/dη
        dn[2][i] = eighth * a * b * zeta_q;     // dN/dζ
    }
    dn
}

/// Compute Jacobian matrix J[3][3] at a Gauss point.
/// J_ij = Σ (dN_k/dξ_i) * x_k_j
/// `node_coords`: [8][3] physical coordinates of element nodes.
pub fn jacobian(dn: &[[Q16; 8]; 3], node_coords: &[[Q16; 3]; 8]) -> [[Q16; 3]; 3] {
    let mut j = [[Q16::ZERO; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let mut sum = Q16::ZERO;
            for k in 0..8 {
                sum = sum + dn[r][k] * node_coords[k][c];
            }
            j[r][c] = sum;
        }
    }
    j
}

/// Determinant of 3×3 matrix.
pub fn det3(m: &[[Q16; 3]; 3]) -> Q16 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of 3×3 matrix (cofactor method).
pub fn inv3(m: &[[Q16; 3]; 3]) -> [[Q16; 3]; 3] {
    let d = det3(m);
    if d.raw() == 0 { return [[Q16::ZERO; 3]; 3]; }

    let mut inv = [[Q16::ZERO; 3]; 3];

    inv[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]).div(d);
    inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]).div(d);
    inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]).div(d);
    inv[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]).div(d);
    inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]).div(d);
    inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]).div(d);
    inv[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]).div(d);
    inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]).div(d);
    inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]).div(d);

    inv
}

/// Strain-displacement matrix B[6][24] at a Gauss point.
/// B relates nodal displacements to strain: ε = B·u
/// dn_dx: shape function derivatives in physical coordinates [3][8].
pub fn b_matrix(dn_dx: &[[Q16; 8]; 3]) -> Vec<Q16> {
    // B is 6×24 (6 strain components, 8 nodes × 3 DOFs)
    let mut b = vec![Q16::ZERO; 6 * 24];

    for i in 0..8 {
        let col = i * 3;
        // εxx = ∂u/∂x
        b[0 * 24 + col + 0] = dn_dx[0][i];
        // εyy = ∂v/∂y
        b[1 * 24 + col + 1] = dn_dx[1][i];
        // εzz = ∂w/∂z
        b[2 * 24 + col + 2] = dn_dx[2][i];
        // γxy = ∂u/∂y + ∂v/∂x
        b[3 * 24 + col + 0] = dn_dx[1][i];
        b[3 * 24 + col + 1] = dn_dx[0][i];
        // γyz = ∂v/∂z + ∂w/∂y
        b[4 * 24 + col + 1] = dn_dx[2][i];
        b[4 * 24 + col + 2] = dn_dx[1][i];
        // γxz = ∂u/∂z + ∂w/∂x
        b[5 * 24 + col + 0] = dn_dx[2][i];
        b[5 * 24 + col + 2] = dn_dx[0][i];
    }

    b
}

/// Compute element stiffness matrix Ke[24×24] by numerical integration.
/// Ke = ∫ BᵀDB |J| dV ≈ Σ BᵀDB |J| w (2×2×2 Gauss)
pub fn element_stiffness(
    node_coords: &[[Q16; 3]; 8],
    material: &IsotropicMaterial,
) -> Vec<Q16> {
    let d = material.constitutive_matrix();
    let gps = gauss_points();
    let mut ke = vec![Q16::ZERO; 24 * 24];

    for (gp, weight) in &gps {
        let dn_nat = shape_derivatives(gp[0], gp[1], gp[2]);
        let jac = jacobian(&dn_nat, node_coords);
        let det_j = det3(&jac);
        let jac_inv = inv3(&jac);

        // Transform derivatives to physical coordinates: dN/dx = J^{-1} · dN/dξ
        let mut dn_dx = [[Q16::ZERO; 8]; 3];
        for i in 0..3 {
            for k in 0..8 {
                let mut sum = Q16::ZERO;
                for j in 0..3 {
                    sum = sum + jac_inv[i][j] * dn_nat[j][k];
                }
                dn_dx[i][k] = sum;
            }
        }

        let b = b_matrix(&dn_dx);

        // Ke += BᵀDB · |J| · w
        // First compute DB [6×24]
        let mut db = vec![Q16::ZERO; 6 * 24];
        for r in 0..6 {
            for c in 0..24 {
                let mut sum = Q16::ZERO;
                for k in 0..6 {
                    sum = sum + d[r * 6 + k] * b[k * 24 + c];
                }
                db[r * 24 + c] = sum;
            }
        }

        // Then Ke += Bᵀ(DB) · |J| · w
        let scale = det_j.abs() * *weight;
        for r in 0..24 {
            for c in 0..24 {
                let mut sum = Q16::ZERO;
                for k in 0..6 {
                    sum = sum + b[k * 24 + r] * db[k * 24 + c];
                }
                ke[r * 24 + c] = ke[r * 24 + c] + sum * scale;
            }
        }
    }

    ke
}

/// Recover stress σ = D·B·u at element centroid (ξ=η=ζ=0).
pub fn element_stress(
    node_coords: &[[Q16; 3]; 8],
    material: &IsotropicMaterial,
    displacements: &[Q16; 24],
) -> [Q16; 6] {
    let d = material.constitutive_matrix();
    let dn_nat = shape_derivatives(Q16::ZERO, Q16::ZERO, Q16::ZERO);
    let jac = jacobian(&dn_nat, node_coords);
    let jac_inv = inv3(&jac);

    let mut dn_dx = [[Q16::ZERO; 8]; 3];
    for i in 0..3 {
        for k in 0..8 {
            let mut sum = Q16::ZERO;
            for j in 0..3 { sum = sum + jac_inv[i][j] * dn_nat[j][k]; }
            dn_dx[i][k] = sum;
        }
    }

    let b = b_matrix(&dn_dx);

    // ε = B·u
    let mut strain = [Q16::ZERO; 6];
    for i in 0..6 {
        let mut sum = Q16::ZERO;
        for j in 0..24 {
            sum = sum + b[i * 24 + j] * displacements[j];
        }
        strain[i] = sum;
    }

    // σ = D·ε
    let mut stress = [Q16::ZERO; 6];
    for i in 0..6 {
        let mut sum = Q16::ZERO;
        for j in 0..6 {
            sum = sum + d[i * 6 + j] * strain[j];
        }
        stress[i] = sum;
    }

    stress
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_functions_partition_of_unity() {
        let gps = gauss_points();
        for (gp, _) in &gps {
            let n = shape_functions(gp[0], gp[1], gp[2]);
            let sum: i32 = n.iter().map(|v| v.raw()).sum();
            let err = (sum - Q16::ONE.raw()).abs();
            // Q16.16 chained multiply across 8 shape functions accumulates ~12 LSB error
            assert!(err <= 16, "Partition of unity failed: sum={}", Q16(sum));
        }
    }

    #[test]
    fn test_shape_function_at_node() {
        for i in 0..8 {
            let (xi, eta, zeta) = NODE_COORDS[i];
            let n = shape_functions(Q16::from_int(xi), Q16::from_int(eta), Q16::from_int(zeta));
            // N_i should be 1 at node i, 0 at others
            assert!((n[i].to_f64() - 1.0).abs() < 0.01, "N[{}] at node {} = {}", i, i, n[i]);
            for j in 0..8 {
                if j != i {
                    assert!(n[j].abs().to_f64() < 0.01, "N[{}] at node {} = {}", j, i, n[j]);
                }
            }
        }
    }

    #[test]
    fn test_unit_cube_jacobian() {
        // Unit cube [0,1]³
        let coords: [[Q16; 3]; 8] = [
            [Q16::ZERO, Q16::ZERO, Q16::ZERO],
            [Q16::ONE,  Q16::ZERO, Q16::ZERO],
            [Q16::ONE,  Q16::ONE,  Q16::ZERO],
            [Q16::ZERO, Q16::ONE,  Q16::ZERO],
            [Q16::ZERO, Q16::ZERO, Q16::ONE],
            [Q16::ONE,  Q16::ZERO, Q16::ONE],
            [Q16::ONE,  Q16::ONE,  Q16::ONE],
            [Q16::ZERO, Q16::ONE,  Q16::ONE],
        ];

        let dn = shape_derivatives(Q16::ZERO, Q16::ZERO, Q16::ZERO);
        let j = jacobian(&dn, &coords);
        let det = det3(&j);

        // det(J) for unit cube mapped from [-1,1]³ to [0,1]³ = (0.5)³ = 0.125
        assert!((det.to_f64() - 0.125).abs() < 0.01, "det(J) = {}", det);
    }

    #[test]
    fn test_stiffness_symmetry() {
        let coords: [[Q16; 3]; 8] = [
            [Q16::ZERO, Q16::ZERO, Q16::ZERO],
            [Q16::ONE,  Q16::ZERO, Q16::ZERO],
            [Q16::ONE,  Q16::ONE,  Q16::ZERO],
            [Q16::ZERO, Q16::ONE,  Q16::ZERO],
            [Q16::ZERO, Q16::ZERO, Q16::ONE],
            [Q16::ONE,  Q16::ZERO, Q16::ONE],
            [Q16::ONE,  Q16::ONE,  Q16::ONE],
            [Q16::ZERO, Q16::ONE,  Q16::ONE],
        ];

        let mat = IsotropicMaterial::new(10.0, 0.3);
        let ke = element_stiffness(&coords, &mat);

        // Check symmetry
        for i in 0..24 {
            for j in 0..24 {
                let diff = (ke[i*24+j].raw() - ke[j*24+i].raw()).abs();
                assert!(diff <= 4, "Ke not symmetric at ({},{}): {} vs {}",
                    i, j, ke[i*24+j], ke[j*24+i]);
            }
        }
    }
}
