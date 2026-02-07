//! Global Stiffness Assembly and Conjugate Gradient Solver
//!
//! Assembles element stiffness matrices into global system Ku = F
//! and solves via Conjugate Gradient iteration in Q16.16.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::mesh::HexMesh;
use crate::element::element_stiffness;
use crate::material::MaterialMap;

/// Sparse matrix in COO format (for assembly), converted to CSR for solve.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<Q16>,
    pub n: usize,
}

impl SparseMatrix {
    pub fn new(n: usize) -> Self {
        SparseMatrix { rows: Vec::new(), cols: Vec::new(), vals: Vec::new(), n }
    }

    /// Add value to (row, col). Duplicate entries accumulated during matvec.
    pub fn add(&mut self, row: usize, col: usize, val: Q16) {
        if val.raw() != 0 {
            self.rows.push(row);
            self.cols.push(col);
            self.vals.push(val);
        }
    }

    /// Matrix-vector product y = Ax.
    pub fn matvec(&self, x: &[Q16]) -> Vec<Q16> {
        let mut y = vec![Q16::ZERO; self.n];
        for i in 0..self.rows.len() {
            y[self.rows[i]] = y[self.rows[i]] + self.vals[i] * x[self.cols[i]];
        }
        y
    }

    /// Number of nonzeros.
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }
}

/// Boundary condition: fixed DOF.
#[derive(Clone, Debug)]
pub struct DirichletBC {
    pub dof: usize,
    pub value: Q16,
}

/// Point load on a DOF.
#[derive(Clone, Debug)]
pub struct PointLoad {
    pub dof: usize,
    pub value: Q16,
}

/// Distributed pressure on a face.
#[derive(Clone, Debug)]
pub struct Pressure {
    pub node_ids: Vec<usize>,
    pub direction: usize, // 0=x, 1=y, 2=z
    pub magnitude: Q16,
}

/// Assemble global stiffness matrix from mesh and materials.
pub fn assemble_stiffness(mesh: &HexMesh, materials: &MaterialMap) -> SparseMatrix {
    let ndof = mesh.num_dofs();
    let mut k_global = SparseMatrix::new(ndof);

    for elem in &mesh.elements {
        let coords = mesh.element_coords(elem);
        let mat = materials.get(elem.id);
        let ke = element_stiffness(&coords, mat);

        // Scatter element stiffness to global
        for i in 0..8 {
            let gdof_i = HexMesh::node_dofs(elem.nodes[i]);
            for j in 0..8 {
                let gdof_j = HexMesh::node_dofs(elem.nodes[j]);
                for di in 0..3 {
                    for dj in 0..3 {
                        let val = ke[(i*3+di) * 24 + (j*3+dj)];
                        if val.raw() != 0 {
                            k_global.add(gdof_i[di], gdof_j[dj], val);
                        }
                    }
                }
            }
        }
    }

    k_global
}

/// Assemble force vector from point loads and pressures.
pub fn assemble_forces(ndof: usize, loads: &[PointLoad], pressures: &[Pressure]) -> Vec<Q16> {
    let mut f = vec![Q16::ZERO; ndof];

    for load in loads {
        f[load.dof] = f[load.dof] + load.value;
    }

    for pressure in pressures {
        // Distribute pressure equally to nodes (simplified)
        let n = pressure.node_ids.len();
        if n == 0 { continue; }
        let per_node = pressure.magnitude.div(Q16::from_int(n as i32));
        for &nid in &pressure.node_ids {
            let dof = nid * 3 + pressure.direction;
            f[dof] = f[dof] + per_node;
        }
    }

    f
}

/// Apply Dirichlet BCs via penalty method.
/// Modifies K and F in-place (conceptually; adds penalty to diagonal).
pub fn apply_dirichlet_penalty(
    k: &mut SparseMatrix,
    f: &mut Vec<Q16>,
    bcs: &[DirichletBC],
) {
    let penalty = Q16::from_f64(100.0); // Tuned for Q16.16 range stability

    for bc in bcs {
        k.add(bc.dof, bc.dof, penalty);
        f[bc.dof] = f[bc.dof] + penalty * bc.value;
    }
}

/// Conjugate Gradient solver for Ku = F.
/// Returns (displacement, iterations, final_residual).
pub fn solve_cg(
    k: &SparseMatrix,
    f: &[Q16],
    max_iter: usize,
    tol: Q16,
) -> (Vec<Q16>, usize, Q16) {
    let n = k.n;
    let mut u = vec![Q16::ZERO; n]; // initial guess = 0
    let mut r = f.to_vec();         // r = F - K·u = F (since u=0)
    let mut p = r.clone();           // p = r

    let mut r_dot_r = dot(&r, &r);

    for iter in 0..max_iter {
        let ap = k.matvec(&p); // Ap
        let p_ap = dot(&p, &ap);

        if p_ap.raw() == 0 { break; }

        let alpha = r_dot_r.div(p_ap);

        // u = u + α·p
        for i in 0..n {
            u[i] = u[i] + alpha * p[i];
        }

        // r = r - α·Ap
        for i in 0..n {
            r[i] = r[i] - alpha * ap[i];
        }

        let r_dot_r_new = dot(&r, &r);

        // Check convergence
        let residual = r_dot_r_new.sqrt();
        if residual.raw() <= tol.raw() {
            return (u, iter + 1, residual);
        }

        let beta = r_dot_r_new.div(r_dot_r);

        // p = r + β·p
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        r_dot_r = r_dot_r_new;
    }

    let final_res = dot(&r, &r).sqrt();
    (u, max_iter, final_res)
}

/// Dot product in Q16.16.
fn dot(a: &[Q16], b: &[Q16]) -> Q16 {
    let mut sum = Q16::ZERO;
    for i in 0..a.len() {
        sum = sum + a[i] * b[i];
    }
    sum
}

/// Strain energy: U = ½ uᵀKu = ½ Fᵀu.
pub fn strain_energy(f: &[Q16], u: &[Q16]) -> Q16 {
    Q16::HALF * dot(f, u)
}

/// External work: W = Fᵀu (should equal 2×strain_energy for linear elastic).
pub fn external_work(f: &[Q16], u: &[Q16]) -> Q16 {
    dot(f, u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matvec() {
        let mut m = SparseMatrix::new(3);
        m.add(0, 0, Q16::TWO);
        m.add(1, 1, Q16::THREE);
        m.add(2, 2, Q16::ONE);

        let x = vec![Q16::ONE, Q16::ONE, Q16::ONE];
        let y = m.matvec(&x);
        assert_eq!(y[0].to_f64(), 2.0);
        assert_eq!(y[1].to_f64(), 3.0);
        assert_eq!(y[2].to_f64(), 1.0);
    }

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
}
