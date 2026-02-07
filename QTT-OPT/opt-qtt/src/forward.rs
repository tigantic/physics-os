//! 2D Forward Elasticity Solver (Quad4)
//!
//! Plane stress, bilinear quadrilateral elements, 2×2 Gauss quadrature.
//! Inner solver for topology optimization loop.
//!
//! Element stiffness: Ke = t ∫ BᵀDB dA
//! Global system: Ku = F
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// 2D structured quad mesh.
#[derive(Clone, Debug)]
pub struct Mesh2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: Q16,
    pub dy: Q16,
}

impl Mesh2D {
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        Mesh2D {
            nx, ny,
            dx: Q16::from_f64(lx / nx as f64),
            dy: Q16::from_f64(ly / ny as f64),
        }
    }

    pub fn num_nodes(&self) -> usize { (self.nx + 1) * (self.ny + 1) }
    pub fn num_elements(&self) -> usize { self.nx * self.ny }
    pub fn num_dofs(&self) -> usize { self.num_nodes() * 2 }

    #[inline]
    pub fn node_id(&self, i: usize, j: usize) -> usize { j * (self.nx + 1) + i }

    /// Element connectivity: bottom-left, bottom-right, top-right, top-left.
    pub fn element_nodes(&self, ex: usize, ey: usize) -> [usize; 4] {
        [self.node_id(ex, ey), self.node_id(ex+1, ey),
         self.node_id(ex+1, ey+1), self.node_id(ex, ey+1)]
    }

    /// Element DOF indices [8].
    pub fn element_dofs(&self, ex: usize, ey: usize) -> [usize; 8] {
        let ns = self.element_nodes(ex, ey);
        [ns[0]*2, ns[0]*2+1, ns[1]*2, ns[1]*2+1,
         ns[2]*2, ns[2]*2+1, ns[3]*2, ns[3]*2+1]
    }

    /// Boundary node IDs.
    pub fn boundary_nodes(&self, side: &str) -> Vec<usize> {
        match side {
            "left"   => (0..=self.ny).map(|j| self.node_id(0, j)).collect(),
            "right"  => (0..=self.ny).map(|j| self.node_id(self.nx, j)).collect(),
            "bottom" => (0..=self.nx).map(|i| self.node_id(i, 0)).collect(),
            "top"    => (0..=self.nx).map(|i| self.node_id(i, self.ny)).collect(),
            _ => vec![],
        }
    }
}

/// Plane stress constitutive matrix D[3×3].
pub fn plane_stress_d(nu: Q16) -> [Q16; 9] {
    // D = (1/(1-ν²)) [1  ν  0; ν  1  0; 0  0  (1-ν)/2]
    // Factor out E (applied via SIMP per-element)
    let denom = Q16::ONE - nu * nu;
    let factor = Q16::ONE.div(denom);
    let d11 = factor;
    let d12 = factor * nu;
    let d33 = factor * (Q16::ONE - nu).div(Q16::TWO);
    let z = Q16::ZERO;
    [d11, d12, z, d12, d11, z, z, z, d33]
}

/// Unit element stiffness Ke0[8×8] for unit-E square element.
pub fn unit_element_stiffness(dx: Q16, dy: Q16, nu: Q16) -> [Q16; 64] {
    let d = plane_stress_d(nu);
    let gp = Q16::from_f64(0.5773502691896258);
    let mgp = -gp;
    let gauss = [(mgp, mgp), (gp, mgp), (gp, gp), (mgp, gp)];

    let mut ke = [Q16::ZERO; 64];
    let inv_dx = Q16::TWO.div(dx);
    let inv_dy = Q16::TWO.div(dy);
    let jac_det = (dx * dy).div(Q16::from_int(4));

    for &(xi, eta) in &gauss {
        // dN/dξ and dN/dη
        let q4 = Q16::from_ratio(1, 4);
        let dn_dxi = [
            -q4 * (Q16::ONE - eta), q4 * (Q16::ONE - eta),
             q4 * (Q16::ONE + eta), -q4 * (Q16::ONE + eta),
        ];
        let dn_deta = [
            -q4 * (Q16::ONE - xi), -q4 * (Q16::ONE + xi),
             q4 * (Q16::ONE + xi),  q4 * (Q16::ONE - xi),
        ];

        // B[3×8]
        let mut b = [Q16::ZERO; 24];
        for i in 0..4 {
            let dndx = dn_dxi[i] * inv_dx;
            let dndy = dn_deta[i] * inv_dy;
            b[0*8 + i*2]     = dndx;  // εxx
            b[1*8 + i*2 + 1] = dndy;  // εyy
            b[2*8 + i*2]     = dndy;  // γxy
            b[2*8 + i*2 + 1] = dndx;
        }

        // DB[3×8]
        let mut db = [Q16::ZERO; 24];
        for r in 0..3 {
            for c in 0..8 {
                let mut s = Q16::ZERO;
                for k in 0..3 { s = s + d[r*3+k] * b[k*8+c]; }
                db[r*8+c] = s;
            }
        }

        // Ke += BᵀDB |J|
        for r in 0..8 {
            for c in 0..8 {
                let mut s = Q16::ZERO;
                for k in 0..3 { s = s + b[k*8+r] * db[k*8+c]; }
                ke[r*8+c] = ke[r*8+c] + s * jac_det;
            }
        }
    }

    ke
}

/// Sparse COO matrix.
#[derive(Clone, Debug)]
pub struct SparseMat {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<Q16>,
    pub n: usize,
}

impl SparseMat {
    pub fn new(n: usize) -> Self { SparseMat { rows: vec![], cols: vec![], vals: vec![], n } }
    pub fn clear(&mut self) { self.rows.clear(); self.cols.clear(); self.vals.clear(); }

    pub fn add(&mut self, r: usize, c: usize, v: Q16) {
        if v.raw() != 0 { self.rows.push(r); self.cols.push(c); self.vals.push(v); }
    }

    pub fn matvec(&self, x: &[Q16]) -> Vec<Q16> {
        let mut y = vec![Q16::ZERO; self.n];
        for i in 0..self.rows.len() {
            y[self.rows[i]] = y[self.rows[i]] + self.vals[i] * x[self.cols[i]];
        }
        y
    }
}

/// Assemble K with SIMP-modified densities.
/// E(ρ) = E_min + ρ^p · (1 - E_min)
pub fn assemble_stiffness(
    mesh: &Mesh2D,
    ke0: &[Q16; 64],
    densities: &[Q16],
    penal: u32,
    e_min: Q16,
    k: &mut SparseMat,
) {
    k.clear();
    for ey in 0..mesh.ny {
        for ex in 0..mesh.nx {
            let eid = ey * mesh.nx + ex;
            let rho_p = densities[eid].powi(penal);
            let e_eff = e_min + rho_p * (Q16::ONE - e_min);

            let dofs = mesh.element_dofs(ex, ey);
            for i in 0..8 {
                for j in 0..8 {
                    let val = e_eff * ke0[i*8+j];
                    if val.raw() != 0 { k.add(dofs[i], dofs[j], val); }
                }
            }
        }
    }
}

/// Apply penalty BCs.
pub fn apply_bcs(k: &mut SparseMat, _f: &mut [Q16], fixed_dofs: &[usize]) {
    let penalty = Q16::from_f64(100.0);
    for &dof in fixed_dofs { k.add(dof, dof, penalty); }
}

/// CG solver with 64-bit accumulation.
pub fn solve_cg(k: &SparseMat, f: &[Q16], max_iter: usize, tol_sq: i64) -> (Vec<Q16>, usize) {
    let n = k.n;
    let mut u = vec![Q16::ZERO; n];
    let mut r = f.to_vec();
    let mut p = r.clone();

    let dot64 = |a: &[Q16], b: &[Q16]| -> i64 {
        a.iter().zip(b.iter()).map(|(x,y)| (x.0 as i64 * y.0 as i64) >> 16).sum::<i64>()
    };

    let mut rr = dot64(&r, &r);

    for iter in 0..max_iter {
        let ap = k.matvec(&p);
        let pap = dot64(&p, &ap);
        if pap == 0 { return (u, iter); }
        let alpha = Q16(((rr * 65536i64) / pap) as i32);

        for i in 0..n {
            u[i] = u[i] + alpha * p[i];
            r[i] = r[i] - alpha * ap[i];
        }

        let rr_new = dot64(&r, &r);
        if rr_new.abs() < tol_sq { return (u, iter + 1); }
        let beta = Q16(((rr_new * 65536i64) / rr.max(1)) as i32);
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        rr = rr_new;
    }
    (u, max_iter)
}

/// Compute element compliance: ce = ue^T Ke0 ue.
pub fn element_compliance(ke0: &[Q16; 64], u: &[Q16], dofs: &[usize; 8]) -> Q16 {
    let mut ce = Q16::ZERO;
    for i in 0..8 {
        for j in 0..8 {
            ce = ce + u[dofs[i]] * ke0[i*8+j] * u[dofs[j]];
        }
    }
    ce
}
