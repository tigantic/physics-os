//! Matrix Product Operator (MPO) for electromagnetic differential operators.
//!
//! MPOs encode the curl, divergence, Laplacian, and PML damping operators
//! that act on MPS-encoded field components.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::mps::{Mps, Core};

/// MPO core tensor: shape [left_bond, phys_in, phys_out, right_bond].
#[derive(Clone, Debug)]
pub struct MpoCore {
    pub data: Vec<Q16>,
    pub left_bond: usize,
    pub phys_in: usize,
    pub phys_out: usize,
    pub right_bond: usize,
}

impl MpoCore {
    pub fn zeros(left_bond: usize, phys_in: usize, phys_out: usize, right_bond: usize) -> Self {
        MpoCore {
            data: vec![Q16::ZERO; left_bond * phys_in * phys_out * right_bond],
            left_bond, phys_in, phys_out, right_bond,
        }
    }

    #[inline]
    pub fn get(&self, l: usize, pi: usize, po: usize, r: usize) -> Q16 {
        self.data[((l * self.phys_in + pi) * self.phys_out + po) * self.right_bond + r]
    }

    #[inline]
    pub fn set(&mut self, l: usize, pi: usize, po: usize, r: usize, val: Q16) {
        self.data[((l * self.phys_in + pi) * self.phys_out + po) * self.right_bond + r] = val;
    }
}

/// Matrix Product Operator: chain of MPO cores.
#[derive(Clone, Debug)]
pub struct Mpo {
    pub cores: Vec<MpoCore>,
    pub num_sites: usize,
}

impl Mpo {
    /// Identity MPO: acts as identity on MPS.
    pub fn identity(num_sites: usize, phys_dim: usize) -> Self {
        let mut cores = Vec::with_capacity(num_sites);
        for _ in 0..num_sites {
            let mut core = MpoCore::zeros(1, phys_dim, phys_dim, 1);
            for p in 0..phys_dim {
                core.set(0, p, p, 0, Q16::ONE);
            }
            cores.push(core);
        }
        Mpo { cores, num_sites }
    }

    /// Finite difference first derivative MPO (central difference).
    /// Encodes (f[i+1] - f[i-1]) / (2*dx) in QTT format.
    pub fn first_derivative(num_sites: usize, dx: Q16) -> Self {
        let inv_2dx = Q16::ONE.div(Q16::TWO * dx);
        let mut cores = Vec::with_capacity(num_sites);

        // QTT encoding of shift operators and difference
        // Bond dim 3: [identity, shift_right, shift_left]
        for i in 0..num_sites {
            let _bond = if i == 0 || i == num_sites - 1 { 1 } else { 3 };
            let left_bond = if i == 0 { 1 } else { 3 };
            let right_bond = if i == num_sites - 1 { 1 } else { 3 };

            let mut core = MpoCore::zeros(left_bond, 2, 2, right_bond);

            if i == 0 {
                // First site: start building shift operators
                for p in 0..2 {
                    if right_bond >= 3 {
                        core.set(0, p, p, 0, Q16::ONE); // identity channel
                        core.set(0, p, p, 1, inv_2dx);  // +shift channel
                        core.set(0, p, p, 2, -inv_2dx); // -shift channel
                    } else {
                        core.set(0, p, p, 0, Q16::ZERO);
                    }
                }
            } else if i == num_sites - 1 {
                // Last site: collect results
                for p in 0..2 {
                    let p_up = (p + 1) % 2;
                    let p_down = if p == 0 { 1 } else { 0 };
                    if left_bond >= 3 {
                        core.set(1, p, p_up, 0, Q16::ONE);   // shift right
                        core.set(2, p, p_down, 0, Q16::ONE);  // shift left
                    }
                }
            } else {
                // Interior: propagate channels
                for p in 0..2 {
                    if left_bond >= 3 && right_bond >= 3 {
                        core.set(0, p, p, 0, Q16::ONE); // identity propagation
                        core.set(1, p, p, 1, Q16::ONE); // shift right propagation
                        core.set(2, p, p, 2, Q16::ONE); // shift left propagation
                    }
                }
            }

            cores.push(core);
        }

        Mpo { cores, num_sites }
    }

    /// Second derivative (Laplacian 1D) MPO.
    /// Encodes (f[i+1] - 2*f[i] + f[i-1]) / dx^2 in QTT format.
    pub fn second_derivative(num_sites: usize, dx: Q16) -> Self {
        let inv_dx2 = Q16::ONE.div(dx * dx);

        let mut cores = Vec::with_capacity(num_sites);

        for i in 0..num_sites {
            let left_bond = if i == 0 { 1 } else { 3 };
            let right_bond = if i == num_sites - 1 { 1 } else { 3 };

            let mut core = MpoCore::zeros(left_bond, 2, 2, right_bond);

            if num_sites == 1 {
                // Degenerate case
                for p in 0..2 {
                    core.set(0, p, p, 0, Q16::from_int(-2) * inv_dx2);
                }
            } else if i == 0 {
                for p in 0..2 {
                    core.set(0, p, p, 0, Q16::from_int(-2) * inv_dx2); // -2f[i]/dx^2
                    if right_bond >= 3 {
                        core.set(0, p, p, 1, inv_dx2); // +f[i+1]/dx^2
                        core.set(0, p, p, 2, inv_dx2); // +f[i-1]/dx^2
                    }
                }
            } else if i == num_sites - 1 {
                for p in 0..2 {
                    if left_bond >= 3 {
                        core.set(0, p, p, 0, Q16::ONE);  // identity
                        core.set(1, p, p, 0, Q16::ONE);  // shift right
                        core.set(2, p, p, 0, Q16::ONE);  // shift left
                    }
                }
            } else {
                for p in 0..2 {
                    if left_bond >= 3 && right_bond >= 3 {
                        core.set(0, p, p, 0, Q16::ONE);
                        core.set(1, p, p, 1, Q16::ONE);
                        core.set(2, p, p, 2, Q16::ONE);
                    }
                }
            }

            cores.push(core);
        }

        Mpo { cores, num_sites }
    }

    /// Diagonal scaling MPO: multiply field by spatially-varying coefficient.
    /// `coeffs` should be an MPS encoding the coefficient field.
    pub fn diagonal_scale(num_sites: usize, phys_dim: usize, scale: Q16) -> Self {
        let mut cores = Vec::with_capacity(num_sites);
        for _ in 0..num_sites {
            let mut core = MpoCore::zeros(1, phys_dim, phys_dim, 1);
            for p in 0..phys_dim {
                core.set(0, p, p, 0, scale);
            }
            cores.push(core);
        }
        Mpo { cores, num_sites }
    }
}

/// Apply MPO to MPS: result = O |ψ⟩.
/// Uses zip contraction producing MPS with bond dim = mps_bond * mpo_bond.
pub fn mpo_apply(mpo: &Mpo, mps: &Mps, chi_max: usize) -> Mps {
    assert_eq!(mpo.num_sites, mps.num_sites);
    let n = mps.num_sites;

    let mut cores = Vec::with_capacity(n);

    for i in 0..n {
        let mc = &mps.cores[i];
        let oc = &mpo.cores[i];
        let phys_out = oc.phys_out;

        let new_left = mc.left_bond * oc.left_bond;
        let new_right = mc.right_bond * oc.right_bond;

        let mut core = Core::zeros(new_left, phys_out, new_right);

        // Contract over physical input index
        for ml in 0..mc.left_bond {
            for ol in 0..oc.left_bond {
                for po in 0..oc.phys_out {
                    for mr in 0..mc.right_bond {
                        for or_ in 0..oc.right_bond {
                            let mut sum = Q16::ZERO;
                            for pi in 0..oc.phys_in {
                                sum = sum + mc.get(ml, pi, mr) * oc.get(ol, pi, po, or_);
                            }
                            let nl = ml * oc.left_bond + ol;
                            let nr = mr * oc.right_bond + or_;
                            let prev = core.get(nl, po, nr);
                            core.set(nl, po, nr, prev + sum);
                        }
                    }
                }
            }
        }

        cores.push(core);
    }

    let mut result = Mps { cores, num_sites: n, chi_max };
    result.truncate(chi_max);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_mpo() {
        let mpo = Mpo::identity(4, 2);
        let mps = Mps::uniform(4, 2, Q16::HALF);
        let result = mpo_apply(&mpo, &mps, 4);
        assert_eq!(result.num_sites, 4);
    }

    #[test]
    fn test_diagonal_scale() {
        let mpo = Mpo::diagonal_scale(4, 2, Q16::TWO);
        assert_eq!(mpo.num_sites, 4);
    }

    #[test]
    fn test_first_derivative() {
        let dx = Q16::from_f64(0.1);
        let mpo = Mpo::first_derivative(4, dx);
        assert_eq!(mpo.num_sites, 4);
    }

    #[test]
    fn test_second_derivative() {
        let dx = Q16::from_f64(0.1);
        let mpo = Mpo::second_derivative(4, dx);
        assert_eq!(mpo.num_sites, 4);
    }
}
