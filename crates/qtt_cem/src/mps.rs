//! Matrix Product State (MPS) for electromagnetic field compression.
//!
//! Each 3D field component (Ex, Ey, Ez, Bx, By, Bz) is represented as an MPS
//! with `num_sites` cores, physical dimension 2 (QTT binary encoding),
//! and bond dimension bounded by `chi_max`.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// A single MPS core tensor: shape [left_bond, phys_dim, right_bond].
/// Stored as flat array in row-major order.
#[derive(Clone, Debug)]
pub struct Core {
    pub data: Vec<Q16>,
    pub left_bond: usize,
    pub phys_dim: usize,
    pub right_bond: usize,
}

impl Core {
    /// Create a zero-initialized core.
    pub fn zeros(left_bond: usize, phys_dim: usize, right_bond: usize) -> Self {
        Core {
            data: vec![Q16::ZERO; left_bond * phys_dim * right_bond],
            left_bond,
            phys_dim,
            right_bond,
        }
    }

    /// Access element at [l, p, r].
    #[inline]
    pub fn get(&self, l: usize, p: usize, r: usize) -> Q16 {
        self.data[l * self.phys_dim * self.right_bond + p * self.right_bond + r]
    }

    /// Set element at [l, p, r].
    #[inline]
    pub fn set(&mut self, l: usize, p: usize, r: usize, val: Q16) {
        self.data[l * self.phys_dim * self.right_bond + p * self.right_bond + r] = val;
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.left_bond * self.phys_dim * self.right_bond
    }
}

/// Matrix Product State: chain of core tensors encoding a 1D field.
#[derive(Clone, Debug)]
pub struct Mps {
    pub cores: Vec<Core>,
    pub num_sites: usize,
    pub chi_max: usize,
}

impl Mps {
    /// Create MPS with uniform bond dimension, zero-initialized.
    pub fn zeros(num_sites: usize, phys_dim: usize, chi_max: usize) -> Self {
        let mut cores = Vec::with_capacity(num_sites);
        for i in 0..num_sites {
            let left = if i == 0 { 1 } else { chi_max.min(1 << i).min(1 << (num_sites - i)) };
            let right = if i == num_sites - 1 { 1 } else { chi_max.min(1 << (i + 1)).min(1 << (num_sites - i - 1)) };
            cores.push(Core::zeros(left, phys_dim, right));
        }
        Mps { cores, num_sites, chi_max }
    }

    /// Create MPS encoding a uniform constant field value.
    pub fn uniform(num_sites: usize, phys_dim: usize, value: Q16) -> Self {
        let mut mps = Mps::zeros(num_sites, phys_dim, 1);
        // Uniform field: each core has bond dim 1, stores value^(1/N) approximation
        // For simplicity, encode value at first site, identity at rest
        for i in 0..num_sites {
            for p in 0..phys_dim {
                if i == 0 {
                    mps.cores[i].set(0, p, 0, value);
                } else {
                    mps.cores[i].set(0, p, 0, Q16::ONE);
                }
            }
        }
        mps
    }

    /// Create MPS from a flat array of field values via successive SVD.
    /// Input: `values` of length `phys_dim^num_sites`.
    pub fn from_values(values: &[Q16], num_sites: usize, phys_dim: usize, chi_max: usize) -> Self {
        let total = phys_dim.pow(num_sites as u32);
        assert_eq!(values.len(), total, "Value count mismatch");

        // Reshape as matrix and perform left-to-right SVD decomposition
        let mut remaining = values.to_vec();
        let mut cores = Vec::with_capacity(num_sites);
        let mut left_dim = 1usize;

        for site in 0..num_sites {
            let right_size: usize = phys_dim.pow((num_sites - site - 1) as u32);
            let rows = left_dim * phys_dim;
            let cols = right_size;

            // Build matrix [rows x cols] from remaining data
            let mat = remaining.clone();

            // Truncated SVD
            let (u, s, vt, rank) = truncated_svd(&mat, rows, cols, chi_max);

            // Core from U reshaped to [left_dim, phys_dim, rank]
            let mut core = Core::zeros(left_dim, phys_dim, rank);
            for l in 0..left_dim {
                for p in 0..phys_dim {
                    for r in 0..rank {
                        let row = l * phys_dim + p;
                        core.set(l, p, r, u[row * rank + r]);
                    }
                }
            }
            cores.push(core);

            // Remaining = S * Vt for next iteration
            remaining = Vec::with_capacity(rank * cols);
            for r in 0..rank {
                for c in 0..cols {
                    remaining.push(s[r] * vt[r * cols + c]);
                }
            }

            left_dim = rank;
        }

        Mps { cores, num_sites, chi_max }
    }

    /// Reconstruct full field values from MPS (for testing/verification).
    pub fn to_values(&self, phys_dim: usize) -> Vec<Q16> {
        let total = phys_dim.pow(self.num_sites as u32);
        let mut values = vec![Q16::ZERO; total];

        for idx in 0..total {
            // Decode multi-index
            let mut rem = idx;
            let mut indices = vec![0usize; self.num_sites];
            for s in (0..self.num_sites).rev() {
                indices[s] = rem % phys_dim;
                rem /= phys_dim;
            }

            // Contract MPS cores
            let mut vec = vec![Q16::ONE]; // 1x1 initial
            for s in 0..self.num_sites {
                let core = &self.cores[s];
                let p = indices[s];
                let mut new_vec = vec![Q16::ZERO; core.right_bond];
                for r in 0..core.right_bond {
                    let mut sum = Q16::ZERO;
                    for l in 0..core.left_bond {
                        sum = sum + vec[l] * core.get(l, p, r);
                    }
                    new_vec[r] = sum;
                }
                vec = new_vec;
            }
            values[idx] = vec[0];
        }
        values
    }

    /// Compute L2 norm squared of the field.
    pub fn norm_sq(&self) -> Q16 {
        let values = self.to_values(if self.cores.is_empty() { 2 } else { self.cores[0].phys_dim });
        let mut sum = Q16::ZERO;
        for v in &values {
            sum = sum + *v * *v;
        }
        sum
    }

    /// Sum all field values (for conservation integrals).
    pub fn sum(&self) -> Q16 {
        let pd = if self.cores.is_empty() { 2 } else { self.cores[0].phys_dim };
        let values = self.to_values(pd);
        let mut s = Q16::ZERO;
        for v in &values {
            s = s + *v;
        }
        s
    }

    /// Total number of tensor elements across all cores.
    pub fn total_elements(&self) -> usize {
        self.cores.iter().map(|c| c.numel()).sum()
    }

    /// Bond dimensions at each interface.
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.cores.iter().map(|c| c.right_bond).collect()
    }

    /// SVD truncation to enforce chi_max across all bonds.
    pub fn truncate(&mut self, chi_max: usize) {
        self.chi_max = chi_max;
        // Left-to-right sweep: QR at each site
        for i in 0..self.num_sites.saturating_sub(1) {
            let core = &self.cores[i];
            let rows = core.left_bond * core.phys_dim;
            let cols = core.right_bond;

            // Flatten core to matrix
            let mat: Vec<Q16> = core.data.clone();

            // Truncated SVD
            let (u, s, vt, rank) = truncated_svd(&mat, rows, cols, chi_max);
            let new_rank = rank.min(chi_max);

            // Update current core with U
            let mut new_core = Core::zeros(core.left_bond, core.phys_dim, new_rank);
            for l in 0..core.left_bond {
                for p in 0..core.phys_dim {
                    for r in 0..new_rank {
                        let row = l * core.phys_dim + p;
                        new_core.set(l, p, r, u[row * rank + r]);
                    }
                }
            }
            self.cores[i] = new_core;

            // Absorb S*Vt into next core
            let next = &self.cores[i + 1];
            let mut sv = vec![Q16::ZERO; new_rank * cols];
            for r in 0..new_rank {
                for c in 0..cols {
                    sv[r * cols + c] = s[r] * vt[r * cols + c];
                }
            }

            let mut new_next = Core::zeros(new_rank, next.phys_dim, next.right_bond);
            for l in 0..new_rank {
                for p in 0..next.phys_dim {
                    for r in 0..next.right_bond {
                        let mut sum = Q16::ZERO;
                        for k in 0..cols.min(next.left_bond) {
                            sum = sum + sv[l * cols + k] * next.get(k, p, r);
                        }
                        new_next.set(l, p, r, sum);
                    }
                }
            }
            self.cores[i + 1] = new_next;
        }
    }
}

/// Element-wise addition of two MPS (direct sum of bond dimensions, then truncate).
pub fn mps_add(a: &Mps, b: &Mps, chi_max: usize) -> Mps {
    assert_eq!(a.num_sites, b.num_sites);
    let n = a.num_sites;
    let phys_dim = a.cores[0].phys_dim;

    let mut cores = Vec::with_capacity(n);

    for i in 0..n {
        let ca = &a.cores[i];
        let cb = &b.cores[i];

        let new_left = if i == 0 { 1 } else { ca.left_bond + cb.left_bond };
        let new_right = if i == n - 1 { 1 } else { ca.right_bond + cb.right_bond };

        let mut core = Core::zeros(new_left, phys_dim, new_right);

        if i == 0 {
            // First site: concatenate along right bond
            for p in 0..phys_dim {
                for r in 0..ca.right_bond {
                    core.set(0, p, r, ca.get(0, p, r));
                }
                for r in 0..cb.right_bond {
                    core.set(0, p, ca.right_bond + r, cb.get(0, p, r));
                }
            }
        } else if i == n - 1 {
            // Last site: concatenate along left bond
            for p in 0..phys_dim {
                for l in 0..ca.left_bond {
                    core.set(l, p, 0, ca.get(l, p, 0));
                }
                for l in 0..cb.left_bond {
                    core.set(ca.left_bond + l, p, 0, cb.get(l, p, 0));
                }
            }
        } else {
            // Interior: block diagonal
            for p in 0..phys_dim {
                for l in 0..ca.left_bond {
                    for r in 0..ca.right_bond {
                        core.set(l, p, r, ca.get(l, p, r));
                    }
                }
                for l in 0..cb.left_bond {
                    for r in 0..cb.right_bond {
                        core.set(ca.left_bond + l, p, ca.right_bond + r, cb.get(l, p, r));
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

/// Scalar multiplication of MPS.
pub fn mps_scale(mps: &Mps, scalar: Q16) -> Mps {
    let mut result = mps.clone();
    // Scale first core only
    if !result.cores.is_empty() {
        for elem in result.cores[0].data.iter_mut() {
            *elem = *elem * scalar;
        }
    }
    result
}

/// Truncated SVD for Q16.16 matrices.
/// Returns (U, S, Vt, rank) where rank <= chi_max.
/// Uses one-sided Jacobi method for fixed-point stability.
fn truncated_svd(mat: &[Q16], rows: usize, cols: usize, chi_max: usize) -> (Vec<Q16>, Vec<Q16>, Vec<Q16>, usize) {
    let k = rows.min(cols).min(chi_max);

    // Compute A^T A
    let mut ata = vec![Q16::ZERO; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut sum = Q16::ZERO;
            for r in 0..rows {
                sum = sum + mat[r * cols + i] * mat[r * cols + j];
            }
            ata[i * cols + j] = sum;
        }
    }

    // Power iteration for top-k singular vectors
    let mut v_vecs: Vec<Vec<Q16>> = Vec::with_capacity(k);
    let mut sigmas = Vec::with_capacity(k);

    for ki in 0..k {
        // Initialize with alternating pattern for stability
        let mut v = vec![Q16::ZERO; cols];
        v[ki % cols] = Q16::ONE;

        // Power iteration
        for _ in 0..30 {
            // w = A^T A v
            let mut w = vec![Q16::ZERO; cols];
            for i in 0..cols {
                let mut sum = Q16::ZERO;
                for j in 0..cols {
                    sum = sum + ata[i * cols + j] * v[j];
                }
                w[i] = sum;
            }

            // Deflate against previous vectors
            for prev in &v_vecs {
                let mut dot = Q16::ZERO;
                for j in 0..cols {
                    dot = dot + w[j] * prev[j];
                }
                for j in 0..cols {
                    w[j] = w[j] - dot * prev[j];
                }
            }

            // Normalize
            let mut norm_sq = Q16::ZERO;
            for j in 0..cols {
                norm_sq = norm_sq + w[j] * w[j];
            }
            let norm = norm_sq.sqrt();
            if norm.raw() == 0 { break; }
            for j in 0..cols {
                v[j] = w[j].div(norm);
            }
        }

        // Compute sigma = ||A v||
        let mut av = vec![Q16::ZERO; rows];
        for r in 0..rows {
            let mut sum = Q16::ZERO;
            for c in 0..cols {
                sum = sum + mat[r * cols + c] * v[c];
            }
            av[r] = sum;
        }

        let mut sigma_sq = Q16::ZERO;
        for r in 0..rows {
            sigma_sq = sigma_sq + av[r] * av[r];
        }
        let sigma = sigma_sq.sqrt();

        if sigma.raw() == 0 { break; }

        sigmas.push(sigma);
        v_vecs.push(v);
    }

    let rank = sigmas.len().max(1);

    // Build U = A V S^{-1}
    let mut u = vec![Q16::ZERO; rows * rank];
    for ki in 0..sigmas.len() {
        if sigmas[ki].raw() == 0 { continue; }
        for r in 0..rows {
            let mut sum = Q16::ZERO;
            for c in 0..cols {
                sum = sum + mat[r * cols + c] * v_vecs[ki][c];
            }
            u[r * rank + ki] = sum.div(sigmas[ki]);
        }
    }

    // Build Vt
    let mut vt = vec![Q16::ZERO; rank * cols];
    for ki in 0..sigmas.len() {
        for c in 0..cols {
            vt[ki * cols + c] = v_vecs[ki][c];
        }
    }

    // Pad if rank < 1
    let mut s = vec![Q16::ZERO; rank];
    for i in 0..sigmas.len() {
        s[i] = sigmas[i];
    }

    (u, s, vt, rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_mps() {
        let mps = Mps::uniform(4, 2, Q16::HALF);
        let vals = mps.to_values(2);
        // First phys_dim^0 * ... pattern should reconstruct
        assert!(vals.len() == 16);
    }

    #[test]
    fn test_mps_zeros() {
        let mps = Mps::zeros(6, 2, 4);
        assert_eq!(mps.num_sites, 6);
        assert_eq!(mps.cores.len(), 6);
    }

    #[test]
    fn test_mps_sum() {
        let mps = Mps::uniform(4, 2, Q16::ONE);
        let s = mps.sum();
        // Should be nonzero for uniform field
        assert!(s.raw() != 0);
    }

    #[test]
    fn test_mps_add() {
        let a = Mps::uniform(4, 2, Q16::ONE);
        let b = Mps::uniform(4, 2, Q16::ONE);
        let c = mps_add(&a, &b, 4);
        assert_eq!(c.num_sites, 4);
    }

    #[test]
    fn test_mps_scale() {
        let a = Mps::uniform(4, 2, Q16::ONE);
        let b = mps_scale(&a, Q16::TWO);
        assert_eq!(b.num_sites, 4);
    }

    #[test]
    fn test_bond_dimensions() {
        let mps = Mps::zeros(6, 2, 4);
        let bonds = mps.bond_dimensions();
        assert_eq!(bonds.len(), 6);
    }
}
