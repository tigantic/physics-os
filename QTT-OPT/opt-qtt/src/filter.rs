//! Sensitivity & Density Filtering
//!
//! Prevents checkerboarding and ensures mesh-independent solutions.
//!
//! Weighted average filter (Sigmund 2007):
//!   dc̃ₑ = (Σⱼ wₑⱼ ρⱼ dcⱼ) / (ρₑ Σⱼ wₑⱼ)
//!
//! Weight function: wₑⱼ = max(0, rmin - dist(e,j))
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::forward::Mesh2D;

/// Precomputed filter weights for each element.
#[derive(Clone, Debug)]
pub struct SensitivityFilter {
    /// For each element: list of (neighbor_id, weight).
    pub neighbors: Vec<Vec<(usize, Q16)>>,
    /// Sum of weights per element.
    pub weight_sums: Vec<Q16>,
    pub rmin: Q16,
}

impl SensitivityFilter {
    /// Build filter with radius rmin (in element widths).
    pub fn new(mesh: &Mesh2D, rmin: f64) -> Self {
        let ne = mesh.num_elements();
        let rmin_q = Q16::from_f64(rmin);
        let mut neighbors = vec![Vec::new(); ne];
        let mut weight_sums = vec![Q16::ZERO; ne];

        for ey in 0..mesh.ny {
            for ex in 0..mesh.nx {
                let eid = ey * mesh.nx + ex;
                let cx = ex as f64 + 0.5;
                let cy = ey as f64 + 0.5;

                // Search neighborhood
                let r_ceil = rmin.ceil() as isize;
                let ex_i = ex as isize;
                let ey_i = ey as isize;

                for dy in -r_ceil..=r_ceil {
                    for dx in -r_ceil..=r_ceil {
                        let jx = ex_i + dx;
                        let jy = ey_i + dy;
                        if jx < 0 || jx >= mesh.nx as isize { continue; }
                        if jy < 0 || jy >= mesh.ny as isize { continue; }
                        let jid = jy as usize * mesh.nx + jx as usize;
                        let jcx = jx as f64 + 0.5;
                        let jcy = jy as f64 + 0.5;
                        let dist = ((cx - jcx).powi(2) + (cy - jcy).powi(2)).sqrt();
                        if dist < rmin {
                            let w = Q16::from_f64(rmin - dist);
                            neighbors[eid].push((jid, w));
                            weight_sums[eid] = weight_sums[eid] + w;
                        }
                    }
                }
            }
        }

        SensitivityFilter { neighbors, weight_sums, rmin: rmin_q }
    }

    /// Apply sensitivity filter.
    /// dc̃ₑ = (Σⱼ wₑⱼ ρⱼ dcⱼ) / (ρₑ max(Σwₑⱼ, ε))
    pub fn apply(&self, densities: &[Q16], dc: &[Q16]) -> Vec<Q16> {
        let ne = densities.len();
        let mut dc_filtered = vec![Q16::ZERO; ne];

        for e in 0..ne {
            let mut numer = Q16::ZERO;
            for &(j, w) in &self.neighbors[e] {
                numer = numer + w * densities[j] * dc[j];
            }

            let denom = densities[e].max(Q16::EPSILON) * self.weight_sums[e].max(Q16::EPSILON);
            dc_filtered[e] = numer.div(denom);
        }

        dc_filtered
    }
}

/// Density filter (for Heaviside projection methods).
/// ρ̃ₑ = (Σⱼ wₑⱼ ρⱼ) / (Σⱼ wₑⱼ)
pub fn density_filter(filter: &SensitivityFilter, densities: &[Q16]) -> Vec<Q16> {
    let ne = densities.len();
    let mut rho_filt = vec![Q16::ZERO; ne];

    for e in 0..ne {
        let mut numer = Q16::ZERO;
        for &(j, w) in &filter.neighbors[e] {
            numer = numer + w * densities[j];
        }
        rho_filt[e] = numer.div(filter.weight_sums[e].max(Q16::EPSILON));
    }

    rho_filt
}
