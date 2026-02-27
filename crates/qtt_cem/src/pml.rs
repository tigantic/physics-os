//! Perfectly Matched Layer (PML) Absorbing Boundaries
//!
//! Implements Berenger's split-field PML for terminating the FDTD domain
//! without spurious reflections. The PML introduces a graded conductivity
//! profile that attenuates outgoing waves while remaining impedance-matched.
//!
//! Damping profile: σ(d) = σ_max * (d / L_pml)^m
//! where d = distance into PML, L_pml = PML thickness, m = grading order.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// PML configuration parameters.
#[derive(Clone, Debug)]
pub struct PmlParams {
    /// Number of PML cells on each face.
    pub thickness: usize,
    /// Maximum conductivity at PML outer boundary.
    pub sigma_max: Q16,
    /// Polynomial grading order (typically 3 or 4).
    pub grading_order: u32,
    /// Target reflection coefficient for normal incidence.
    pub target_reflection: Q16,
}

impl PmlParams {
    /// Create PML parameters optimized for given grid spacing.
    pub fn optimal(thickness: usize, dx: Q16) -> Self {
        // σ_max = -(m+1) * ln(R) / (2 * η * L_pml)
        // For R = 10^-6, m = 3, η = 1 (normalized)
        let m = 3u32;
        let l_pml = Q16::from_int(thickness as i32) * dx;
        // ln(10^-6) ≈ -13.8
        let ln_r = Q16::from_f64(-13.8);
        let sigma_max = (Q16::from_int(-(m as i32 + 1)) * ln_r).div(Q16::TWO * l_pml);

        PmlParams {
            thickness,
            sigma_max,
            grading_order: m,
            target_reflection: Q16::from_f64(1e-6),
        }
    }

    /// Compute damping coefficient at distance `d` cells into PML.
    pub fn sigma_at(&self, d: usize) -> Q16 {
        if d == 0 { return Q16::ZERO; }
        let ratio = Q16::from_ratio(d as i32, self.thickness as i32);
        // ratio^m: repeated multiplication
        let mut val = ratio;
        for _ in 1..self.grading_order {
            val = val * ratio;
        }
        self.sigma_max * val
    }
}

/// PML auxiliary fields for split-field formulation.
/// Stores the split components needed for PML update equations.
#[derive(Clone, Debug)]
pub struct PmlFields {
    /// Split E-field components (xy, xz, yx, yz, zx, zy).
    pub exy: Vec<Q16>,
    pub exz: Vec<Q16>,
    pub eyx: Vec<Q16>,
    pub eyz: Vec<Q16>,
    pub ezx: Vec<Q16>,
    pub ezy: Vec<Q16>,
    /// Split H-field components.
    pub hxy: Vec<Q16>,
    pub hxz: Vec<Q16>,
    pub hyx: Vec<Q16>,
    pub hyz: Vec<Q16>,
    pub hzx: Vec<Q16>,
    pub hzy: Vec<Q16>,
    /// Damping profiles along each axis.
    pub sigma_x: Vec<Q16>,
    pub sigma_y: Vec<Q16>,
    pub sigma_z: Vec<Q16>,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl PmlFields {
    /// Initialize PML auxiliary fields and damping profiles.
    pub fn new(nx: usize, ny: usize, nz: usize, params: &PmlParams) -> Self {
        let n = nx * ny * nz;

        // Build damping profiles
        let mut sigma_x = vec![Q16::ZERO; nx];
        let mut sigma_y = vec![Q16::ZERO; ny];
        let mut sigma_z = vec![Q16::ZERO; nz];

        let t = params.thickness;

        // Left PML region
        for i in 0..t.min(nx) {
            let d = t - i;
            sigma_x[i] = params.sigma_at(d);
        }
        // Right PML region
        for i in nx.saturating_sub(t)..nx {
            let d = i - (nx - t - 1);
            sigma_x[i] = params.sigma_at(d);
        }

        for j in 0..t.min(ny) {
            let d = t - j;
            sigma_y[j] = params.sigma_at(d);
        }
        for j in ny.saturating_sub(t)..ny {
            let d = j - (ny - t - 1);
            sigma_y[j] = params.sigma_at(d);
        }

        for k in 0..t.min(nz) {
            let d = t - k;
            sigma_z[k] = params.sigma_at(d);
        }
        for k in nz.saturating_sub(t)..nz {
            let d = k - (nz - t - 1);
            sigma_z[k] = params.sigma_at(d);
        }

        PmlFields {
            exy: vec![Q16::ZERO; n], exz: vec![Q16::ZERO; n],
            eyx: vec![Q16::ZERO; n], eyz: vec![Q16::ZERO; n],
            ezx: vec![Q16::ZERO; n], ezy: vec![Q16::ZERO; n],
            hxy: vec![Q16::ZERO; n], hxz: vec![Q16::ZERO; n],
            hyx: vec![Q16::ZERO; n], hyz: vec![Q16::ZERO; n],
            hzx: vec![Q16::ZERO; n], hzy: vec![Q16::ZERO; n],
            sigma_x, sigma_y, sigma_z,
            nx, ny, nz,
        }
    }

    /// Check if a point is inside any PML region.
    pub fn is_pml_region(&self, i: usize, j: usize, k: usize) -> bool {
        self.sigma_x[i].raw() != 0
            || self.sigma_y[j].raw() != 0
            || self.sigma_z[k].raw() != 0
    }

    /// Get PML damping factor for E-field update: (1 - σΔt/2ε) / (1 + σΔt/2ε).
    pub fn damping_factor(sigma: Q16, dt: Q16) -> Q16 {
        let half_sigma_dt = Q16::HALF * sigma * dt;
        let numer = Q16::ONE - half_sigma_dt;
        let denom = Q16::ONE + half_sigma_dt;
        if denom.raw() == 0 { return Q16::ONE; }
        numer.div(denom)
    }

    /// Get PML coefficient for curl term: (Δt/ε) / (1 + σΔt/2ε).
    pub fn curl_coefficient(sigma: Q16, dt: Q16, dx: Q16) -> Q16 {
        let denom = Q16::ONE + Q16::HALF * sigma * dt;
        if denom.raw() == 0 { return dt.div(dx); }
        dt.div(denom * dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pml_params() {
        let dx = Q16::from_f64(0.01);
        let params = PmlParams::optimal(8, dx);
        assert!(params.sigma_max.raw() > 0);
        assert_eq!(params.grading_order, 3);
    }

    #[test]
    fn test_pml_grading() {
        let dx = Q16::from_f64(0.01);
        let params = PmlParams::optimal(8, dx);
        // Damping should increase into PML
        let s1 = params.sigma_at(1);
        let s4 = params.sigma_at(4);
        let s8 = params.sigma_at(8);
        assert!(s1.raw() < s4.raw());
        assert!(s4.raw() < s8.raw());
    }

    #[test]
    fn test_pml_fields_init() {
        let dx = Q16::from_f64(0.01);
        let params = PmlParams::optimal(2, dx);
        let pml = PmlFields::new(8, 8, 8, &params);
        // Interior point should have zero damping
        assert_eq!(pml.sigma_x[4].raw(), 0);
        // PML region should have nonzero damping
        assert!(pml.sigma_x[0].raw() > 0);
    }

    #[test]
    fn test_damping_factor() {
        let sigma = Q16::from_f64(1.0);
        let dt = Q16::from_f64(0.01);
        let factor = PmlFields::damping_factor(sigma, dt);
        // Should be close to but less than 1.0
        assert!(factor.to_f64() < 1.0);
        assert!(factor.to_f64() > 0.9);
    }
}
