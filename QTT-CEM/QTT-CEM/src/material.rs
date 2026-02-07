//! Material Properties for Electromagnetic Media
//!
//! Encodes permittivity (ε), permeability (μ), and conductivity (σ)
//! as Q16.16 fields for use in FDTD updates.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// Physical constants in Q16.16.
pub struct Constants;

impl Constants {
    /// Speed of light (normalized to 1.0 for computational units).
    pub const C: Q16 = Q16::ONE;

    /// Free-space permittivity (normalized to 1.0).
    pub const EPSILON_0: Q16 = Q16::ONE;

    /// Free-space permeability (normalized to 1.0).
    pub const MU_0: Q16 = Q16::ONE;

    /// Free-space impedance (sqrt(mu_0/epsilon_0) = 1.0 in normalized units).
    pub const ETA_0: Q16 = Q16::ONE;
}

/// Material properties at a single grid point.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Relative permittivity εr.
    pub epsilon_r: Q16,
    /// Relative permeability μr.
    pub mu_r: Q16,
    /// Conductivity σ (S/m in normalized units).
    pub sigma: Q16,
    /// Magnetic loss σ_m.
    pub sigma_m: Q16,
}

impl Material {
    /// Free space.
    pub const fn vacuum() -> Self {
        Material {
            epsilon_r: Q16::ONE,
            mu_r: Q16::ONE,
            sigma: Q16::ZERO,
            sigma_m: Q16::ZERO,
        }
    }

    /// Dielectric with given εr.
    pub fn dielectric(epsilon_r: Q16) -> Self {
        Material {
            epsilon_r,
            mu_r: Q16::ONE,
            sigma: Q16::ZERO,
            sigma_m: Q16::ZERO,
        }
    }

    /// Conductor with given σ.
    pub fn conductor(sigma: Q16) -> Self {
        Material {
            epsilon_r: Q16::ONE,
            mu_r: Q16::ONE,
            sigma,
            sigma_m: Q16::ZERO,
        }
    }

    /// Lossy dielectric.
    pub fn lossy_dielectric(epsilon_r: Q16, sigma: Q16) -> Self {
        Material {
            epsilon_r,
            mu_r: Q16::ONE,
            sigma,
            sigma_m: Q16::ZERO,
        }
    }

    /// FDTD E-field update coefficients.
    /// ca = (1 - σΔt/2ε) / (1 + σΔt/2ε)
    /// cb = (Δt/εΔx) / (1 + σΔt/2ε)
    pub fn e_update_coefficients(&self, dt: Q16, dx: Q16) -> (Q16, Q16) {
        let eps = self.epsilon_r * Constants::EPSILON_0;
        let sigma_dt_2eps = self.sigma * dt;
        let two_eps = Q16::TWO * eps;

        let numer_a = two_eps - sigma_dt_2eps;
        let denom = two_eps + sigma_dt_2eps;

        if denom.raw() == 0 {
            return (Q16::ONE, dt.div(eps * dx));
        }

        let ca = numer_a.div(denom);
        let cb = (Q16::TWO * dt).div(denom * dx);

        (ca, cb)
    }

    /// FDTD H-field update coefficients.
    /// da = (1 - σ_mΔt/2μ) / (1 + σ_mΔt/2μ)
    /// db = (Δt/μΔx) / (1 + σ_mΔt/2μ)
    pub fn h_update_coefficients(&self, dt: Q16, dx: Q16) -> (Q16, Q16) {
        let mu = self.mu_r * Constants::MU_0;
        let sigma_m_dt_2mu = self.sigma_m * dt;
        let two_mu = Q16::TWO * mu;

        let numer_a = two_mu - sigma_m_dt_2mu;
        let denom = two_mu + sigma_m_dt_2mu;

        if denom.raw() == 0 {
            return (Q16::ONE, dt.div(mu * dx));
        }

        let da = numer_a.div(denom);
        let db = (Q16::TWO * dt).div(denom * dx);

        (da, db)
    }
}

/// Spatially-varying material map for a 3D grid.
#[derive(Clone, Debug)]
pub struct MaterialMap {
    pub materials: Vec<Material>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl MaterialMap {
    /// Uniform vacuum.
    pub fn vacuum(nx: usize, ny: usize, nz: usize) -> Self {
        MaterialMap {
            materials: vec![Material::vacuum(); nx * ny * nz],
            nx, ny, nz,
        }
    }

    /// Get material at grid point (i, j, k).
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> &Material {
        &self.materials[i * self.ny * self.nz + j * self.nz + k]
    }

    /// Set material at grid point (i, j, k).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, mat: Material) {
        self.materials[i * self.ny * self.nz + j * self.nz + k] = mat;
    }

    /// Insert a dielectric slab along one axis.
    pub fn insert_slab_x(&mut self, x_start: usize, x_end: usize, mat: Material) {
        for i in x_start..x_end.min(self.nx) {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    self.set(i, j, k, mat);
                }
            }
        }
    }

    /// Insert a dielectric sphere (approximate on grid).
    pub fn insert_sphere(&mut self, cx: usize, cy: usize, cz: usize, radius: usize, mat: Material) {
        let r2 = (radius * radius) as i64;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let di = i as i64 - cx as i64;
                    let dj = j as i64 - cy as i64;
                    let dk = k as i64 - cz as i64;
                    if di*di + dj*dj + dk*dk <= r2 {
                        self.set(i, j, k, mat);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_coefficients() {
        let mat = Material::vacuum();
        let dt = Q16::from_f64(0.01);
        let dx = Q16::from_f64(0.1);
        let (ca, cb) = mat.e_update_coefficients(dt, dx);
        // Vacuum: ca ≈ 1.0, cb ≈ dt/(ε₀*dx) = 0.1
        assert!((ca.to_f64() - 1.0).abs() < 0.01);
        assert!((cb.to_f64() - 0.1).abs() < 0.02);
    }

    #[test]
    fn test_material_map() {
        let mut map = MaterialMap::vacuum(8, 8, 8);
        map.insert_slab_x(2, 4, Material::dielectric(Q16::from_f64(4.0)));
        assert!((map.get(3, 0, 0).epsilon_r.to_f64() - 4.0).abs() < 0.01);
        assert!((map.get(0, 0, 0).epsilon_r.to_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lossy_coefficients() {
        let mat = Material::lossy_dielectric(Q16::from_f64(2.0), Q16::from_f64(0.5));
        let dt = Q16::from_f64(0.01);
        let dx = Q16::from_f64(0.1);
        let (ca, _cb) = mat.e_update_coefficients(dt, dx);
        // ca should be < 1.0 due to loss
        assert!(ca.to_f64() < 1.0);
    }
}
