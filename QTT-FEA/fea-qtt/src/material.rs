//! Isotropic Linear Elastic Material
//!
//! Constitutive relation: σ = D·ε (Voigt notation)
//!
//! D matrix for isotropic material (6×6):
//!   D = (E/((1+ν)(1-2ν))) * [1-ν  ν    ν    0       0       0      ]
//!                             [ν    1-ν  ν    0       0       0      ]
//!                             [ν    ν    1-ν  0       0       0      ]
//!                             [0    0    0    (1-2ν)/2 0       0     ]
//!                             [0    0    0    0       (1-2ν)/2 0     ]
//!                             [0    0    0    0       0       (1-2ν)/2]
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// Isotropic linear elastic material.
#[derive(Clone, Copy, Debug)]
pub struct IsotropicMaterial {
    /// Young's modulus E.
    pub youngs_modulus: Q16,
    /// Poisson's ratio ν.
    pub poisson_ratio: Q16,
    /// Density ρ (for dynamic analysis).
    pub density: Q16,
}

impl IsotropicMaterial {
    pub fn new(e: f64, nu: f64) -> Self {
        IsotropicMaterial {
            youngs_modulus: Q16::from_f64(e),
            poisson_ratio: Q16::from_f64(nu),
            density: Q16::ONE,
        }
    }

    /// Steel: E=200 (normalized), ν=0.3.
    pub fn steel() -> Self { Self::new(200.0, 0.3) }

    /// Aluminum: E=70 (normalized), ν=0.33.
    pub fn aluminum() -> Self { Self::new(70.0, 0.33) }

    /// Rubber-like: E=1 (normalized), ν=0.49.
    pub fn rubber() -> Self { Self::new(1.0, 0.49) }

    /// Shear modulus G = E / (2(1+ν)).
    pub fn shear_modulus(&self) -> Q16 {
        self.youngs_modulus.div(Q16::TWO * (Q16::ONE + self.poisson_ratio))
    }

    /// Bulk modulus K = E / (3(1-2ν)).
    pub fn bulk_modulus(&self) -> Q16 {
        let denom = Q16::THREE * (Q16::ONE - Q16::TWO * self.poisson_ratio);
        self.youngs_modulus.div(denom)
    }

    /// Lamé's first parameter λ = Eν / ((1+ν)(1-2ν)).
    pub fn lame_lambda(&self) -> Q16 {
        let nu = self.poisson_ratio;
        let e = self.youngs_modulus;
        let numer = e * nu;
        let denom = (Q16::ONE + nu) * (Q16::ONE - Q16::TWO * nu);
        numer.div(denom)
    }

    /// Lamé's second parameter μ = G = E / (2(1+ν)).
    pub fn lame_mu(&self) -> Q16 {
        self.shear_modulus()
    }

    /// 6×6 constitutive matrix D in Voigt notation.
    /// Returns flat [36] array in row-major order.
    pub fn constitutive_matrix(&self) -> [Q16; 36] {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;

        let one = Q16::ONE;
        let two = Q16::TWO;

        // factor = E / ((1+ν)(1-2ν))
        let denom = (one + nu) * (one - two * nu);
        let factor = e.div(denom);

        let d11 = factor * (one - nu);  // (1-ν) * factor
        let d12 = factor * nu;           // ν * factor
        let d44 = factor * (one - two * nu).div(two); // (1-2ν)/2 * factor

        let z = Q16::ZERO;

        [
            d11, d12, d12,   z,   z,   z,
            d12, d11, d12,   z,   z,   z,
            d12, d12, d11,   z,   z,   z,
              z,   z,   z, d44,   z,   z,
              z,   z,   z,   z, d44,   z,
              z,   z,   z,   z,   z, d44,
        ]
    }
}

/// Spatially-varying material map.
#[derive(Clone, Debug)]
pub struct MaterialMap {
    pub materials: Vec<IsotropicMaterial>,
    pub num_elements: usize,
}

impl MaterialMap {
    /// Uniform material for all elements.
    pub fn uniform(num_elements: usize, mat: IsotropicMaterial) -> Self {
        MaterialMap {
            materials: vec![mat; num_elements],
            num_elements,
        }
    }

    pub fn get(&self, elem: usize) -> &IsotropicMaterial {
        &self.materials[elem]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steel_properties() {
        let steel = IsotropicMaterial::steel();
        let g = steel.shear_modulus();
        // G = 200 / (2 * 1.3) ≈ 76.9
        assert!((g.to_f64() - 76.9).abs() < 1.0);
    }

    #[test]
    fn test_constitutive_symmetry() {
        let mat = IsotropicMaterial::steel();
        let d = mat.constitutive_matrix();
        // D should be symmetric: D[i][j] == D[j][i]
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(d[i*6+j].raw(), d[j*6+i].raw(),
                    "D not symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_constitutive_positive_definite() {
        let mat = IsotropicMaterial::steel();
        let d = mat.constitutive_matrix();
        // Diagonal elements should be positive
        for i in 0..6 {
            assert!(d[i*6+i].raw() > 0, "D[{},{}] not positive", i, i);
        }
    }
}
