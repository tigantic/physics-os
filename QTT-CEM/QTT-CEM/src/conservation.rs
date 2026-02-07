//! Conservation Verification — Poynting Theorem
//!
//! Verifies electromagnetic energy conservation:
//!   ∂u/∂t + ∇·S = −J·E − σ|E|²
//!
//! where:
//!   u = 0.5*(ε|E|² + μ|H|²)  — energy density
//!   S = E × H                  — Poynting vector (energy flux)
//!   J·E                        — source work
//!   σ|E|²                      — ohmic loss
//!
//! In lossless vacuum with no sources and periodic BC:
//!   Total energy is exactly conserved (modulo fixed-point truncation).
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::fdtd::YeeFields;
use crate::material::MaterialMap;

/// Conservation check result for a single timestep.
#[derive(Clone, Debug)]
pub struct ConservationResult {
    /// Electromagnetic energy before step.
    pub energy_before: Q16,
    /// Electromagnetic energy after step.
    pub energy_after: Q16,
    /// Energy injected by sources this step.
    pub source_energy: Q16,
    /// Energy dissipated by ohmic loss this step.
    pub ohmic_loss: Q16,
    /// Residual: |ΔU - source + loss|.
    pub residual: Q16,
    /// Conservation tolerance.
    pub tolerance: Q16,
    /// Whether conservation holds within tolerance.
    pub conserved: bool,
}

/// Conservation verifier tracks energy balance across simulation.
pub struct ConservationVerifier {
    pub results: Vec<ConservationResult>,
    pub tolerance: Q16,
}

impl ConservationVerifier {
    /// Create verifier with given tolerance.
    pub fn new(tolerance: Q16) -> Self {
        ConservationVerifier {
            results: Vec::new(),
            tolerance,
        }
    }

    /// Default tolerance based on Q16.16 precision.
    /// ε_cons = 7 raw ≈ 1.07×10⁻⁴.
    pub fn default() -> Self {
        Self::new(Q16::from_raw(7))
    }

    /// Check conservation for a single timestep.
    pub fn check_step(
        &mut self,
        fields_before: &YeeFields,
        fields_after: &YeeFields,
        materials: &MaterialMap,
        source_energy: Q16,
    ) -> &ConservationResult {
        let e_before = fields_before.total_energy(materials);
        let e_after = fields_after.total_energy(materials);

        // Compute ohmic loss: σ|E|²ΔV over domain
        let mut ohmic = Q16::ZERO;
        for i in 0..fields_after.nx {
            for j in 0..fields_after.ny {
                for k in 0..fields_after.nz {
                    let idx = i * fields_after.ny * fields_after.nz + j * fields_after.nz + k;
                    let mat = materials.get(i, j, k);
                    if mat.sigma.raw() != 0 {
                        let e_sq = fields_after.ex[idx] * fields_after.ex[idx]
                            + fields_after.ey[idx] * fields_after.ey[idx]
                            + fields_after.ez[idx] * fields_after.ez[idx];
                        ohmic = ohmic + mat.sigma * e_sq;
                    }
                }
            }
        }

        // Energy balance: ΔU = source - loss
        let delta_u = e_after - e_before;
        let expected = source_energy - ohmic;
        let residual = (delta_u - expected).abs();
        let conserved = residual.raw() <= self.tolerance.raw();

        self.results.push(ConservationResult {
            energy_before: e_before,
            energy_after: e_after,
            source_energy,
            ohmic_loss: ohmic,
            residual,
            tolerance: self.tolerance,
            conserved,
        });

        self.results.last().unwrap()
    }

    /// Check conservation over an entire simulation run.
    /// Returns (all_conserved, max_residual, total_steps).
    pub fn verify_run(energies: &[Q16], tolerance: Q16) -> (bool, Q16, usize) {
        if energies.len() < 2 {
            return (true, Q16::ZERO, 0);
        }

        let e0 = energies[0];
        let mut max_residual = Q16::ZERO;
        let mut all_conserved = true;

        for e in &energies[1..] {
            let residual = (*e - e0).abs();
            if residual.raw() > max_residual.raw() {
                max_residual = residual;
            }
            if residual.raw() > tolerance.raw() {
                all_conserved = false;
            }
        }

        (all_conserved, max_residual, energies.len() - 1)
    }

    /// Summary statistics.
    pub fn summary(&self) -> ConservationSummary {
        let total = self.results.len();
        let conserved = self.results.iter().filter(|r| r.conserved).count();
        let max_residual = self.results.iter()
            .map(|r| r.residual)
            .max_by_key(|r| r.raw())
            .unwrap_or(Q16::ZERO);
        let mean_residual = if total > 0 {
            let sum: i64 = self.results.iter().map(|r| r.residual.raw() as i64).sum();
            Q16::from_raw((sum / total as i64) as i32)
        } else {
            Q16::ZERO
        };

        ConservationSummary {
            total_steps: total,
            steps_conserved: conserved,
            max_residual,
            mean_residual,
            all_conserved: conserved == total,
            verdict: if conserved == total { "CONSERVED" } else { "VIOLATION" },
        }
    }
}

/// Summary of conservation verification across simulation.
#[derive(Clone, Debug)]
pub struct ConservationSummary {
    pub total_steps: usize,
    pub steps_conserved: usize,
    pub max_residual: Q16,
    pub mean_residual: Q16,
    pub all_conserved: bool,
    pub verdict: &'static str,
}

impl std::fmt::Display for ConservationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Conservation: {} ({}/{} steps, max residual: {}, mean: {})",
            self.verdict, self.steps_conserved, self.total_steps,
            self.max_residual, self.mean_residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_constant_energy() {
        let energies = vec![Q16::ONE; 10];
        let (conserved, max_res, steps) = ConservationVerifier::verify_run(
            &energies, Q16::from_raw(7));
        assert!(conserved);
        assert_eq!(max_res.raw(), 0);
        assert_eq!(steps, 9);
    }

    #[test]
    fn test_verify_violation() {
        let mut energies = vec![Q16::ONE; 10];
        energies[5] = Q16::TWO; // Sudden energy spike
        let (conserved, _, _) = ConservationVerifier::verify_run(
            &energies, Q16::from_raw(7));
        assert!(!conserved);
    }

    #[test]
    fn test_default_verifier() {
        let v = ConservationVerifier::default();
        assert_eq!(v.tolerance.raw(), 7);
    }
}
