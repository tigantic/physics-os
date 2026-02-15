//! NS-IMEX circuit: witness generation and structural validation.
//!
//! Generates the complete witness and validates constraints without
//! a cryptographic proving backend. The real ZK proof for NS-IMEX
//! will be added via a STARK AIR.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::config::{NSIMEXCircuitSizing, NSIMEXParams};
use super::witness::{NSIMEXWitness, WitnessGenerator};

/// NS-IMEX circuit: generates witness and validates constraints.
///
/// Generates and validates the witness but does not create
/// a cryptographic proof. The stub prover wraps this with
/// simulated proof bytes for pipeline compatibility.
#[derive(Clone)]
pub struct NSIMEXCircuit {
    /// Solver parameters (public).
    pub params: NSIMEXParams,
    /// Circuit sizing.
    pub sizing: NSIMEXCircuitSizing,
    /// Complete witness data.
    pub witness: NSIMEXWitness,
}

impl NSIMEXCircuit {
    /// Create a new circuit from input states and shift MPOs.
    pub fn new(
        params: NSIMEXParams,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<Self, String> {
        let sizing = NSIMEXCircuitSizing::from_params(&params);
        let generator = WitnessGenerator::new(params.clone());

        let witness = generator
            .generate(input_states, shift_mpos)
            .map_err(|e| format!("Witness generation failed: {}", e))?;

        Ok(Self {
            params,
            sizing,
            witness,
        })
    }

    /// Get the k parameter for the circuit.
    pub fn k(&self) -> u32 {
        self.sizing.k as u32
    }

    /// Get the number of estimated constraints.
    pub fn estimate_constraints(&self) -> usize {
        self.sizing.total_constraints
    }

    /// Validate the witness (check all constraints without ZK).
    ///
    /// Returns Ok(()) if all constraints pass, or an error describing
    /// the first failed constraint.
    pub fn validate_witness(&self) -> Result<(), String> {
        // Validate SVD ordering for all truncations
        for (stage_idx, stage) in self.witness.stages.iter().enumerate() {
            for (sweep_idx, sweep) in stage.variable_sweeps.iter().enumerate() {
                for trunc in &sweep.truncations {
                    for (i, sv) in trunc.truncated_values.iter().enumerate() {
                        if sv.raw < 0 {
                            return Err(format!(
                                "Negative SV at stage {}, sweep {}, bond {}, sv {}: {}",
                                stage_idx, sweep_idx, trunc.bond, i, sv.to_f64(),
                            ));
                        }
                    }
                    for pair in trunc.truncated_values.windows(2) {
                        if pair[0].raw < pair[1].raw {
                            return Err(format!(
                                "SV ordering violation at stage {}, sweep {}, bond {}: {} < {}",
                                stage_idx, sweep_idx, trunc.bond,
                                pair[0].to_f64(), pair[1].to_f64(),
                            ));
                        }
                    }
                }
            }
        }

        // Validate conservation: KE
        let ke_residual = self.witness.kinetic_energy_after.raw
            - self.witness.kinetic_energy_before.raw;
        if ke_residual.unsigned_abs() as i64 > self.params.conservation_tolerance.raw {
            return Err(format!(
                "KE conservation violation: residual {} > tolerance {}",
                ke_residual as f64 / 65536.0,
                self.params.conservation_tolerance.to_f64(),
            ));
        }

        // Validate divergence
        if self.witness.divergence_residual.raw.unsigned_abs() as i64
            > self.params.divergence_tolerance.raw
        {
            return Err(format!(
                "Divergence violation: {} > tolerance {}",
                self.witness.divergence_residual.to_f64(),
                self.params.divergence_tolerance.to_f64(),
            ));
        }

        // Validate diffusion solve residuals
        for stage in &self.witness.stages {
            if let Some(ref diff) = stage.diffusion_witness {
                for var_w in &diff.variables {
                    if var_w.solve_residual.raw.unsigned_abs() as i64
                        > self.params.diffusion_solve_tolerance.raw
                    {
                        return Err(format!(
                            "Diffusion solve residual too large for {}: {} > {}",
                            var_w.variable,
                            var_w.solve_residual.to_f64(),
                            self.params.diffusion_solve_tolerance.to_f64(),
                        ));
                    }
                }
            }
        }

        // Validate CG convergence
        for stage in &self.witness.stages {
            if let Some(ref proj) = stage.projection_witness {
                if proj.cg_residual.raw.unsigned_abs() as i64
                    > self.params.cg_tolerance.raw
                {
                    return Err(format!(
                        "CG not converged: residual {} > tolerance {}",
                        proj.cg_residual.to_f64(),
                        self.params.cg_tolerance.to_f64(),
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::{NUM_NS_VARIABLES, PHYS_DIM, NUM_DIMENSIONS};

    #[test]
    fn test_circuit_creation() {
        let params = NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
            .collect();
        let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
            .map(|_| MPO::identity(num_sites, PHYS_DIM))
            .collect();

        let circuit =
            NSIMEXCircuit::new(params, &input_states, &shift_mpos)
                .expect("Circuit creation failed");

        assert!(circuit.estimate_constraints() > 0);
    }

    #[test]
    fn test_witness_validation() {
        let params = NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
            .collect();
        let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
            .map(|_| MPO::identity(num_sites, PHYS_DIM))
            .collect();

        let circuit =
            NSIMEXCircuit::new(params, &input_states, &shift_mpos)
                .expect("Circuit creation failed");

        let result = circuit.validate_witness();
        assert!(
            result.is_ok(),
            "Witness validation failed: {:?}",
            result.err()
        );
    }
}
