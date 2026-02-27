//! Euler 3D circuit: witness generation and structural validation.
//!
//! Generates the complete witness and validates constraints without
//! a cryptographic proving backend. The real ZK proof for Euler3D
//! will be added via a STARK AIR (similar to the thermal module).
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::config::{Euler3DCircuitSizing, Euler3DParams};
use super::witness::{Euler3DWitness, WitnessGenerator};

/// Euler 3D circuit: generates witness and validates constraints.
///
/// Generates and validates the witness but does not create
/// a cryptographic proof. The stub prover wraps this with
/// simulated proof bytes for pipeline compatibility.
#[derive(Clone)]
pub struct Euler3DCircuit {
    /// Physics parameters (public).
    pub params: Euler3DParams,

    /// Circuit sizing.
    pub sizing: Euler3DCircuitSizing,

    /// Complete witness data.
    pub witness: Euler3DWitness,
}

impl Euler3DCircuit {
    /// Create a new circuit from input states and shift MPOs.
    pub fn new(
        params: Euler3DParams,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<Self, String> {
        let sizing = Euler3DCircuitSizing::from_params(&params);
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
        self.sizing.k
    }

    /// Get the number of estimated constraints.
    pub fn estimate_constraints(&self) -> usize {
        self.sizing.estimate_constraints()
    }

    /// Validate the witness (check all constraints without ZK).
    ///
    /// Returns Ok(()) if all constraints pass, or an error describing
    /// the first failed constraint.
    pub fn validate_witness(&self) -> Result<(), String> {
        // Validate SVD ordering for all truncations
        for (stage_idx, stage) in self.witness.strang_stages.iter().enumerate() {
            for (sweep_idx, sweep) in stage.variable_sweeps.iter().enumerate() {
                for bond in &sweep.truncation.bond_data {
                    // Check SV non-negativity
                    for (i, sv) in bond.singular_values.iter().enumerate() {
                        if sv.raw < 0 {
                            return Err(format!(
                                "Negative SV at stage {}, sweep {}, bond {}, sv {}: {}",
                                stage_idx, sweep_idx, bond.bond_index, i, sv.to_f64()
                            ));
                        }
                    }
                    // Check SV ordering
                    for pair in bond.singular_values.windows(2) {
                        if pair[0].raw < pair[1].raw {
                            return Err(format!(
                                "SV ordering violation at stage {}, sweep {}, bond {}: {} < {}",
                                stage_idx, sweep_idx, bond.bond_index,
                                pair[0].to_f64(), pair[1].to_f64()
                            ));
                        }
                    }
                }
            }
        }

        // Validate conservation
        let tolerance = self.params.conservation_tolerance;
        for (i, residual) in self.witness.conservation.residuals.iter().enumerate() {
            if residual.raw.abs() > tolerance.raw {
                return Err(format!(
                    "Conservation violation for variable {}: residual {} > tolerance {}",
                    i, residual.to_f64(), tolerance.to_f64()
                ));
            }
        }

        // Validate fixed-point MAC arithmetic
        for stage in &self.witness.strang_stages {
            for sweep in &stage.variable_sweeps {
                for site_data in &sweep.contraction.site_data {
                    for remainder in &site_data.fp_remainders {
                        if *remainder < 0 || *remainder >= super::config::Q16_SCALE as i64 {
                            return Err(format!(
                                "FP remainder out of range: {} (expected [0, {}))",
                                remainder,
                                super::config::Q16_SCALE,
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
