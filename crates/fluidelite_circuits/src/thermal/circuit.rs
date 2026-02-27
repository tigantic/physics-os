//! Thermal circuit: witness generation and structural validation.
//!
//! Generates the complete witness and validates constraints without
//! a cryptographic proving backend. The real ZK proof for thermal
//! uses the STARK AIR in `stark_impl.rs`.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::config::{ThermalCircuitSizing, ThermalParams};
use super::witness::{ThermalWitness, WitnessGenerator};

/// Thermal circuit: generates witness and validates constraints.
///
/// Generates and validates the witness but does not create
/// a cryptographic proof directly. The STARK prover in `stark_impl.rs`
/// produces verification-ready proofs; this struct provides the
/// circuit-level witness for either path.
#[derive(Clone, Debug)]
pub struct ThermalCircuit {
    /// Witness data.
    pub witness: ThermalWitness,

    /// Circuit sizing.
    pub sizing: ThermalCircuitSizing,
}

impl ThermalCircuit {
    /// Create a new circuit from parameters and inputs.
    pub fn new(
        params: ThermalParams,
        input_states: &[MPS],
        laplacian_mpos: &[MPO],
    ) -> Result<Self, String> {
        let sizing = ThermalCircuitSizing::from_params(&params);
        let gen = WitnessGenerator::new(params);
        let witness = gen
            .generate(input_states, laplacian_mpos)
            .map_err(|e| format!("Witness generation failed: {}", e))?;

        Ok(Self { witness, sizing })
    }

    /// Get circuit k.
    pub fn k(&self) -> u32 {
        self.sizing.k
    }

    /// Estimate constraints.
    pub fn estimate_constraints(&self) -> usize {
        self.sizing.estimate_constraints()
    }

    /// Validate the witness (checks conservation and SVD ordering).
    ///
    /// Returns Ok(()) if all constraints pass, or an error describing
    /// the first failed constraint.
    pub fn validate_witness(&self) -> Result<(), String> {
        // Check conservation residual is within tolerance
        let residual = self.witness.conservation.residual;
        let tol = self.witness.params.conservation_tol;

        if residual.raw.abs() > tol.raw {
            return Err(format!(
                "Conservation violation: |{}| > {}",
                residual.to_f64(),
                tol.to_f64(),
            ));
        }

        // Check SVD ordering
        for bond in &self.witness.truncation.bond_data {
            for pair in bond.singular_values.windows(2) {
                if pair[0].raw < pair[1].raw {
                    return Err(format!(
                        "SVD ordering violation at bond {}: {} < {}",
                        bond.bond_index,
                        pair[0].to_f64(),
                        pair[1].to_f64(),
                    ));
                }
            }
        }

        Ok(())
    }
}
