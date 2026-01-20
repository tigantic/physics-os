//! Halo2 Circuit Implementation for FluidElite
//!
//! This module defines the constraint system for ZK-provable FluidElite inference.
//!
//! # Circuit Structure
//!
//! The circuit uses custom gates for efficient tensor operations:
//! 1. **Multiplication Gate**: a × b = c (1 constraint)
//! 2. **MAC Gate**: a × b + c = d (1 constraint)  
//! 3. **Copy Constraint**: a = b (free via permutation)
//!
//! # Columns
//!
//! | Column | Type   | Purpose                          |
//! |--------|--------|----------------------------------|
//! | a      | Advice | First operand / accumulator      |
//! | b      | Advice | Second operand                   |
//! | c      | Advice | Result                           |
//! | s_mul  | Select | Enable multiplication gate       |
//! | s_mac  | Select | Enable MAC gate                  |
//! | public | Instan | Public inputs (token, logits)    |

pub mod config;
pub mod gadgets;
pub mod hybrid_lookup;
pub mod hybrid_unified;

#[cfg(feature = "halo2")]
mod halo2_impl;

#[cfg(feature = "halo2")]
pub use halo2_impl::*;

#[cfg(feature = "halo2")]
pub use hybrid_lookup::{HybridLookupCircuit, HybridLookupConfig, FallbackConfig};

#[cfg(feature = "halo2")]
pub use hybrid_unified::{
    ArithmeticConfig, UnifiedHybridConfig, TokenWitness, BatchedHybridCircuit,
};

// Stub types when Halo2 is not enabled
#[cfg(not(feature = "halo2"))]
mod stub {
    use crate::config as model_config;
    use crate::field::Q16;
    use crate::mpo::MPO;
    use crate::mps::MPS;

    /// Stub circuit when Halo2 is not enabled
    #[derive(Clone)]
    pub struct FluidEliteCircuit {
        /// Token ID (public input)
        pub token_id: u64,
        /// Current context MPS
        pub context: MPS,
        /// W_hidden MPO weights
        pub w_hidden: MPO,
        /// W_input MPO weights
        pub w_input: MPO,
        /// Readout weights
        pub readout_weights: Vec<Q16>,
        /// Expected output logits
        pub expected_logits: Vec<Q16>,
    }

    impl FluidEliteCircuit {
        /// Create a new circuit instance (stub)
        pub fn new(
            token_id: u64,
            context: MPS,
            w_hidden: MPO,
            w_input: MPO,
            readout_weights: Vec<Q16>,
        ) -> Self {
            let new_context = crate::ops::fluidelite_step(
                &context,
                token_id as usize,
                &w_hidden,
                &w_input,
                model_config::CHI,
            );
            let expected_logits =
                crate::ops::readout(&new_context, &readout_weights, model_config::VOCAB_SIZE);

            Self {
                token_id,
                context,
                w_hidden,
                w_input,
                readout_weights,
                expected_logits,
            }
        }
    }
}

#[cfg(not(feature = "halo2"))]
pub use stub::*;

// Tests that work without Halo2
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpo::MPO;
    use crate::mps::MPS;
    use crate::field::Q16;

    #[test]
    fn test_circuit_construction() {
        let context = MPS::new(4, 4, 2);
        let w_hidden = MPO::identity(4, 2);
        let w_input = MPO::identity(4, 2);
        let readout_weights = vec![Q16::from_f64(0.1); 4 * 64];

        let circuit = FluidEliteCircuit::new(5, context, w_hidden, w_input, readout_weights);

        assert_eq!(circuit.token_id, 5);
    }
}
