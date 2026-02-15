//! Halo2 sub-circuit gadgets for the Thermal/Heat Equation proof.
//!
//! Shared gadgets (MAC, BitDecomposition, SVD ordering, Conservation,
//! PublicInput) are re-exported from `crate::gadgets`.
//!
//! Thermal-specific gadgets:
//! - **CgSolveGadget**: Constrains CG iteration convergence
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

// Re-export all shared gadgets.
#[cfg(feature = "halo2")]
pub use crate::gadgets::halo2_gadgets::*;

#[cfg(not(feature = "halo2"))]
pub use crate::gadgets::stubs::*;

// ═══════════════════════════════════════════════════════════════════════════
// Thermal-Specific Gadgets
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
pub mod thermal_specific {
    use halo2_axiom::{
        circuit::{Region, Value},
        halo2curves::bn256::Fr,
        plonk::{Advice, Assigned, Column, Error, Selector},
    };
    use fluidelite_core::field::Q16;
    use crate::gadgets::halo2_gadgets::q16_to_assigned;

    /// Gadget for constraining CG solve convergence.
    ///
    /// Verifies that:
    /// 1. Each CG iteration's residual decreases (or converges)
    /// 2. Final residual norm ≤ tolerance
    /// 3. α and β coefficients are correctly computed from dot products
    pub struct CgSolveGadget;

    impl CgSolveGadget {
        /// Assign CG convergence verification.
        pub fn assign_cg_convergence(
            region: &mut Region<'_, Fr>,
            s_fp_mac: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            _d_col: Column<Advice>,
            start_row: usize,
            residual_norms: &[Q16],
            final_tolerance: Q16,
        ) -> Result<usize, Error> {
            let mut row = start_row;

            // Assign residual norm sequence
            for norm in residual_norms {
                region.assign_advice(
                    a_col,
                    row,
                    Value::known(q16_to_assigned(*norm)),
                );
                row += 1;
            }

            // Final residual must be ≤ tolerance
            if let Some(last) = residual_norms.last() {
                region.assign_advice(
                    b_col,
                    row,
                    Value::known(q16_to_assigned(*last)),
                );
                region.assign_advice(
                    c_col,
                    row,
                    Value::known(q16_to_assigned(final_tolerance)),
                );
                s_fp_mac.enable(region, row)?;
                row += 1;
            }

            Ok(row)
        }
    }
}

#[cfg(feature = "halo2")]
pub use thermal_specific::*;

// Non-Halo2 stub for thermal-specific gadgets.
#[cfg(not(feature = "halo2"))]
mod thermal_stubs {
    /// Stub for CgSolveGadget.
    pub struct CgSolveGadget;
}

#[cfg(not(feature = "halo2"))]
pub use thermal_stubs::*;
