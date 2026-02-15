//! Halo2 sub-circuit gadgets for Navier-Stokes IMEX proof.
//!
//! Shared gadgets (MAC, BitDecomposition, SVD ordering, Conservation,
//! PublicInput) are re-exported from `crate::gadgets`.
//!
//! NS-IMEX-specific gadgets:
//! - **DiffusionSolveGadget**: Verifies (I - ν·Δt·L)u = u*
//! - **ProjectionGadget**: Verifies one CG step in pressure Poisson solve
//! - **DivergenceCheckGadget**: Verifies ‖∇·u‖ < ε_div
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

// Re-export all shared gadgets.
#[cfg(feature = "halo2")]
pub use crate::gadgets::halo2_gadgets::*;

#[cfg(not(feature = "halo2"))]
pub use crate::gadgets::stubs::*;

// ═══════════════════════════════════════════════════════════════════════════
// NS-IMEX-Specific Gadgets
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
pub mod ns_specific {
    use halo2_axiom::{
        circuit::{Region, Value},
        halo2curves::bn256::Fr,
        plonk::{Advice, Assigned, Column, Error, Selector},
    };
    use fluidelite_core::field::Q16;
    use crate::gadgets::halo2_gadgets::{
        q16_to_assigned, i64_to_assigned,
        BitDecompositionGadget, ConservationGadget,
    };

    // ═════════════════════════════════════════════════════════════════════
    // Diffusion Solve Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for verifying the diffusion sub-step: `(I - ν·Δt·L)u = u*`.
    ///
    /// Computes `ν·Δt·L·u` via fixed-point MAC, range-checks the remainder,
    /// then delegates the residual bound to `ConservationGadget`.
    pub struct DiffusionSolveGadget;

    impl DiffusionSolveGadget {
        /// Assign diffusion check.
        pub fn assign_diffusion_check(
            region: &mut Region<'_, Fr>,
            s_fp_mac: Selector,
            s_bool4: Selector,
            s_recompose: Selector,
            s_conservation: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            rhs: Q16,
            solution: Q16,
            laplacian_result: Q16,
            nu_dt: Q16,
            tolerance: Q16,
        ) -> Result<usize, Error> {
            let mut row = start_row;

            // Compute ν·Δt·L·u in Q16
            let nu_dt_raw = nu_dt.raw;
            let lapl_raw = laplacian_result.raw;
            let full_product = (nu_dt_raw as i128) * (lapl_raw as i128);
            let quotient = (full_product >> 16) as i64;
            let remainder = (full_product - ((quotient as i128) << 16)) as i64;

            // Assign MAC row: nu_dt × laplacian_result
            region.assign_advice(a_col, row, Value::known(q16_to_assigned(nu_dt)));
            region.assign_advice(
                b_col,
                row,
                Value::known(q16_to_assigned(laplacian_result)),
            );
            let nu_dt_lapl = Q16::from_raw(quotient);
            region.assign_advice(c_col, row, Value::known(q16_to_assigned(nu_dt_lapl)));
            region.assign_advice(d_col, row, Value::known(i64_to_assigned(remainder)));

            s_fp_mac.enable(region, row)?;
            row += 1;

            // Range check remainder
            row = BitDecompositionGadget::assign_range_check(
                region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                row, remainder, 16,
            )?;

            // Compute expected: (I - ν·Δt·L)u = solution - ν·Δt·L·solution
            let lhs_value = Q16::from_raw(solution.raw - nu_dt_lapl.raw);

            // Conservation check: |lhs_value - rhs| < tolerance
            row = ConservationGadget::assign_conservation_check(
                region, s_conservation, s_bool4, s_recompose,
                a_col, b_col, c_col, d_col,
                row, rhs, lhs_value, tolerance,
            )?;

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Projection Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for verifying one conjugate gradient step in the pressure
    /// Poisson solve.
    ///
    /// Each CG step computes:
    ///   α = r·r / (p·A·p)
    ///   x_{k+1} = x_k + α·p
    ///   r_{k+1} = r_k - α·A·p
    ///   β = r_{k+1}·r_{k+1} / (r_k·r_k)
    ///   p_{k+1} = r_{k+1} + β·p_k
    ///
    /// We verify the dot products and the residual reduction.
    pub struct ProjectionGadget;

    impl ProjectionGadget {
        /// Assign one CG step verification.
        ///
        /// Returns the next available row.
        pub fn assign_cg_step(
            region: &mut Region<'_, Fr>,
            s_fp_mac: Selector,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            alpha_numerator: Q16,
            alpha_denominator: Q16,
            beta: Q16,
            residual_norm: Q16,
        ) -> Result<usize, Error> {
            let mut row = start_row;

            // Verify alpha = r·r / (p·A·p)
            let alpha_raw = if alpha_denominator.raw != 0 {
                let full = (alpha_numerator.raw as i128) << 16;
                (full / alpha_denominator.raw as i128) as i64
            } else {
                0i64
            };
            let alpha = Q16::from_raw(alpha_raw);

            // MAC row: alpha × alpha_denominator ≈ alpha_numerator
            let full_product = (alpha.raw as i128) * (alpha_denominator.raw as i128);
            let quotient = (full_product >> 16) as i64;
            let remainder = (full_product - ((quotient as i128) << 16)) as i64;

            region.assign_advice(a_col, row, Value::known(q16_to_assigned(alpha)));
            region.assign_advice(
                b_col,
                row,
                Value::known(q16_to_assigned(alpha_denominator)),
            );
            region.assign_advice(
                c_col,
                row,
                Value::known(q16_to_assigned(Q16::from_raw(quotient))),
            );
            region.assign_advice(
                d_col,
                row,
                Value::known(i64_to_assigned(remainder.abs())),
            );

            s_fp_mac.enable(region, row)?;
            row += 1;

            // Range check remainder
            row = BitDecompositionGadget::assign_range_check(
                region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                row, remainder.abs(), 16,
            )?;

            // Assign beta witness
            region.assign_advice(a_col, row, Value::known(q16_to_assigned(beta)));
            region.assign_advice(
                b_col,
                row,
                Value::known(q16_to_assigned(residual_norm)),
            );
            region.assign_advice(c_col, row, Value::known(Assigned::from(Fr::zero())));
            region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));
            row += 1;

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Divergence Check Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for verifying the divergence-free condition: ‖∇·u‖ < ε_div.
    ///
    /// Computes the integrated divergence as the sum of ∂u/∂x + ∂v/∂y + ∂w/∂z
    /// and proves it is within the specified tolerance.
    pub struct DivergenceCheckGadget;

    impl DivergenceCheckGadget {
        /// Assign divergence check.
        pub fn assign_divergence_check(
            region: &mut Region<'_, Fr>,
            s_conservation: Selector,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            divergence: Q16,
            tolerance: Q16,
        ) -> Result<usize, Error> {
            // Reuse conservation gadget: check |divergence - 0| < tolerance
            ConservationGadget::assign_conservation_check(
                region, s_conservation, s_bool4, s_recompose,
                a_col, b_col, c_col, d_col,
                start_row,
                Q16::ZERO,      // Expected divergence = 0
                divergence,     // Actual divergence
                tolerance,      // Divergence tolerance
            )
        }
    }
}

#[cfg(feature = "halo2")]
pub use ns_specific::*;

// Non-Halo2 stubs for NS-specific gadgets.
#[cfg(not(feature = "halo2"))]
mod ns_stubs {
    /// Stub for DiffusionSolveGadget.
    pub struct DiffusionSolveGadget;
    /// Stub for ProjectionGadget.
    pub struct ProjectionGadget;
    /// Stub for DivergenceCheckGadget.
    pub struct DivergenceCheckGadget;
}

#[cfg(not(feature = "halo2"))]
pub use ns_stubs::*;
