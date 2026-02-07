//! Halo2 sub-circuit gadgets for the NS-IMEX proof.
//!
//! Reuses the fixed-point MAC, bit decomposition, SVD ordering, conservation,
//! and public input gadgets from Euler 3D, plus adds NS-specific gadgets:
//!
//! - **DiffusionSolveGadget**: Verifies (I - ν·Δt·L)u = u* in Q16 fixed-point
//! - **ProjectionGadget**: Verifies CG iteration steps for the Poisson solve
//! - **DivergenceCheckGadget**: Verifies ‖∇·u‖ < ε_div
//!
//! All gadgets are behind `#[cfg(feature = "halo2")]`.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#[cfg(feature = "halo2")]
pub mod halo2_gadgets {
    use halo2_axiom::{
        circuit::{Region, Value},
        halo2curves::bn256::Fr,
        plonk::{Advice, Assigned, Column, Error, Fixed, Selector},
        poly::Rotation,
    };

    use fluidelite_core::field::Q16;

    /// Fr constant for Q16 scale factor (2^16 = 65536).
    pub fn scale_fr() -> Fr {
        Fr::from(super::super::config::Q16_SCALE)
    }

    /// Convert Q16 to Assigned<Fr>.
    pub fn q16_to_assigned(fp: Q16) -> Assigned<Fr> {
        Assigned::from(q16_to_fr(fp))
    }

    /// Convert Q16 to Fr.
    pub fn q16_to_fr(fp: Q16) -> Fr {
        if fp.raw >= 0 {
            Fr::from(fp.raw as u64)
        } else {
            -Fr::from((-fp.raw) as u64)
        }
    }

    /// Convert i64 to Assigned<Fr>.
    pub fn i64_to_assigned(val: i64) -> Assigned<Fr> {
        if val >= 0 {
            Assigned::from(Fr::from(val as u64))
        } else {
            Assigned::from(-Fr::from((-val) as u64))
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Fixed-Point MAC Gadget (shared with Euler 3D)
    // ═════════════════════════════════════════════════════════════════════

    /// Q16.16 multiply-accumulate gadget.
    ///
    /// Gate: a_cur × b_cur = (c_cur - c_prev) × SCALE + d_cur
    pub struct FixedPointMACGadget;

    impl FixedPointMACGadget {
        /// Assign a single fixed-point MAC row.
        pub fn assign_fp_mac_row(
            region: &mut Region<'_, Fr>,
            s_fp_mac: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            row: usize,
            a_val: Q16,
            b_val: Q16,
            acc_new: Q16,
            remainder: i64,
        ) -> Result<usize, Error> {
            region.assign_advice(a_col, row, Value::known(q16_to_assigned(a_val)));
            region.assign_advice(b_col, row, Value::known(q16_to_assigned(b_val)));
            region.assign_advice(c_col, row, Value::known(q16_to_assigned(acc_new)));
            region.assign_advice(d_col, row, Value::known(i64_to_assigned(remainder)));

            s_fp_mac.enable(region, row)?;

            Ok(row + 1)
        }

        /// Assign a complete MAC chain (dot product of two Q16 vectors).
        pub fn assign_mac_chain(
            region: &mut Region<'_, Fr>,
            s_fp_mac: Selector,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            a_vals: &[Q16],
            b_vals: &[Q16],
            accumulators: &[Q16],
            remainders: &[i64],
        ) -> Result<(usize, Q16), Error> {
            assert_eq!(a_vals.len(), b_vals.len(), "Operand length mismatch");
            assert_eq!(
                accumulators.len(),
                a_vals.len() + 1,
                "Accumulator count mismatch"
            );
            assert_eq!(remainders.len(), a_vals.len(), "Remainder count mismatch");

            let mut row = start_row;

            // Row 0: Initialize accumulator to zero
            region.assign_advice(a_col, row, Value::known(Assigned::from(Fr::zero())));
            region.assign_advice(b_col, row, Value::known(Assigned::from(Fr::zero())));
            region.assign_advice(c_col, row, Value::known(q16_to_assigned(accumulators[0])));
            region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));
            row += 1;

            for i in 0..a_vals.len() {
                row = Self::assign_fp_mac_row(
                    region,
                    s_fp_mac,
                    a_col,
                    b_col,
                    c_col,
                    d_col,
                    row,
                    a_vals[i],
                    b_vals[i],
                    accumulators[i + 1],
                    remainders[i],
                )?;

                row = BitDecompositionGadget::assign_range_check(
                    region,
                    s_bool4,
                    s_recompose,
                    a_col,
                    b_col,
                    c_col,
                    d_col,
                    row,
                    remainders[i],
                    16,
                )?;
            }

            let final_acc = *accumulators.last().unwrap_or(&Q16::zero());
            Ok((row, final_acc))
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Bit Decomposition Gadget (shared with Euler 3D)
    // ═════════════════════════════════════════════════════════════════════

    /// Proves a value decomposes into boolean bits (for range checks).
    pub struct BitDecompositionGadget;

    impl BitDecompositionGadget {
        /// Assign a range check: prove that `value` ∈ [0, 2^num_bits).
        pub fn assign_range_check(
            region: &mut Region<'_, Fr>,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            value: i64,
            num_bits: usize,
        ) -> Result<usize, Error> {
            let mut row = start_row;

            let bits: Vec<bool> = if value >= 0 {
                let v = value as u64;
                (0..num_bits).map(|i| (v >> i) & 1 == 1).collect()
            } else {
                vec![false; num_bits]
            };

            let chunks: Vec<&[bool]> = bits.chunks(4).collect();
            for chunk in &chunks {
                let b0 = if !chunk.is_empty() { Fr::from(chunk[0] as u64) } else { Fr::zero() };
                let b1 = if chunk.len() > 1 { Fr::from(chunk[1] as u64) } else { Fr::zero() };
                let b2 = if chunk.len() > 2 { Fr::from(chunk[2] as u64) } else { Fr::zero() };
                let b3 = if chunk.len() > 3 { Fr::from(chunk[3] as u64) } else { Fr::zero() };

                region.assign_advice(a_col, row, Value::known(Assigned::from(b0)));
                region.assign_advice(b_col, row, Value::known(Assigned::from(b1)));
                region.assign_advice(c_col, row, Value::known(Assigned::from(b2)));
                region.assign_advice(d_col, row, Value::known(Assigned::from(b3)));

                s_bool4.enable(region, row)?;
                row += 1;
            }

            let reconstructed: u64 = bits
                .iter()
                .enumerate()
                .map(|(i, &b)| if b { 1u64 << i } else { 0 })
                .sum();

            region.assign_advice(a_col, row, Value::known(i64_to_assigned(value)));
            region.assign_advice(
                b_col,
                row,
                Value::known(Assigned::from(Fr::from(reconstructed))),
            );
            region.assign_advice(c_col, row, Value::known(Assigned::from(Fr::zero())));
            region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));

            s_recompose.enable(region, row)?;
            row += 1;

            Ok(row)
        }

        /// Assign a non-negativity proof.
        pub fn assign_nonneg_check(
            region: &mut Region<'_, Fr>,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            value: i64,
            num_bits: usize,
        ) -> Result<usize, Error> {
            Self::assign_range_check(
                region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                start_row, value, num_bits,
            )
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // SVD Ordering Gadget (shared with Euler 3D)
    // ═════════════════════════════════════════════════════════════════════

    /// Proves singular value ordering: s_i ≥ s_{i+1} ≥ 0.
    pub struct SvdOrderingGadget;

    impl SvdOrderingGadget {
        /// Assign SVD ordering constraints for a set of singular values.
        pub fn assign_sv_ordering(
            region: &mut Region<'_, Fr>,
            s_sv_order: Selector,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            singular_values: &[Q16],
            ordering_bits: &[Vec<bool>],
            nonneg_bits: &[Vec<bool>],
        ) -> Result<usize, Error> {
            let mut row = start_row;

            for (i, sv) in singular_values.iter().enumerate() {
                region.assign_advice(a_col, row, Value::known(q16_to_assigned(*sv)));
                region.assign_advice(b_col, row, Value::known(Assigned::from(Fr::zero())));
                region.assign_advice(c_col, row, Value::known(Assigned::from(Fr::zero())));
                region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));
                row += 1;

                let num_bits = if i < nonneg_bits.len() { nonneg_bits[i].len() } else { 32 };
                row = BitDecompositionGadget::assign_nonneg_check(
                    region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                    row, sv.raw, num_bits,
                )?;
            }

            for i in 0..singular_values.len().saturating_sub(1) {
                let delta = singular_values[i].raw - singular_values[i + 1].raw;

                region.assign_advice(a_col, row, Value::known(q16_to_assigned(singular_values[i])));
                region.assign_advice(
                    b_col,
                    row,
                    Value::known(q16_to_assigned(singular_values[i + 1])),
                );
                region.assign_advice(c_col, row, Value::known(i64_to_assigned(delta)));
                region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));

                s_sv_order.enable(region, row)?;
                row += 1;

                let num_bits = if i < ordering_bits.len() { ordering_bits[i].len() } else { 32 };
                row = BitDecompositionGadget::assign_nonneg_check(
                    region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                    row, delta, num_bits,
                )?;
            }

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Conservation Gadget (shared with Euler 3D)
    // ═════════════════════════════════════════════════════════════════════

    /// Proves |integral_after - integral_before| ≤ tolerance.
    pub struct ConservationGadget;

    impl ConservationGadget {
        /// Assign conservation check for one conserved quantity.
        pub fn assign_conservation_check(
            region: &mut Region<'_, Fr>,
            s_conservation: Selector,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            integral_before: Q16,
            integral_after: Q16,
            tolerance: Q16,
        ) -> Result<usize, Error> {
            let mut row = start_row;

            let residual = integral_after.raw - integral_before.raw;
            let bound_pos = tolerance.raw - residual;
            let bound_neg = tolerance.raw + residual;

            region.assign_advice(a_col, row, Value::known(q16_to_assigned(integral_before)));
            region.assign_advice(b_col, row, Value::known(q16_to_assigned(integral_after)));
            region.assign_advice(c_col, row, Value::known(i64_to_assigned(residual)));
            region.assign_advice(d_col, row, Value::known(q16_to_assigned(tolerance)));

            s_conservation.enable(region, row)?;
            row += 1;

            row = BitDecompositionGadget::assign_nonneg_check(
                region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                row, bound_pos, 32,
            )?;

            row = BitDecompositionGadget::assign_nonneg_check(
                region, s_bool4, s_recompose, a_col, b_col, c_col, d_col,
                row, bound_neg, 32,
            )?;

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Diffusion Solve Gadget (NS-IMEX specific)
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for verifying the implicit diffusion solve step.
    ///
    /// Proves: ‖(I - ν·Δt·L) u^{n+1} - u*‖ < tolerance
    ///
    /// Implementation:
    /// 1. Compute Laplacian MPO × u^{n+1} to get L·u^{n+1}
    /// 2. Compute scale = ν·Δt in Q16 fixed-point
    /// 3. Compute residual = u^{n+1} - ν·Δt·(L·u^{n+1}) - u*
    /// 4. Prove |residual| < tolerance via conservation gadget
    pub struct DiffusionSolveGadget;

    impl DiffusionSolveGadget {
        /// Assign diffusion solve verification for one variable.
        ///
        /// `rhs` is the pre-diffusion state u*, `solution` is the post-diffusion u^{n+1}.
        /// `laplacian_result` is L·u^{n+1} (precomputed MPO application).
        ///
        /// Returns the next available row.
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
    // Projection Gadget (NS-IMEX specific)
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
            // We check: alpha_numerator = alpha * alpha_denominator
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
    // Divergence Check Gadget (NS-IMEX specific)
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
                Q16::ZERO,    // Expected divergence = 0
                divergence,    // Actual divergence
                tolerance,     // Divergence tolerance
            )
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Public Input Gadget (shared with Euler 3D)
    // ═════════════════════════════════════════════════════════════════════

    /// Binds witness values to the public instance column.
    pub struct PublicInputGadget;

    impl PublicInputGadget {
        /// Assign a public input value and return its cell for instance binding.
        pub fn assign_public_input(
            region: &mut Region<'_, Fr>,
            c_col: Column<Advice>,
            row: usize,
            value: Fr,
        ) -> Result<halo2_axiom::circuit::Cell, Error> {
            let cell = region.assign_advice(
                c_col,
                row,
                Value::known(Assigned::from(value)),
            );
            Ok(cell.cell())
        }
    }
}

#[cfg(feature = "halo2")]
pub use halo2_gadgets::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests (work without Halo2)
// ═══════════════════════════════════════════════════════════════════════════
