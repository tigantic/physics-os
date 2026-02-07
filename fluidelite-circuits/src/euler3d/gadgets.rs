//! Halo2 sub-circuit gadgets for Euler 3D proof.
//!
//! Each gadget encapsulates a reusable constraint pattern:
//!
//! - **FixedPointMACGadget**: Q16.16 multiply-accumulate with range-checked remainder
//! - **BitDecompositionGadget**: Proves a value decomposes into boolean bits
//! - **SvdOrderingGadget**: Proves singular values are non-negative and descending
//! - **ConservationGadget**: Proves conservation residuals are within tolerance
//! - **PublicInputGadget**: Binds witness values to public instance column
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
    // Fixed-Point MAC Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for Q16.16 fixed-point multiply-accumulate.
    ///
    /// # Gate
    ///
    /// The `fp_mac` gate constrains:
    /// ```text
    /// a_cur × b_cur = (c_cur - c_prev) × SCALE + d_cur
    /// ```
    /// Where:
    /// - `a_cur`, `b_cur`: Q16 multiplicands (raw values as field elements)
    /// - `c_prev`: previous accumulator value
    /// - `c_cur`: new accumulator value (= c_prev + floor(a*b / SCALE))
    /// - `d_cur`: remainder ∈ [0, SCALE), range-checked via bit decomposition
    ///
    /// This ensures the fixed-point multiplication and accumulation is correct.
    pub struct FixedPointMACGadget;

    impl FixedPointMACGadget {
        /// Assign a single fixed-point MAC row.
        ///
        /// Returns the row after the MAC row (caller must then assign
        /// range check rows for the remainder).
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
        ///
        /// This assigns:
        /// - Row 0: c = 0 (initial accumulator)
        /// - Rows 1..n: MAC + range check rows
        ///
        /// Returns (next_row, final_accumulator).
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
            assert_eq!(
                remainders.len(),
                a_vals.len(),
                "Remainder count mismatch"
            );

            let mut row = start_row;

            // Row 0: Initialize accumulator to zero
            region.assign_advice(a_col, row, Value::known(Assigned::from(Fr::zero())));
            region.assign_advice(b_col, row, Value::known(Assigned::from(Fr::zero())));
            region.assign_advice(
                c_col,
                row,
                Value::known(q16_to_assigned(accumulators[0])),
            );
            region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));
            row += 1;

            // MAC rows + range checks
            for i in 0..a_vals.len() {
                // MAC row
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

                // Range check for remainder
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
                    16, // Q16 fractional bits
                )?;
            }

            let final_acc = *accumulators.last().unwrap_or(&Q16::zero());
            Ok((row, final_acc))
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Bit Decomposition Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for proving a value decomposes into boolean bits.
    ///
    /// Used for range checks (proving remainder ∈ [0, 2^n)).
    ///
    /// # Layout
    ///
    /// Uses 4 advice columns to check 4 bits per row.
    /// For n-bit range check: ceil(n/4) boolean rows + 1 recompose row.
    ///
    /// ## Boolean rows (s_bool4 enabled):
    /// ```text
    /// a = b_0, b = b_1, c = b_2, d = b_3
    /// Constraint: a(1-a) = 0, b(1-b) = 0, c(1-c) = 0, d(1-d) = 0
    /// ```
    ///
    /// ## Recompose row (s_recompose enabled):
    /// ```text
    /// a = target_value, b = reconstructed_value
    /// Constraint: a - b = 0
    /// ```
    pub struct BitDecompositionGadget;

    impl BitDecompositionGadget {
        /// Assign a range check: prove that `value` ∈ [0, 2^num_bits).
        ///
        /// Returns the next available row.
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

            // Decompose value into bits (LSB first)
            let bits: Vec<bool> = if value >= 0 {
                let v = value as u64;
                (0..num_bits).map(|i| (v >> i) & 1 == 1).collect()
            } else {
                // Negative values will fail the recomposition check (soundness!)
                vec![false; num_bits]
            };

            // Assign boolean rows (4 bits per row)
            let chunks: Vec<&[bool]> = bits.chunks(4).collect();
            for chunk in &chunks {
                let b0 = if !chunk.is_empty() {
                    Fr::from(chunk[0] as u64)
                } else {
                    Fr::zero()
                };
                let b1 = if chunk.len() > 1 {
                    Fr::from(chunk[1] as u64)
                } else {
                    Fr::zero()
                };
                let b2 = if chunk.len() > 2 {
                    Fr::from(chunk[2] as u64)
                } else {
                    Fr::zero()
                };
                let b3 = if chunk.len() > 3 {
                    Fr::from(chunk[3] as u64)
                } else {
                    Fr::zero()
                };

                region.assign_advice(a_col, row, Value::known(Assigned::from(b0)));
                region.assign_advice(b_col, row, Value::known(Assigned::from(b1)));
                region.assign_advice(c_col, row, Value::known(Assigned::from(b2)));
                region.assign_advice(d_col, row, Value::known(Assigned::from(b3)));

                s_bool4.enable(region, row)?;
                row += 1;
            }

            // Recompose row: verify bits reconstruct to the original value
            let reconstructed: u64 = bits
                .iter()
                .enumerate()
                .map(|(i, &b)| if b { 1u64 << i } else { 0 })
                .sum();

            region.assign_advice(
                a_col,
                row,
                Value::known(i64_to_assigned(value)),
            );
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

        /// Assign a non-negativity proof: prove that `value` ≥ 0.
        ///
        /// Decomposes `value` into `num_bits` boolean bits and verifies
        /// they reconstruct to the original. If `value < 0`, the field
        /// representation won't match any boolean decomposition → unsatisfiable.
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
                region,
                s_bool4,
                s_recompose,
                a_col,
                b_col,
                c_col,
                d_col,
                start_row,
                value,
                num_bits,
            )
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // SVD Ordering Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for proving singular value ordering: s_i ≥ s_{i+1} ≥ 0.
    ///
    /// For each adjacent pair (s_i, s_{i+1}):
    /// 1. Compute delta = s_i - s_{i+1}
    /// 2. Prove delta ≥ 0 via bit decomposition
    ///
    /// For each singular value s_i:
    /// 1. Prove s_i ≥ 0 via bit decomposition
    pub struct SvdOrderingGadget;

    impl SvdOrderingGadget {
        /// Assign SVD ordering constraints for a set of singular values.
        ///
        /// Returns the next available row.
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

            // Prove non-negativity of each singular value
            for (i, sv) in singular_values.iter().enumerate() {
                // Assign the singular value
                region.assign_advice(
                    a_col,
                    row,
                    Value::known(q16_to_assigned(*sv)),
                );
                region.assign_advice(b_col, row, Value::known(Assigned::from(Fr::zero())));
                region.assign_advice(c_col, row, Value::known(Assigned::from(Fr::zero())));
                region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));
                row += 1;

                // Bit decomposition for non-negativity
                let num_bits = if i < nonneg_bits.len() {
                    nonneg_bits[i].len()
                } else {
                    32
                };
                row = BitDecompositionGadget::assign_nonneg_check(
                    region,
                    s_bool4,
                    s_recompose,
                    a_col,
                    b_col,
                    c_col,
                    d_col,
                    row,
                    sv.raw,
                    num_bits,
                )?;
            }

            // Prove ordering: s_i >= s_{i+1}
            for i in 0..singular_values.len().saturating_sub(1) {
                let delta = singular_values[i].raw - singular_values[i + 1].raw;

                // Assign the ordering row
                region.assign_advice(
                    a_col,
                    row,
                    Value::known(q16_to_assigned(singular_values[i])),
                );
                region.assign_advice(
                    b_col,
                    row,
                    Value::known(q16_to_assigned(singular_values[i + 1])),
                );
                region.assign_advice(
                    c_col,
                    row,
                    Value::known(i64_to_assigned(delta)),
                );
                region.assign_advice(d_col, row, Value::known(Assigned::from(Fr::zero())));

                s_sv_order.enable(region, row)?;
                row += 1;

                // Bit decomposition for delta >= 0
                let num_bits = if i < ordering_bits.len() {
                    ordering_bits[i].len()
                } else {
                    32
                };
                row = BitDecompositionGadget::assign_nonneg_check(
                    region,
                    s_bool4,
                    s_recompose,
                    a_col,
                    b_col,
                    c_col,
                    d_col,
                    row,
                    delta,
                    num_bits,
                )?;
            }

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Conservation Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for proving conservation laws.
    ///
    /// Proves: |integral_after - integral_before| ≤ tolerance
    ///
    /// Implementation:
    /// 1. Compute residual = integral_after - integral_before
    /// 2. Compute bound_pos = tolerance - residual (must be ≥ 0)
    /// 3. Compute bound_neg = tolerance + residual (must be ≥ 0)
    /// 4. Prove both bounds are non-negative
    pub struct ConservationGadget;

    impl ConservationGadget {
        /// Assign conservation check for one conserved variable.
        ///
        /// Returns the next available row.
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
            let bound_pos = tolerance.raw - residual; // tolerance - residual >= 0
            let bound_neg = tolerance.raw + residual; // tolerance + residual >= 0

            // Assign the conservation row
            region.assign_advice(
                a_col,
                row,
                Value::known(q16_to_assigned(integral_before)),
            );
            region.assign_advice(
                b_col,
                row,
                Value::known(q16_to_assigned(integral_after)),
            );
            region.assign_advice(
                c_col,
                row,
                Value::known(i64_to_assigned(residual)),
            );
            region.assign_advice(
                d_col,
                row,
                Value::known(q16_to_assigned(tolerance)),
            );

            s_conservation.enable(region, row)?;
            row += 1;

            // Prove bound_pos >= 0 (tolerance >= residual)
            row = BitDecompositionGadget::assign_nonneg_check(
                region,
                s_bool4,
                s_recompose,
                a_col,
                b_col,
                c_col,
                d_col,
                row,
                bound_pos,
                32,
            )?;

            // Prove bound_neg >= 0 (tolerance >= -residual)
            row = BitDecompositionGadget::assign_nonneg_check(
                region,
                s_bool4,
                s_recompose,
                a_col,
                b_col,
                c_col,
                d_col,
                row,
                bound_neg,
                32,
            )?;

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Public Input Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for binding witness values to public instance column.
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
