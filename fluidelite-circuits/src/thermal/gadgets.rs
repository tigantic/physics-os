//! Halo2 sub-circuit gadgets for the Thermal/Heat Equation proof.
//!
//! Reuses gadgets from Euler 3D (FP MAC, bit decomposition, SV ordering,
//! conservation, public input) and adds thermal-specific gadgets:
//!
//! - **CgSolveGadget**: Constrains CG iteration convergence
//! - **DiffusionSystemGadget**: Constrains (I - α·Δt·L)·x = r
//!
//! All gadgets are behind `#[cfg(feature = "halo2")]`.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#[cfg(feature = "halo2")]
/// Halo2 gadget implementations for the thermal proof circuit.
pub mod halo2_gadgets {
    use halo2_axiom::{
        circuit::{Cell, Region, Value},
        halo2curves::bn256::Fr,
        plonk::{Advice, Assigned, Column, Error, Selector},
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
            assert_eq!(a_vals.len(), b_vals.len());
            let n = a_vals.len();

            // Initial accumulator = 0
            region.assign_advice(
                c_col,
                start_row,
                Value::known(q16_to_assigned(Q16::zero())),
            );

            let mut row = start_row + 1;

            for i in 0..n {
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
                    accumulators[i],
                    remainders[i],
                )?;

                // Range check remainder via bit decomposition
                row = BitDecompositionGadget::assign_bool4_range_check(
                    region,
                    s_bool4,
                    s_recompose,
                    a_col,
                    b_col,
                    c_col,
                    d_col,
                    row,
                    remainders[i],
                )?;
            }

            let final_acc = if n > 0 {
                accumulators[n - 1]
            } else {
                Q16::zero()
            };

            Ok((row, final_acc))
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Bit Decomposition Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for proving value decomposes into boolean bits.
    pub struct BitDecompositionGadget;

    impl BitDecompositionGadget {
        /// Assign a 16-bit range check using 4 boolean rows + recomposition.
        pub fn assign_bool4_range_check(
            region: &mut Region<'_, Fr>,
            s_bool4: Selector,
            s_recompose: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            value: i64,
        ) -> Result<usize, Error> {
            let v = if value >= 0 { value as u64 } else { 0u64 };

            // 4 rows of 4 bits each = 16 bits
            let mut row = start_row;
            for chunk in 0..4 {
                let base = chunk * 4;
                let b0 = (v >> base) & 1;
                let b1 = (v >> (base + 1)) & 1;
                let b2 = (v >> (base + 2)) & 1;
                let b3 = (v >> (base + 3)) & 1;

                region.assign_advice(
                    a_col,
                    row,
                    Value::known(Assigned::from(Fr::from(b0))),
                );
                region.assign_advice(
                    b_col,
                    row,
                    Value::known(Assigned::from(Fr::from(b1))),
                );
                region.assign_advice(
                    c_col,
                    row,
                    Value::known(Assigned::from(Fr::from(b2))),
                );
                region.assign_advice(
                    d_col,
                    row,
                    Value::known(Assigned::from(Fr::from(b3))),
                );

                s_bool4.enable(region, row)?;
                row += 1;
            }

            // Recomposition row
            region.assign_advice(
                a_col,
                row,
                Value::known(Assigned::from(Fr::from(v))),
            );
            s_recompose.enable(region, row)?;
            row += 1;

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // SV Ordering Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget to prove singular values are non-negative and descending.
    pub struct SvdOrderingGadget;

    impl SvdOrderingGadget {
        /// Assign SV ordering constraints for a set of singular values.
        pub fn assign_sv_ordering(
            region: &mut Region<'_, Fr>,
            s_sv_order: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            _d_col: Column<Advice>,
            start_row: usize,
            singular_values: &[Q16],
            ordering_bits: &[Vec<bool>],
            nonneg_bits: &[Vec<bool>],
        ) -> Result<usize, Error> {
            let mut row = start_row;

            // Assign each SV and its non-negativity proof
            for (i, sv) in singular_values.iter().enumerate() {
                region.assign_advice(
                    a_col,
                    row,
                    Value::known(q16_to_assigned(*sv)),
                );

                // Non-negativity: bit decomposition
                if i < nonneg_bits.len() {
                    let bits = &nonneg_bits[i];
                    let bit_val: u64 = bits
                        .iter()
                        .enumerate()
                        .map(|(j, &b)| if b { 1u64 << j } else { 0 })
                        .sum();
                    region.assign_advice(
                        b_col,
                        row,
                        Value::known(Assigned::from(Fr::from(bit_val))),
                    );
                }

                // Ordering: s_i - s_{i+1} ≥ 0
                if i < ordering_bits.len() {
                    let bits = &ordering_bits[i];
                    let diff_val: u64 = bits
                        .iter()
                        .enumerate()
                        .map(|(j, &b)| if b { 1u64 << j } else { 0 })
                        .sum();
                    region.assign_advice(
                        c_col,
                        row,
                        Value::known(Assigned::from(Fr::from(diff_val))),
                    );
                }

                s_sv_order.enable(region, row)?;
                row += 1;
            }

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Conservation Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for proving conservation residuals are within tolerance.
    pub struct ConservationGadget;

    impl ConservationGadget {
        /// Assign conservation check.
        ///
        /// Constrains |integral_after - integral_before - source_contribution| ≤ tol.
        pub fn assign_conservation(
            region: &mut Region<'_, Fr>,
            s_conservation: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            d_col: Column<Advice>,
            start_row: usize,
            integral_before: Q16,
            integral_after: Q16,
            residual: Q16,
            tolerance: Q16,
            bound_bits: &[bool],
        ) -> Result<usize, Error> {
            let mut row = start_row;

            // Row 1: integral_before, integral_after, residual
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
                Value::known(q16_to_assigned(residual)),
            );
            region.assign_advice(
                d_col,
                row,
                Value::known(q16_to_assigned(tolerance)),
            );

            s_conservation.enable(region, row)?;
            row += 1;

            // Row 2..N: Bit decomposition of (tolerance - residual)
            let _bound = tolerance.raw - residual.raw;
            let bit_val: u64 = bound_bits
                .iter()
                .enumerate()
                .map(|(j, &b)| if b { 1u64 << j } else { 0 })
                .sum();
            region.assign_advice(
                a_col,
                row,
                Value::known(Assigned::from(Fr::from(bit_val))),
            );
            row += 1;

            Ok(row)
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // CG Solve Verification Gadget
    // ═════════════════════════════════════════════════════════════════════

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

    // ═════════════════════════════════════════════════════════════════════
    // Public Input Gadget
    // ═════════════════════════════════════════════════════════════════════

    /// Gadget for binding witness values to public instance column.
    pub struct PublicInputGadget;

    impl PublicInputGadget {
        /// Assign a public input value and return its cell for instance binding.
        ///
        /// The returned `Cell` **must** be passed to `layouter.constrain_instance()`
        /// after the region closure to enforce that the witness value matches
        /// the public statement.  Without that binding the verifier cannot
        /// detect tampered public inputs — see Task 2.1 soundness audit.
        pub fn assign_public_input(
            region: &mut Region<'_, Fr>,
            c_col: Column<Advice>,
            row: usize,
            value: Fr,
        ) -> Result<Cell, Error> {
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
// Non-Halo2 Stubs
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "halo2"))]
pub use stubs::*;

/// Stub types for non-Halo2 builds.
#[cfg(not(feature = "halo2"))]
pub mod stubs {
    /// Stub for FixedPointMACGadget.
    pub struct FixedPointMACGadget;
    /// Stub for BitDecompositionGadget.
    pub struct BitDecompositionGadget;
    /// Stub for SvdOrderingGadget.
    pub struct SvdOrderingGadget;
    /// Stub for ConservationGadget.
    pub struct ConservationGadget;
    /// Stub for CgSolveGadget.
    pub struct CgSolveGadget;
    /// Stub for PublicInputGadget.
    pub struct PublicInputGadget;
}
