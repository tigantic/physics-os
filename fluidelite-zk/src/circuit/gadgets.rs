//! Reusable circuit gadgets for tensor operations
//!
//! These gadgets encapsulate common patterns for efficient constraint generation.

#[cfg(feature = "halo2")]
use crate::field::Q16;

// ============================================================================
// Halo2 gadgets (requires halo2 feature)
// ============================================================================

#[cfg(feature = "halo2")]
mod halo2_gadgets {
    use super::Q16;
    use halo2_axiom::{
        circuit::{Region, Value},
        halo2curves::bn256::Fr,
        plonk::{Advice, Assigned, Column, Error, Selector},
    };

    /// Gadget for multiply-accumulate operation
    pub struct MACGadget;

    impl MACGadget {
        /// Assign a MAC row: a × b + prev = result
        /// Returns the next row index
        pub fn assign_mac_row(
            region: &mut Region<'_, Fr>,
            s_mac: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            row: usize,
            a_val: Q16,
            b_val: Q16,
            result_val: Q16,
        ) -> Result<usize, Error> {
            // Assign operands
            region.assign_advice(a_col, row, Value::known(q16_to_assigned(a_val)));
            region.assign_advice(b_col, row, Value::known(q16_to_assigned(b_val)));
            region.assign_advice(c_col, row, Value::known(q16_to_assigned(result_val)));

            // Enable MAC gate
            s_mac.enable(region, row)?;

            Ok(row + 1)
        }
    }

    fn q16_to_assigned(fp: Q16) -> Assigned<Fr> {
        let fr = if fp.raw >= 0 { 
            Fr::from(fp.raw as u64) 
        } else { 
            -Fr::from((-fp.raw) as u64) 
        };
        Assigned::from(fr)
    }

    /// Gadget for dot product of two vectors
    pub struct DotProductGadget;

    impl DotProductGadget {
        /// Compute and assign dot product
        /// Returns the next row index
        pub fn assign_dot_product(
            region: &mut Region<'_, Fr>,
            s_mac: Selector,
            a_col: Column<Advice>,
            b_col: Column<Advice>,
            c_col: Column<Advice>,
            start_row: usize,
            a_vals: &[Q16],
            b_vals: &[Q16],
        ) -> Result<usize, Error> {
            assert_eq!(a_vals.len(), b_vals.len(), "Vector length mismatch");

            let mut row = start_row;

            // Initialize accumulator
            region.assign_advice(c_col, row, Value::known(Assigned::Zero));
            row += 1;

            // Compute running sum and assign each row
            let mut acc = Q16::zero();
            for (a, b) in a_vals.iter().zip(b_vals.iter()) {
                let product = a.mul(*b);
                acc = acc + product;
                row = MACGadget::assign_mac_row(
                    region, s_mac, a_col, b_col, c_col, row, *a, *b, acc
                )?;
            }

            Ok(row)
        }
    }
}

#[cfg(feature = "halo2")]
pub use halo2_gadgets::*;

// ============================================================================
// Tests (work without Halo2)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_computation() {
        let a = Q16::from_f64(2.5);
        let b = Q16::from_f64(3.0);
        let acc = Q16::from_f64(1.0);

        // Expected: 2.5 * 3.0 + 1.0 = 8.5
        let result = acc + a.mul(b);
        assert!((result.to_f64() - 8.5).abs() < 0.001);
    }

    #[test]
    fn test_dot_product_computation() {
        let a = vec![Q16::from_f64(1.0), Q16::from_f64(2.0), Q16::from_f64(3.0)];
        let b = vec![Q16::from_f64(4.0), Q16::from_f64(5.0), Q16::from_f64(6.0)];

        // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let mut acc = Q16::zero();
        for (x, y) in a.iter().zip(b.iter()) {
            acc = acc + x.mul(*y);
        }
        assert!((acc.to_f64() - 32.0).abs() < 0.001);
    }
}
