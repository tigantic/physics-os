//! Fixed-point field arithmetic for ZK circuits
//!
//! ZK circuits operate over finite fields, not floating point.
//! We use Q16.16 fixed-point representation:
//! - 16 bits for integer part
//! - 16 bits for fractional part
//! - Range: [-32768.0, 32767.999984741]
//!
//! All FluidElite weights and activations must be quantized to this format.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg, Sub};
use std::cmp::Ordering;

/// Fixed-point number with configurable precision
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedPoint<const FRAC_BITS: u32> {
    /// Raw integer representation
    pub raw: i64,
}

/// Q16.16 fixed-point (default for FluidElite)
pub type Q16 = FixedPoint<16>;

impl<const FRAC_BITS: u32> FixedPoint<FRAC_BITS> {
    /// Scale factor for this precision
    pub const SCALE: i64 = 1 << FRAC_BITS;

    /// Create from floating point (for initialization only - not in circuit)
    pub fn from_f64(val: f64) -> Self {
        Self {
            raw: (val * Self::SCALE as f64).round() as i64,
        }
    }

    /// Convert to floating point (for debugging only)
    pub fn to_f64(self) -> f64 {
        self.raw as f64 / Self::SCALE as f64
    }

    /// Create from raw integer representation
    pub const fn from_raw(raw: i64) -> Self {
        Self { raw }
    }

    /// Zero constant
    pub const fn zero() -> Self {
        Self { raw: 0 }
    }

    /// One constant
    pub const fn one() -> Self {
        Self {
            raw: 1 << FRAC_BITS,
        }
    }

    /// Multiply two fixed-point numbers
    /// Result needs right-shift to maintain scale
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            raw: (self.raw * rhs.raw) >> FRAC_BITS,
        }
    }

    /// Multiply-accumulate: self + a * b
    /// More efficient than separate mul and add
    pub fn mac(self, a: Self, b: Self) -> Self {
        Self {
            raw: self.raw + ((a.raw * b.raw) >> FRAC_BITS),
        }
    }

    /// Absolute value
    pub fn abs(self) -> Self {
        Self {
            raw: self.raw.abs(),
        }
    }
}

impl<const FRAC_BITS: u32> Add for FixedPoint<FRAC_BITS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            raw: self.raw + rhs.raw,
        }
    }
}

impl<const FRAC_BITS: u32> Mul for FixedPoint<FRAC_BITS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            raw: (self.raw * rhs.raw) >> FRAC_BITS,
        }
    }
}

impl<const FRAC_BITS: u32> Sub for FixedPoint<FRAC_BITS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            raw: self.raw - rhs.raw,
        }
    }
}

impl<const FRAC_BITS: u32> Neg for FixedPoint<FRAC_BITS> {
    type Output = Self;

    fn neg(self) -> Self {
        Self { raw: -self.raw }
    }
}

impl<const FRAC_BITS: u32> Default for FixedPoint<FRAC_BITS> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const FRAC_BITS: u32> PartialOrd for FixedPoint<FRAC_BITS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.raw.cmp(&other.raw))
    }
}

impl<const FRAC_BITS: u32> Ord for FixedPoint<FRAC_BITS> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }
}

impl Q16 {
    /// Zero constant
    pub const ZERO: Self = Self::zero();

    /// One constant
    pub const ONE: Self = Self::one();

    /// Negative one constant
    pub const NEG_ONE: Self = Self { raw: -(1 << 16) };

    /// Smallest positive value (1 LSB = 2^-16 ≈ 1.52588e-5)
    pub const EPSILON: Self = Self { raw: 1 };

    /// Maximum representable positive value in Q16.16 (i32 range)
    /// raw = 2_147_483_647 → 32767.999984741
    pub const MAX_Q16: Self = Self { raw: i32::MAX as i64 };

    /// Minimum representable negative value in Q16.16 (i32 range)
    /// raw = -2_147_483_648 → -32768.0
    pub const MIN_Q16: Self = Self { raw: i32::MIN as i64 };

    /// Maximum representable positive value (full i64 range)
    pub const MAX_I64: Self = Self { raw: i64::MAX };

    /// Minimum representable negative value (full i64 range).
    /// Note: `to_field(MIN_I64)` requires special handling since `-i64::MIN` overflows.
    pub const MIN_I64: Self = Self { raw: i64::MIN };
}

// ============================================================================
// Halo2 field conversions (requires halo2 feature)
// ============================================================================

#[cfg(feature = "halo2")]
use halo2_axiom::halo2curves::ff::PrimeField;

#[cfg(feature = "halo2")]
use halo2_axiom::plonk::Assigned;

/// Convert fixed-point to Assigned field element for circuit.
///
/// Positive values map to `Fr::from(raw)`.
/// Negative values map to `Fr::from(p - |raw|)` via field negation.
/// Handles `i64::MIN` correctly by computing magnitude as `u64` to avoid
/// overflow from negating `i64::MIN` in two's complement.
#[cfg(feature = "halo2")]
pub fn to_field<const FRAC_BITS: u32>(fp: FixedPoint<FRAC_BITS>) -> Assigned<halo2_axiom::halo2curves::bn256::Fr> {
    use halo2_axiom::halo2curves::bn256::Fr;
    if fp.raw >= 0 {
        Assigned::from(Fr::from(fp.raw as u64))
    } else {
        // Compute magnitude as u64 to avoid overflow on i64::MIN.
        // For i64::MIN (-9223372036854775808), `-fp.raw` would overflow i64,
        // but the two's complement bit pattern as u64 is exactly the magnitude.
        let magnitude = (fp.raw as u64).wrapping_neg();
        Assigned::from(-Fr::from(magnitude))
    }
}

/// Convert field element back to fixed-point (for verification)
///
/// BN254 Fr encodes negative values as `p - |x|`. This function correctly
/// decodes both positive and negative encodings by using field negation:
/// if the element has upper bytes set (i.e. it is large, meaning it is
/// `p - |x|`), we negate in the field to recover `|x|` and return `-|x|`.
#[cfg(feature = "halo2")]
pub fn from_field<const FRAC_BITS: u32>(f: halo2_axiom::halo2curves::bn256::Fr) -> FixedPoint<FRAC_BITS> {
    use halo2_axiom::halo2curves::bn256::Fr;

    // Zero is a special case — avoids negation producing p
    if f == Fr::zero() {
        return FixedPoint { raw: 0 };
    }

    // Little-endian representation of the field element
    let bytes = f.to_repr();

    // Extract the low 64 bits
    let raw_u64 = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]);

    // A "small" positive value has all upper bytes zero and fits in i64
    let is_positive = bytes[8..32].iter().all(|&b| b == 0)
        && raw_u64 <= i64::MAX as u64;

    let raw = if is_positive {
        raw_u64 as i64
    } else {
        // Negative value: f = p - |x| in the field.
        // Field negation gives -f = |x|, a small positive integer.
        let neg_f = -f;
        let neg_bytes = neg_f.to_repr();
        let magnitude = u64::from_le_bytes([
            neg_bytes[0], neg_bytes[1], neg_bytes[2], neg_bytes[3],
            neg_bytes[4], neg_bytes[5], neg_bytes[6], neg_bytes[7],
        ]);
        debug_assert!(
            neg_bytes[8..32].iter().all(|&b| b == 0),
            "from_field: negated value exceeds i64 range — not a valid FixedPoint<{}> encoding",
            FRAC_BITS,
        );
        // Compute the negative i64 value from the u64 magnitude.
        // Using wrapping_neg handles all cases including magnitude = 2^63 (i64::MIN).
        magnitude.wrapping_neg() as i64
    };
    FixedPoint { raw }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_basics() {
        let a = Q16::from_f64(1.5);
        let b = Q16::from_f64(2.0);

        // Addition
        let sum = a + b;
        assert!((sum.to_f64() - 3.5).abs() < 1e-4);

        // Multiplication
        let prod = a.mul(b);
        assert!((prod.to_f64() - 3.0).abs() < 1e-4);

        // MAC
        let mac = Q16::zero().mac(a, b);
        assert!((mac.to_f64() - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_fixed_point_range() {
        // Test near limits
        let large = Q16::from_f64(1000.0);
        let small = Q16::from_f64(0.001);

        let prod = large.mul(small);
        assert!((prod.to_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_negative_values() {
        let pos = Q16::from_f64(2.5);
        let neg = Q16::from_f64(-1.5);

        let sum = pos + neg;
        assert!((sum.to_f64() - 1.0).abs() < 1e-4);

        let prod = pos.mul(neg);
        assert!((prod.to_f64() - (-3.75)).abs() < 1e-4);
    }

    #[test]
    fn test_serialization() {
        let val = Q16::from_f64(3.14159);
        let json = serde_json::to_string(&val).unwrap();
        let recovered: Q16 = serde_json::from_str(&json).unwrap();
        assert_eq!(val, recovered);
    }

    #[cfg(feature = "halo2")]
    mod halo2_field_tests {
        use super::super::*;
        use halo2_axiom::halo2curves::bn256::Fr;

        /// Helper: roundtrip a Q16 value through to_field → Assigned → Fr → from_field
        fn roundtrip(val: f64) -> Q16 {
            let original = Q16::from_f64(val);
            let assigned = to_field::<16>(original);
            // Evaluate the Assigned to get the Fr value
            let fr: Fr = match assigned {
                halo2_axiom::plonk::Assigned::Zero => Fr::zero(),
                halo2_axiom::plonk::Assigned::Trivial(v) => v,
                halo2_axiom::plonk::Assigned::Rational(n, d) => n * d.invert().unwrap(),
            };
            from_field::<16>(fr)
        }

        #[test]
        fn test_roundtrip_positive() {
            let cases = [0.0, 1.0, 3.14159, 100.5, 1000.25, 32767.0];
            for val in cases {
                let recovered = roundtrip(val);
                let original = Q16::from_f64(val);
                assert_eq!(
                    recovered, original,
                    "roundtrip failed for {val}: got {:?}, expected {:?}",
                    recovered, original
                );
            }
        }

        #[test]
        fn test_roundtrip_negative() {
            let cases = [-1.0, -3.14159, -100.5, -1000.25, -32768.0];
            for val in cases {
                let recovered = roundtrip(val);
                let original = Q16::from_f64(val);
                assert_eq!(
                    recovered, original,
                    "roundtrip failed for {val}: got {:?}, expected {:?}",
                    recovered, original
                );
            }
        }

        #[test]
        fn test_roundtrip_zero() {
            let recovered = roundtrip(0.0);
            assert_eq!(recovered, Q16::zero());
        }

        #[test]
        fn test_roundtrip_small_negative() {
            // Smallest representable negative value (1 LSB below zero)
            let fp = Q16::from_raw(-1);
            let assigned = to_field::<16>(fp);
            let fr: Fr = match assigned {
                halo2_axiom::plonk::Assigned::Zero => Fr::zero(),
                halo2_axiom::plonk::Assigned::Trivial(v) => v,
                halo2_axiom::plonk::Assigned::Rational(n, d) => n * d.invert().unwrap(),
            };
            let recovered = from_field::<16>(fr);
            assert_eq!(recovered, fp, "roundtrip failed for raw=-1");
        }

        // =====================================================================
        // Task 2.3: Q16.16 boundary value tests
        // =====================================================================

        #[test]
        fn test_boundary_max_q16() {
            // Maximum positive value in Q16.16 range (≈32767.999984741)
            let fp = Q16::MAX_Q16;
            assert_eq!(fp.raw, i32::MAX as i64);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "MAX_Q16 roundtrip failed");
        }

        #[test]
        fn test_boundary_min_q16() {
            // Minimum negative value in Q16.16 range (-32768.0)
            let fp = Q16::MIN_Q16;
            assert_eq!(fp.raw, i32::MIN as i64);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "MIN_Q16 roundtrip failed");
        }

        #[test]
        fn test_boundary_epsilon() {
            // Smallest positive value (1 LSB ≈ 1.526e-5)
            let fp = Q16::EPSILON;
            assert_eq!(fp.raw, 1);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "EPSILON roundtrip failed");
        }

        #[test]
        fn test_boundary_neg_epsilon() {
            // Smallest (most negative) single step below zero
            let fp = Q16::from_raw(-1);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "neg epsilon roundtrip failed");
        }

        #[test]
        fn test_boundary_i64_max() {
            // Full i64::MAX — beyond Q16.16 range but representable
            let fp = Q16::MAX_I64;
            assert_eq!(fp.raw, i64::MAX);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "i64::MAX roundtrip failed");
        }

        #[test]
        fn test_boundary_i64_min() {
            // i64::MIN — the tricky case where -raw overflows i64.
            // This test validates the wrapping_neg() fix in to_field().
            let fp = Q16::MIN_I64;
            assert_eq!(fp.raw, i64::MIN);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "i64::MIN roundtrip failed");
        }

        #[test]
        fn test_boundary_i64_min_plus_one() {
            // i64::MIN + 1 — just above the overflow boundary
            let fp = Q16::from_raw(i64::MIN + 1);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "i64::MIN+1 roundtrip failed");
        }

        #[test]
        fn test_boundary_i64_max_minus_one() {
            // i64::MAX - 1
            let fp = Q16::from_raw(i64::MAX - 1);
            let recovered = roundtrip_raw(fp);
            assert_eq!(recovered, fp, "i64::MAX-1 roundtrip failed");
        }

        #[test]
        fn test_boundary_one_and_neg_one() {
            let one = Q16::ONE;
            assert_eq!(one.raw, 65536);
            assert_eq!(roundtrip_raw(one), one);

            let neg_one = Q16::NEG_ONE;
            assert_eq!(neg_one.raw, -65536);
            assert_eq!(roundtrip_raw(neg_one), neg_one);
        }

        #[test]
        fn test_boundary_u64_threshold() {
            // The exact boundary where u64 casting changes behavior:
            // i64::MAX as u64 = 9223372036854775807
            // (i64::MAX as u64) + 1 = 9223372036854775808 (this is i64::MIN as u64)
            let just_below = Q16::from_raw(i64::MAX);
            assert_eq!(roundtrip_raw(just_below), just_below);

            let just_above = Q16::from_raw(i64::MIN);
            assert_eq!(roundtrip_raw(just_above), just_above);
        }

        #[test]
        fn test_to_field_from_field_consistency() {
            // Verify to_field and from_field agree on sign and magnitude
            // for a range of positive and negative values
            let test_raws: Vec<i64> = vec![
                0, 1, -1, 65536, -65536,
                i32::MAX as i64, i32::MIN as i64,
                i64::MAX, i64::MIN,
                i64::MAX / 2, i64::MIN / 2,
                100, -100, 1_000_000, -1_000_000,
                2_147_483_647, -2_147_483_648,
            ];
            for raw in test_raws {
                let fp = Q16::from_raw(raw);
                let recovered = roundtrip_raw(fp);
                assert_eq!(
                    recovered, fp,
                    "to_field/from_field mismatch for raw={raw}"
                );
            }
        }

        /// Helper: roundtrip a raw Q16 value through field
        fn roundtrip_raw(fp: Q16) -> Q16 {
            let assigned = to_field::<16>(fp);
            let fr: Fr = match assigned {
                halo2_axiom::plonk::Assigned::Zero => Fr::zero(),
                halo2_axiom::plonk::Assigned::Trivial(v) => v,
                halo2_axiom::plonk::Assigned::Rational(n, d) => n * d.invert().unwrap(),
            };
            from_field::<16>(fr)
        }
    }

    // =========================================================================
    // Task 2.4: Proptest property-based roundtrip testing
    // =========================================================================

    #[cfg(feature = "halo2")]
    mod proptest_field {
        use super::super::*;
        use halo2_axiom::halo2curves::bn256::Fr;
        use halo2_axiom::halo2curves::ff::Field;
        use proptest::prelude::*;

        fn roundtrip_raw(fp: Q16) -> Q16 {
            let assigned = to_field::<16>(fp);
            let fr: Fr = match assigned {
                halo2_axiom::plonk::Assigned::Zero => Fr::zero(),
                halo2_axiom::plonk::Assigned::Trivial(v) => v,
                halo2_axiom::plonk::Assigned::Rational(n, d) => n * d.invert().unwrap(),
            };
            from_field::<16>(fr)
        }

        proptest! {
            /// Any Q16 value survives to_field → from_field without corruption.
            #[test]
            fn prop_roundtrip_any_i64(raw in proptest::num::i64::ANY) {
                let fp = Q16::from_raw(raw);
                let recovered = roundtrip_raw(fp);
                prop_assert_eq!(recovered, fp,
                    "roundtrip failed for raw={}", raw);
            }

            /// Q16 values in the documented ±32768 range survive roundtrip.
            #[test]
            fn prop_roundtrip_q16_range(raw in (i32::MIN as i64)..=(i32::MAX as i64)) {
                let fp = Q16::from_raw(raw);
                let recovered = roundtrip_raw(fp);
                prop_assert_eq!(recovered, fp,
                    "roundtrip failed for Q16-range raw={}", raw);
            }

            /// Negation in field corresponds to negation of Q16 value.
            #[test]
            fn prop_negation_consistent(raw in 1i64..=i64::MAX) {
                let pos = Q16::from_raw(raw);
                let neg = Q16::from_raw(-raw);

                let pos_assigned = to_field::<16>(pos);
                let neg_assigned = to_field::<16>(neg);

                let pos_fr: Fr = match pos_assigned {
                    halo2_axiom::plonk::Assigned::Zero => Fr::zero(),
                    halo2_axiom::plonk::Assigned::Trivial(v) => v,
                    halo2_axiom::plonk::Assigned::Rational(n, d) => n * d.invert().unwrap(),
                };
                let neg_fr: Fr = match neg_assigned {
                    halo2_axiom::plonk::Assigned::Zero => Fr::zero(),
                    halo2_axiom::plonk::Assigned::Trivial(v) => v,
                    halo2_axiom::plonk::Assigned::Rational(n, d) => n * d.invert().unwrap(),
                };

                prop_assert_eq!(pos_fr + neg_fr, Fr::zero(),
                    "to_field(x) + to_field(-x) != 0 for raw={}", raw);
            }
        }
    }
}
