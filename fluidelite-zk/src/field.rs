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
}

// ============================================================================
// Halo2 field conversions (requires halo2 feature)
// ============================================================================

#[cfg(feature = "halo2")]
use halo2_axiom::halo2curves::ff::PrimeField;

#[cfg(feature = "halo2")]
use halo2_axiom::plonk::Assigned;

/// Convert fixed-point to Assigned field element for circuit
#[cfg(feature = "halo2")]
pub fn to_field<const FRAC_BITS: u32>(fp: FixedPoint<FRAC_BITS>) -> Assigned<halo2_axiom::halo2curves::bn256::Fr> {
    use halo2_axiom::halo2curves::bn256::Fr;
    if fp.raw >= 0 {
        Assigned::from(Fr::from(fp.raw as u64))
    } else {
        Assigned::from(-Fr::from((-fp.raw) as u64))
    }
}

/// Convert field element back to fixed-point (for verification)
#[cfg(feature = "halo2")]
pub fn from_field<F: PrimeField, const FRAC_BITS: u32>(_f: F) -> FixedPoint<FRAC_BITS> {
    // This is a simplified version - real implementation needs
    // to handle the field's modular arithmetic properly
    todo!("Implement field -> fixed-point conversion")
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
}
