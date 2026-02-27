//! Q16.16 Fixed-Point Arithmetic
//!
//! Deterministic, ZK-friendly arithmetic for electromagnetic field computations.
//! Raw integer `r` represents real value `r / 65536`.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use std::fmt;
use std::ops::{Add, Sub, Mul, Neg};

pub const SCALE: i64 = 65536;
pub const FRAC_BITS: u32 = 16;

/// Q16.16 fixed-point number.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Q16(pub i32);

impl Q16 {
    pub const ZERO: Q16 = Q16(0);
    pub const ONE: Q16 = Q16(SCALE as i32);
    pub const NEG_ONE: Q16 = Q16(-(SCALE as i32));
    pub const HALF: Q16 = Q16((SCALE / 2) as i32);
    pub const TWO: Q16 = Q16((2 * SCALE) as i32);
    pub const EPSILON: Q16 = Q16(1);

    /// Construct from integer.
    #[inline]
    pub const fn from_int(n: i32) -> Self {
        Q16(n << FRAC_BITS)
    }

    /// Construct from rational a/b.
    #[inline]
    pub fn from_ratio(a: i32, b: i32) -> Self {
        Q16(((a as i64 * SCALE) / b as i64) as i32)
    }

    /// Construct from f64 (for initialization only — not used in solver loop).
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        Q16((v * SCALE as f64).round() as i32)
    }

    /// Convert to f64 (for diagnostics/display only).
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE as f64
    }

    /// Raw integer representation.
    #[inline]
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Construct from raw integer.
    #[inline]
    pub const fn from_raw(r: i32) -> Self {
        Q16(r)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Q16(self.0.abs())
    }

    /// Fixed-point division.
    #[inline]
    pub fn div(self, rhs: Q16) -> Q16 {
        Q16(((self.0 as i64 * SCALE) / rhs.0 as i64) as i32)
    }

    /// Square root via Newton's method (integer domain).
    pub fn sqrt(self) -> Q16 {
        if self.0 <= 0 { return Q16::ZERO; }
        let val = self.0 as u64;
        // We need isqrt(val << FRAC_BITS), result in Q16.16 raw units.
        let n = val << (FRAC_BITS as u64);
        // Initial guess via bit-length halving (keeps Newton within ~1 iteration
        // of quadratic convergence immediately).
        let bit_len = 64 - n.leading_zeros();
        let mut x: u64 = 1u64 << ((bit_len + 1) / 2);
        // Newton's method: converges quadratically from a close initial guess.
        loop {
            let next = (x + n / x) >> 1;
            if next >= x { break; }
            x = next;
        }
        Q16(x as i32)
    }

    /// Minimum.
    #[inline]
    pub fn min(self, other: Q16) -> Q16 {
        if self.0 <= other.0 { self } else { other }
    }

    /// Maximum.
    #[inline]
    pub fn max(self, other: Q16) -> Q16 {
        if self.0 >= other.0 { self } else { other }
    }
}

impl Add for Q16 {
    type Output = Q16;
    #[inline]
    fn add(self, rhs: Q16) -> Q16 {
        Q16(self.0.wrapping_add(rhs.0))
    }
}

impl Sub for Q16 {
    type Output = Q16;
    #[inline]
    fn sub(self, rhs: Q16) -> Q16 {
        Q16(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for Q16 {
    type Output = Q16;
    #[inline]
    fn mul(self, rhs: Q16) -> Q16 {
        Q16(((self.0 as i64 * rhs.0 as i64) >> FRAC_BITS) as i32)
    }
}

impl Neg for Q16 {
    type Output = Q16;
    #[inline]
    fn neg(self) -> Q16 {
        Q16(-self.0)
    }
}

impl fmt::Debug for Q16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q16({:.6})", self.to_f64())
    }
}

impl fmt::Display for Q16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Q16::from_f64(3.5);
        let b = Q16::from_f64(2.0);
        assert_eq!((a + b).to_f64(), 5.5);
        assert_eq!((a - b).to_f64(), 1.5);
        assert_eq!((a * b).to_f64(), 7.0);
    }

    #[test]
    fn test_division() {
        let a = Q16::from_f64(7.0);
        let b = Q16::from_f64(2.0);
        assert!((a.div(b).to_f64() - 3.5).abs() < 1e-4);
    }

    #[test]
    fn test_sqrt() {
        let a = Q16::from_f64(4.0);
        assert!((a.sqrt().to_f64() - 2.0).abs() < 1e-3);
        let b = Q16::from_f64(9.0);
        assert!((b.sqrt().to_f64() - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_determinism() {
        let a = Q16::from_f64(1.0 / 3.0);
        let b = Q16::from_f64(1.0 / 3.0);
        assert_eq!(a.raw(), b.raw());
        assert_eq!((a * Q16::from_int(3)).raw(), (b * Q16::from_int(3)).raw());
    }

    #[test]
    fn test_from_ratio() {
        let third = Q16::from_ratio(1, 3);
        assert!((third.to_f64() - 0.333333).abs() < 1e-4);
    }
}
