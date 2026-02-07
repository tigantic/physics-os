//! Q16.16 Fixed-Point Arithmetic
//!
//! Deterministic, ZK-friendly arithmetic for structural mechanics.
//! Raw integer `r` represents real value `r / 65536`.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use std::fmt;
use std::ops::{Add, Sub, Mul, Neg};

pub const SCALE: i64 = 65536;
pub const FRAC_BITS: u32 = 16;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Q16(pub i32);

impl Q16 {
    pub const ZERO: Q16 = Q16(0);
    pub const ONE: Q16 = Q16(SCALE as i32);
    pub const NEG_ONE: Q16 = Q16(-(SCALE as i32));
    pub const HALF: Q16 = Q16((SCALE / 2) as i32);
    pub const TWO: Q16 = Q16((2 * SCALE) as i32);
    pub const THREE: Q16 = Q16((3 * SCALE) as i32);
    pub const EPSILON: Q16 = Q16(1);

    #[inline] pub const fn from_int(n: i32) -> Self { Q16(n << FRAC_BITS) }
    #[inline] pub fn from_ratio(a: i32, b: i32) -> Self { Q16(((a as i64 * SCALE) / b as i64) as i32) }
    #[inline] pub fn from_f64(v: f64) -> Self { Q16((v * SCALE as f64).round() as i32) }
    #[inline] pub fn to_f64(self) -> f64 { self.0 as f64 / SCALE as f64 }
    #[inline] pub const fn raw(self) -> i32 { self.0 }
    #[inline] pub const fn from_raw(r: i32) -> Self { Q16(r) }
    #[inline] pub fn abs(self) -> Self { Q16(self.0.abs()) }

    #[inline]
    pub fn div(self, rhs: Q16) -> Q16 {
        Q16(((self.0 as i64 * SCALE) / rhs.0 as i64) as i32)
    }

    pub fn sqrt(self) -> Q16 {
        if self.0 <= 0 { return Q16::ZERO; }
        // Newton's method on N = val << 16 so result is in Q16.16 directly.
        let val = self.0 as u64;
        let n = val << FRAC_BITS;
        let bit_len = 64 - n.leading_zeros();
        let mut x: u64 = 1u64 << ((bit_len + 1) / 2);
        for _ in 0..16 {
            let next = (x + n / x) >> 1;
            if next >= x { break; }
            x = next;
        }
        Q16(x as i32)
    }

    #[inline] pub fn min(self, o: Q16) -> Q16 { if self.0 <= o.0 { self } else { o } }
    #[inline] pub fn max(self, o: Q16) -> Q16 { if self.0 >= o.0 { self } else { o } }
}

impl Add for Q16 { type Output = Q16; #[inline] fn add(self, r: Q16) -> Q16 { Q16(self.0.wrapping_add(r.0)) } }
impl Sub for Q16 { type Output = Q16; #[inline] fn sub(self, r: Q16) -> Q16 { Q16(self.0.wrapping_sub(r.0)) } }
impl Mul for Q16 { type Output = Q16; #[inline] fn mul(self, r: Q16) -> Q16 { Q16(((self.0 as i64 * r.0 as i64) >> FRAC_BITS) as i32) } }
impl Neg for Q16 { type Output = Q16; #[inline] fn neg(self) -> Q16 { Q16(-self.0) } }
impl fmt::Debug for Q16 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Q16({:.6})", self.to_f64()) } }
impl fmt::Display for Q16 { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:.6}", self.to_f64()) } }
