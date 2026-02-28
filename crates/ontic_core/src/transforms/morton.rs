//! Morton (Z-order) encoding for space-filling curves
//!
//! Morton encoding interleaves the bits of multi-dimensional coordinates
//! to create a 1D index that preserves spatial locality. This is critical
//! for QTT tensor indexing.
//!
//! # Example
//!
//! ```
//! use ontic_core::transforms::morton::{encode_2d, decode_2d};
//!
//! let (x, y) = (5, 3);
//! let morton = encode_2d(x, y);
//! let (x2, y2) = decode_2d(morton);
//! assert_eq!((x, y), (x2, y2));
//! ```

/// Encode 2D coordinates to Morton code
///
/// Interleaves bits: x₀y₀x₁y₁x₂y₂...
#[inline]
pub fn encode_2d(x: u32, y: u32) -> u64 {
    let x = part1by1_32(x) as u64;
    let y = part1by1_32(y) as u64;
    x | (y << 1)
}

/// Decode Morton code to 2D coordinates
#[inline]
pub fn decode_2d(code: u64) -> (u32, u32) {
    let x = compact1by1_64(code) as u32;
    let y = compact1by1_64(code >> 1) as u32;
    (x, y)
}

/// Encode 3D coordinates to Morton code
///
/// Interleaves bits: x₀y₀z₀x₁y₁z₁x₂y₂z₂...
#[inline]
pub fn encode_3d(x: u32, y: u32, z: u32) -> u64 {
    let x = part1by2_32(x) as u64;
    let y = part1by2_32(y) as u64;
    let z = part1by2_32(z) as u64;
    x | (y << 1) | (z << 2)
}

/// Decode Morton code to 3D coordinates
#[inline]
pub fn decode_3d(code: u64) -> (u32, u32, u32) {
    let x = compact1by2_64(code) as u32;
    let y = compact1by2_64(code >> 1) as u32;
    let z = compact1by2_64(code >> 2) as u32;
    (x, y, z)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit manipulation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Insert a 0 bit between each bit (for 2D Morton)
/// 0000_0000_0000_0000_fedc_ba98_7654_3210 ->
/// 0f0e_0d0c_0b0a_0908_0706_0504_0302_0100
#[inline]
fn part1by1_32(mut n: u32) -> u32 {
    n &= 0x0000_ffff;
    n = (n ^ (n << 8)) & 0x00ff_00ff;
    n = (n ^ (n << 4)) & 0x0f0f_0f0f;
    n = (n ^ (n << 2)) & 0x3333_3333;
    n = (n ^ (n << 1)) & 0x5555_5555;
    n
}

/// Extract every other bit (inverse of part1by1)
#[inline]
fn compact1by1_64(mut n: u64) -> u64 {
    n &= 0x5555_5555_5555_5555;
    n = (n ^ (n >> 1)) & 0x3333_3333_3333_3333;
    n = (n ^ (n >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    n = (n ^ (n >> 4)) & 0x00ff_00ff_00ff_00ff;
    n = (n ^ (n >> 8)) & 0x0000_ffff_0000_ffff;
    n = (n ^ (n >> 16)) & 0x0000_0000_ffff_ffff;
    n
}

/// Insert two 0 bits between each bit (for 3D Morton)
/// Input:  00000000_00000000_00000000_00cba987_6543210
/// Output: 00c00b00_a00900_800700_600500_400300_200100_0
#[inline]
fn part1by2_32(mut n: u32) -> u32 {
    n &= 0x000003ff; // 10 bits max for 30-bit output (fits in u32)
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;
    n
}

/// Extract every third bit (inverse of part1by2)
#[inline]
fn compact1by2_64(mut n: u64) -> u64 {
    n &= 0x1249249249249249; // Mask for every 3rd bit
    n = (n ^ (n >> 2)) & 0x30c30c30c30c30c3;
    n = (n ^ (n >> 4)) & 0xf00f00f00f00f00f;
    n = (n ^ (n >> 8)) & 0x00ff0000ff0000ff;
    n = (n ^ (n >> 16)) & 0x00ff00000000ffff;
    n = (n ^ (n >> 32)) & 0x00000000001fffff;
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_decode_2d() {
        for x in 0..64 {
            for y in 0..64 {
                let code = encode_2d(x, y);
                let (x2, y2) = decode_2d(code);
                assert_eq!((x, y), (x2, y2), "Failed for ({}, {})", x, y);
            }
        }
    }
    
    #[test]
    fn test_encode_decode_3d() {
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let code = encode_3d(x, y, z);
                    let (x2, y2, z2) = decode_3d(code);
                    assert_eq!((x, y, z), (x2, y2, z2), "Failed for ({}, {}, {})", x, y, z);
                }
            }
        }
    }
    
    #[test]
    fn test_morton_locality() {
        // Adjacent Morton codes should be spatially close
        let code1 = encode_2d(10, 10);
        let code2 = encode_2d(11, 10);
        let code3 = encode_2d(10, 11);
        
        // All should be within a small range
        assert!((code1 as i64 - code2 as i64).abs() <= 4);
        assert!((code1 as i64 - code3 as i64).abs() <= 4);
    }
}
