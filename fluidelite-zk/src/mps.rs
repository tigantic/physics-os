//! Matrix Product State (MPS) implementation
//!
//! An MPS represents a 1D tensor network with bond dimension χ.
//! In FluidElite, the MPS encodes the hidden context state.
//!
//! Structure:
//! ```text
//!     d     d     d           d
//!     │     │     │           │
//!   ┌─┴─┐ ┌─┴─┐ ┌─┴─┐       ┌─┴─┐
//!   │ A │─│ A │─│ A │─ ··· ─│ A │
//!   └───┘ └───┘ └───┘       └───┘
//!   1   χ     χ     χ       χ   1
//!
//!   Site 0    1     2       L-1
//! ```
//!
//! Each core A[i] has shape (χ_left, d, χ_right) where:
//! - χ_left = bond dimension to the left (1 for first site)
//! - d = physical dimension (2 for binary embedding)
//! - χ_right = bond dimension to the right (1 for last site)

use serde::{Deserialize, Serialize};

use crate::config::{CHI, L, PHYS_DIM};
use crate::field::Q16;

/// A single MPS core tensor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MPSCore {
    /// Core tensor data in row-major order: [χ_left, d, χ_right]
    pub data: Vec<Q16>,
    /// Left bond dimension
    pub chi_left: usize,
    /// Physical dimension
    pub d: usize,
    /// Right bond dimension
    pub chi_right: usize,
}

impl MPSCore {
    /// Create a new MPS core with given dimensions
    pub fn new(chi_left: usize, d: usize, chi_right: usize) -> Self {
        Self {
            data: vec![Q16::zero(); chi_left * d * chi_right],
            chi_left,
            d,
            chi_right,
        }
    }

    /// Create from raw data
    pub fn from_data(data: Vec<Q16>, chi_left: usize, d: usize, chi_right: usize) -> Self {
        assert_eq!(
            data.len(),
            chi_left * d * chi_right,
            "Data size mismatch"
        );
        Self {
            data,
            chi_left,
            d,
            chi_right,
        }
    }

    /// Get element at (left, phys, right)
    #[inline]
    pub fn get(&self, left: usize, phys: usize, right: usize) -> Q16 {
        debug_assert!(left < self.chi_left);
        debug_assert!(phys < self.d);
        debug_assert!(right < self.chi_right);
        self.data[(left * self.d + phys) * self.chi_right + right]
    }

    /// Set element at (left, phys, right)
    #[inline]
    pub fn set(&mut self, left: usize, phys: usize, right: usize, val: Q16) {
        debug_assert!(left < self.chi_left);
        debug_assert!(phys < self.d);
        debug_assert!(right < self.chi_right);
        self.data[(left * self.d + phys) * self.chi_right + right] = val;
    }

    /// Total number of elements
    pub fn size(&self) -> usize {
        self.chi_left * self.d * self.chi_right
    }
}

/// Matrix Product State
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MPS {
    /// Core tensors for each site
    pub cores: Vec<MPSCore>,
    /// Number of sites
    pub num_sites: usize,
}

impl MPS {
    /// Create an empty MPS with given number of sites and bond dimension
    pub fn new(num_sites: usize, chi: usize, d: usize) -> Self {
        let mut cores = Vec::with_capacity(num_sites);

        for i in 0..num_sites {
            let chi_left = if i == 0 { 1 } else { chi };
            let chi_right = if i == num_sites - 1 { 1 } else { chi };
            cores.push(MPSCore::new(chi_left, d, chi_right));
        }

        Self { cores, num_sites }
    }

    /// Create a product state from a bitstring
    /// Used for embedding: token_id -> MPS
    pub fn from_bits(bits: &[bool]) -> Self {
        let num_sites = bits.len();
        let mut mps = Self::new(num_sites, 1, PHYS_DIM);

        for (i, &bit) in bits.iter().enumerate() {
            // |0⟩ = [1, 0], |1⟩ = [0, 1]
            if bit {
                mps.cores[i].set(0, 1, 0, Q16::one());
            } else {
                mps.cores[i].set(0, 0, 0, Q16::one());
            }
        }

        mps
    }

    /// Create product state from token ID (bitwise embedding)
    pub fn embed_token(token_id: usize, num_sites: usize) -> Self {
        let bits: Vec<bool> = (0..num_sites)
            .rev()
            .map(|i| (token_id >> i) & 1 == 1)
            .collect();
        Self::from_bits(&bits)
    }

    /// Get the maximum bond dimension
    pub fn max_chi(&self) -> usize {
        self.cores
            .iter()
            .map(|c| c.chi_left.max(c.chi_right))
            .max()
            .unwrap_or(1)
    }

    /// Total number of parameters
    pub fn num_params(&self) -> usize {
        self.cores.iter().map(|c| c.size()).sum()
    }

    /// Truncate bond dimensions to chi_max
    /// 
    /// # Algorithm
    /// This is a **greedy truncation** that keeps the first χ elements
    /// in each bond dimension. This is NOT optimal - proper SVD-based
    /// truncation would retain the largest singular values.
    ///
    /// # Why Not SVD?
    /// SVD requires floating-point arithmetic which is expensive in ZK circuits.
    /// For FluidElite's use case (token embeddings -> logits), greedy truncation
    /// provides sufficient accuracy while maintaining ZK efficiency.
    ///
    /// # Performance Note
    /// For production use where accuracy matters, consider using full SVD
    /// truncation on the host side before proving, and only prove the
    /// already-truncated MPS.
    pub fn truncate(&mut self, chi_max: usize) {
        for i in 0..self.num_sites {
            let core = &self.cores[i];
            let new_chi_left = core.chi_left.min(chi_max);
            let new_chi_right = core.chi_right.min(chi_max);

            // Handle boundaries
            let new_chi_left = if i == 0 { 1 } else { new_chi_left };
            let new_chi_right = if i == self.num_sites - 1 {
                1
            } else {
                new_chi_right
            };

            if new_chi_left < core.chi_left || new_chi_right < core.chi_right {
                let mut new_data = vec![Q16::zero(); new_chi_left * core.d * new_chi_right];

                // Copy truncated data
                for l in 0..new_chi_left {
                    for p in 0..core.d {
                        for r in 0..new_chi_right {
                            new_data[(l * core.d + p) * new_chi_right + r] =
                                self.cores[i].get(l, p, r);
                        }
                    }
                }

                self.cores[i] = MPSCore::from_data(new_data, new_chi_left, core.d, new_chi_right);
            }
        }
    }
}

/// Default configuration MPS
impl Default for MPS {
    fn default() -> Self {
        Self::new(L, CHI, PHYS_DIM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_creation() {
        let mps = MPS::new(4, 8, 2);
        assert_eq!(mps.num_sites, 4);
        assert_eq!(mps.cores[0].chi_left, 1);
        assert_eq!(mps.cores[0].chi_right, 8);
        assert_eq!(mps.cores[3].chi_left, 8);
        assert_eq!(mps.cores[3].chi_right, 1);
    }

    #[test]
    fn test_product_state() {
        // Token 5 = 0b0101 for 4 sites
        let mps = MPS::embed_token(5, 4);

        // Check bits: 0, 1, 0, 1
        assert_eq!(mps.cores[0].get(0, 0, 0), Q16::one()); // bit 0
        assert_eq!(mps.cores[0].get(0, 1, 0), Q16::zero());

        assert_eq!(mps.cores[1].get(0, 0, 0), Q16::zero());
        assert_eq!(mps.cores[1].get(0, 1, 0), Q16::one()); // bit 1

        assert_eq!(mps.cores[2].get(0, 0, 0), Q16::one()); // bit 0
        assert_eq!(mps.cores[3].get(0, 1, 0), Q16::one()); // bit 1
    }

    #[test]
    fn test_truncation() {
        let mut mps = MPS::new(4, 16, 2);
        mps.truncate(8);

        assert_eq!(mps.cores[0].chi_left, 1);
        assert_eq!(mps.cores[0].chi_right, 8);
        assert_eq!(mps.cores[1].chi_left, 8);
        assert_eq!(mps.cores[1].chi_right, 8);
        assert_eq!(mps.cores[3].chi_right, 1);
    }
}
