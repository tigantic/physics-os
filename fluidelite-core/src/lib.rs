//! FluidElite Core: Tensor Primitives for ZK-Provable Physics
//!
//! This crate provides the foundational types shared across the FluidElite ecosystem:
//!
//! - **field**: Q16 fixed-point arithmetic (`FixedPoint<FRAC_BITS>`)
//! - **mps**: Matrix Product State representation
//! - **mpo**: Matrix Product Operator representation
//! - **ops**: MPS/MPO contraction, addition, truncation
//! - **weights**: Weight matrix handling
//! - **weight_crypto**: AES-256-GCM weight encryption
//! - **physics_traits**: `PhysicsProof`/`PhysicsProver`/`PhysicsVerifier` trait hierarchy
//!
//! # QTT Doctrine
//!
//! All tensor operations maintain the Quantized Tensor Train format:
//! - Memory: O(L·χ²), NEVER O(d^L)
//! - No dense materialization
//! - Cores transmitted directly
//!
//! © 2025-2026 Bradly Biron Baker Adams / Tigantic Labs. All Rights Reserved.
//! SPDX-License-Identifier: LicenseRef-Proprietary

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod field;
pub mod mps;
pub mod mpo;
pub mod ops;
pub mod qtt_operators;
pub mod weights;
pub mod weight_crypto;
pub mod physics_traits;

// Re-exports for convenience
pub use mpo::MPO;
pub use mps::MPS;

/// Configuration constants for FluidElite ZK
pub mod config {
    /// Number of sites in the tensor train (determines virtual context fidelity)
    /// 2^L = maximum distinguishable token positions
    pub const L: usize = 16;

    /// Bond dimension (memory capacity)
    /// Higher χ = more expressive but more constraints
    pub const CHI: usize = 64;

    /// MPO bond dimension
    pub const D: usize = 1;

    /// Physical dimension (binary embedding)
    pub const PHYS_DIM: usize = 2;

    /// Vocabulary size
    pub const VOCAB_SIZE: usize = 256;

    /// Fixed-point precision bits
    #[allow(dead_code)]
    pub const PRECISION_BITS: u32 = 32;

    /// Fixed-point scale factor (2^16 for Q16.16 format)
    #[allow(dead_code)]
    pub const SCALE: i64 = 1 << 16;
}
