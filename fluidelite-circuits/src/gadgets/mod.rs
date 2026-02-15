//! Shared sub-circuit gadgets for all proof modules (STARK-only).
//!
//! Canonical type definitions for gadget types used across
//! euler3d, ns_imex, and thermal proof modules.
//!
//! ## Shared Gadgets
//!
//! - **FixedPointMACGadget**: Q16.16 multiply-accumulate with range-checked remainder
//! - **BitDecompositionGadget**: Proves a value decomposes into boolean bits
//! - **SvdOrderingGadget**: Proves singular values are non-negative and descending
//! - **TruncationErrorGadget**: Proves SVD truncation error Σσᵢ² ≤ ε² is bounded
//! - **ConservationGadget**: Proves conservation residuals are within tolerance
//! - **PublicInputGadget**: Binds witness values to public instance column
//!
//! ## Module-Specific Gadgets (not in this file)
//!
//! - `thermal::gadgets::CgSolveGadget`
//! - `ns_imex::gadgets::DiffusionSolveGadget`
//! - `ns_imex::gadgets::ProjectionGadget`
//! - `ns_imex::gadgets::DivergenceCheckGadget`
//!
//! ## Algebraic Hash Gadgets
//!
//! - **poseidon_stark**: Poseidon permutation + sponge over Goldilocks (STARK AIR)
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#[cfg(feature = "stark")]
pub mod poseidon_stark;

// ═══════════════════════════════════════════════════════════════════════════
// Gadget Types
// ═══════════════════════════════════════════════════════════════════════════

/// Gadget type definitions for STARK builds.
///
/// The actual constraint enforcement for STARK lives in the Winterfell AIR
/// (e.g. `stark_impl.rs`, `qtt_stark.rs`). These types exist for structural
/// compatibility so that test code and witness code can reference gadget
/// names without depending on a specific proving backend.
pub mod stubs {
    /// Fixed-point multiply-accumulate gadget (Q16.16).
    pub struct FixedPointMACGadget;
    /// Bit decomposition gadget for range proofs.
    pub struct BitDecompositionGadget;
    /// SVD singular value ordering gadget.
    pub struct SvdOrderingGadget;
    /// SVD truncation error bound gadget (Task 6.13).
    pub struct TruncationErrorGadget;
    /// Conservation law gadget.
    pub struct ConservationGadget;
    /// Public input binding gadget.
    pub struct PublicInputGadget;
}

pub use stubs::*;
