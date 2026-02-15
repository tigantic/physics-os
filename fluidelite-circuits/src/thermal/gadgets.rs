//! Sub-circuit gadgets for the Thermal/Heat Equation proof.
//!
//! Shared gadgets (MAC, BitDecomposition, SVD ordering, Conservation,
//! PublicInput) are re-exported from `crate::gadgets`.
//!
//! Thermal-specific gadgets:
//! - **CgSolveGadget**: Constrains CG iteration convergence
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub use crate::gadgets::stubs::*;

// ═══════════════════════════════════════════════════════════════════════════
// Thermal-Specific Gadgets
// ═══════════════════════════════════════════════════════════════════════════

/// Stub for CgSolveGadget.
pub struct CgSolveGadget;
