//! Sub-circuit gadgets for Navier-Stokes IMEX proof.
//!
//! Shared gadgets (MAC, BitDecomposition, SVD ordering, Conservation,
//! PublicInput) are re-exported from `crate::gadgets`.
//!
//! NS-IMEX-specific gadgets:
//! - **DiffusionSolveGadget**: Verifies (I - ν·Δt·L)u = u*
//! - **ProjectionGadget**: Verifies one CG step in pressure Poisson solve
//! - **DivergenceCheckGadget**: Verifies ‖∇·u‖ < ε_div
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub use crate::gadgets::stubs::*;

// ═══════════════════════════════════════════════════════════════════════════
// NS-IMEX-Specific Gadgets
// ═══════════════════════════════════════════════════════════════════════════

/// Stub for DiffusionSolveGadget.
pub struct DiffusionSolveGadget;
/// Stub for ProjectionGadget.
pub struct ProjectionGadget;
/// Stub for DivergenceCheckGadget.
pub struct DivergenceCheckGadget;
