//! Halo2 sub-circuit gadgets for Euler 3D proof.
//!
//! All shared gadgets are re-exported from `crate::gadgets`.
//! This module exists for backward compatibility.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

// Re-export all shared gadgets.
#[cfg(feature = "halo2")]
pub use crate::gadgets::halo2_gadgets::*;

#[cfg(not(feature = "halo2"))]
pub use crate::gadgets::stubs::*;
