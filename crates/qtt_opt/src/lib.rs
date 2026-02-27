//! # opt-qtt — PDE-Constrained Optimization via Quantized Tensor Trains
//!
//! Topology optimization (SIMP) and inverse problems with adjoint
//! sensitivity analysis, all in Q16.16 fixed-point arithmetic.
//!
//! ## Architecture
//!
//! ```text
//! Design Variables ρ (densities or parameters)
//!        │
//!        ▼
//! Forward Solve: K(ρ)u = F  (Quad4 plane stress)
//!        │
//!        ▼
//! Objective: J(ρ,u)  (compliance, misfit, etc.)
//!        │
//!        ▼
//! Adjoint Sensitivity: dJ/dρ = -p·ρ^{p-1}·uᵀKe₀u  (self-adjoint)
//!        │                  or  dJ/dρ + λᵀ∂R/∂ρ     (general)
//!        ▼
//! Filter: mesh-independence (weighted average, radius rmin)
//!        │
//!        ▼
//! Update: OC bisection (topology) or projected gradient (inverse)
//!        │
//!        ▼
//! Convergence Check → loop
//! ```
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

pub mod q16;
pub mod forward;
pub mod adjoint;
pub mod filter;
pub mod topology;
pub mod inverse;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod prelude {
    pub use crate::q16::Q16;
    pub use crate::forward::*;
    pub use crate::adjoint::*;
    pub use crate::filter::*;
    pub use crate::topology::*;
    pub use crate::inverse::*;
}
