//! # cem-qtt — Computational Electromagnetics via Quantized Tensor Trains
//!
//! FDTD Maxwell solver in Q16.16 fixed-point arithmetic with MPS/MPO
//! tensor network compression. Part of the HyperTensor physics engine.
//!
//! ## Architecture
//!
//! ```text
//! Maxwell's Equations (∂B/∂t = −∇×E, ∂E/∂t = (1/ε)(∇×H − J − σE))
//!          │
//!          ▼
//!    Yee Lattice (staggered E/H grid)
//!          │
//!          ▼
//!    FDTD Leapfrog (explicit timestepping)
//!          │
//!          ▼
//!    Q16.16 Fixed-Point (deterministic, ZK-friendly)
//!          │
//!          ▼
//!    QTT Compression (MPS/MPO, SVD truncation, χ_max bounded)
//!          │
//!          ▼
//!    Conservation Verification (Poynting theorem)
//! ```
//!
//! ## Modules
//!
//! - [`q16`]: Q16.16 fixed-point arithmetic
//! - [`mps`]: Matrix Product State field representation
//! - [`mpo`]: Matrix Product Operator for differential operators
//! - [`material`]: Electromagnetic material properties (ε, μ, σ)
//! - [`fdtd`]: Yee lattice FDTD solver
//! - [`pml`]: Perfectly Matched Layer absorbing boundaries
//! - [`conservation`]: Poynting theorem energy conservation verification
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

pub mod q16;
pub mod mps;
pub mod mpo;
pub mod material;
pub mod fdtd;
pub mod pml;
pub mod conservation;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude: commonly used types.
pub mod prelude {
    pub use crate::q16::Q16;
    pub use crate::mps::Mps;
    pub use crate::mpo::{Mpo, mpo_apply};
    pub use crate::material::{Material, MaterialMap, Constants};
    pub use crate::fdtd::{FdtdConfig, FdtdSolver, YeeFields, BoundaryCondition, Source, FieldSnapshot};
    pub use crate::pml::PmlParams;
    pub use crate::conservation::{ConservationVerifier, ConservationResult, ConservationSummary};
}
