//! # fea-qtt — Structural Mechanics FEA via Quantized Tensor Trains
//!
//! Static linear elasticity solver in Q16.16 fixed-point arithmetic
//! with Hex8 elements, CG iteration, and energy conservation verification.
//!
//! ## Architecture
//!
//! ```text
//! Elasticity: ∇·σ + f = 0,  σ = D·ε,  ε = ½(∇u + ∇uᵀ)
//!          │
//!          ▼
//!    Hex8 Isoparametric Elements (trilinear shape functions)
//!          │
//!          ▼
//!    2×2×2 Gauss Quadrature (B-matrix, element stiffness Ke)
//!          │
//!          ▼
//!    Global Assembly (Ku = F, sparse COO)
//!          │
//!          ▼
//!    Conjugate Gradient Solver (Q16.16 fixed-point)
//!          │
//!          ▼
//!    Stress Recovery (σ = D·B·u at centroids)
//!          │
//!          ▼
//!    Energy Conservation (½uᵀKu = ½Fᵀu)
//! ```
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

pub mod q16;
pub mod material;
pub mod element;
pub mod mesh;
pub mod solver;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod prelude {
    pub use crate::q16::Q16;
    pub use crate::material::{IsotropicMaterial, MaterialMap};
    pub use crate::element::*;
    pub use crate::mesh::{HexMesh, Node, Element};
    pub use crate::solver::*;
}
