//! HyperCore: Physics Engine Core
//!
//! This crate provides the core physics infrastructure shared between apps:
//! - QTT (Quantized Tensor Train) evaluation on GPU
//! - MPO (Matrix Product Operator) application
//! - CFD operators (advection, diffusion, projection)
//! - Coordinate transforms (Morton encoding, geodetic)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        hyper_core                               │
//! │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
//! │  │   qtt_eval    │  │    mpo_ops    │  │   transforms  │       │
//! │  │ GPU tensor    │  │ Laplacian,    │  │ Morton, geo   │       │
//! │  │ evaluation    │  │ advection     │  │ coordinates   │       │
//! │  └───────────────┘  └───────────────┘  └───────────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Future Expansion
//!
//! This crate will eventually contain:
//! - Rust-native QTT evaluation (currently in tci_core_rust)
//! - WGPU compute shaders for GPU physics
//! - Unified field representation

pub mod transforms;

// Re-export commonly used types
pub use transforms::morton;
