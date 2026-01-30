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
//! │  ┌───────────────┐                                              │
//! │  │      gpu      │  ← NEW: WGPU compute shaders for TT eval    │
//! │  │ TTEvaluator   │                                              │
//! │  └───────────────┘                                              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # QTT Doctrine
//!
//! This crate follows strict QTT doctrine:
//! - **Native QTT**: Tensor train cores transmitted directly, never decompressed
//! - **No Dense**: Memory usage is O(L·χ²), NEVER O(d^L)
//! - **GPU-native**: TT evaluation on GPU without decompression

pub mod transforms;
pub mod gpu;

// Re-export commonly used types
pub use transforms::morton;
pub use gpu::{TTEvaluator, TTParams, TTPipeline, PipelineConfig, MemoryStats};
