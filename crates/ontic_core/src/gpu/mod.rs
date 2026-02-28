//! GPU acceleration module for The Ontic Engine
//!
//! Provides CUDA-based acceleration for TT evaluation and other compute-intensive
//! operations. Uses cudarc for direct NVIDIA GPU access.

mod cuda_pipeline;
mod pipeline;
mod tt_eval;

pub use cuda_pipeline::*;
pub use pipeline::*;
pub use tt_eval::*;
