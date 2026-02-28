//! TCI Core - TT-Cross Interpolation algorithms for The Ontic Engine
//!
//! This crate provides high-performance Rust implementations of:
//! - MaxVol pivot selection algorithm
//! - Fiber-based TCI sampling
//! - Adaptive refinement for shocks
//! - Index manipulation (neighbor computation without GPU divergence)
//!
//! Architecture:
//! - Rust handles: pivot selection, index arithmetic, skeleton matrix operations
//! - Python/PyTorch handles: function evaluation on GPU, tensor storage

use pyo3::prelude::*;

mod types;
mod maxvol;
mod sampler;
mod skeleton;
mod indices;

pub use types::*;
pub use maxvol::*;
pub use sampler::*;
pub use skeleton::*;
pub use indices::*;

/// TCI Core Python module
#[pymodule]
fn _tci_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configuration
    m.add_class::<MaxVolConfig>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<TCIConfig>()?;
    
    // Main sampler
    m.add_class::<TCISampler>()?;
    
    // Index batch for DLPack
    m.add_class::<IndexBatch>()?;
    
    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
