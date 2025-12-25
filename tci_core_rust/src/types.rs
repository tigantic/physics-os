//! Common types and error definitions for TCI Core

use pyo3::prelude::*;
use thiserror::Error;

/// Errors that can occur during TCI operations
#[derive(Error, Debug)]
pub enum TCIError {
    #[error("MaxVol failed: not enough rows (have {m}, need {r})")]
    NotEnoughRows { m: usize, r: usize },
    
    #[error("MaxVol failed: singular matrix encountered")]
    SingularMatrix,
    
    #[error("MaxVol failed: did not converge in {iterations} iterations")]
    MaxVolNotConverged { iterations: usize },
    
    #[error("TCI failed: did not converge in {iterations} iterations")]
    TCINotConverged { iterations: usize },
    
    #[error("TCI failed: rank explosion detected (rank {rank} > hard cap {cap})")]
    RankExplosion { rank: usize, cap: usize },
    
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },
    
    #[error("Linear algebra error: {message}")]
    LinAlgError { message: String },
}

impl From<TCIError> for PyErr {
    fn from(err: TCIError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

/// Configuration for MaxVol pivot selection
#[pyclass]
#[derive(Clone, Debug)]
pub struct MaxVolConfig {
    /// Convergence tolerance (typically 0.01-0.1)
    #[pyo3(get, set)]
    pub tolerance: f64,
    
    /// Maximum iterations before giving up
    #[pyo3(get, set)]
    pub max_iterations: usize,
    
    /// Regularization for pseudo-inverse (for numerical stability)
    #[pyo3(get, set)]
    pub regularization: f64,
    
    /// Number of random restarts if stagnating
    #[pyo3(get, set)]
    pub random_restarts: usize,
}

#[pymethods]
impl MaxVolConfig {
    #[new]
    #[pyo3(signature = (tolerance=0.05, max_iterations=15, regularization=1e-12, random_restarts=2))]
    pub fn new(
        tolerance: f64,
        max_iterations: usize,
        regularization: f64,
        random_restarts: usize,
    ) -> Self {
        Self {
            tolerance,
            max_iterations,
            regularization,
            random_restarts,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "MaxVolConfig(tolerance={}, max_iterations={}, regularization={:.0e}, random_restarts={})",
            self.tolerance, self.max_iterations, self.regularization, self.random_restarts
        )
    }
}

impl Default for MaxVolConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.05,
            max_iterations: 15,
            regularization: 1e-12,
            random_restarts: 2,
        }
    }
}

/// Configuration for rank truncation
#[pyclass]
#[derive(Clone, Debug)]
pub struct TruncationPolicy {
    /// Preferred rank (SVD will try to achieve this)
    #[pyo3(get, set)]
    pub target_rank: usize,
    
    /// Absolute maximum rank (hard cap, non-negotiable)
    #[pyo3(get, set)]
    pub hard_cap: usize,
    
    /// Relative tolerance for SVD truncation
    #[pyo3(get, set)]
    pub relative_tol: f64,
    
    /// Whether to warn when rank is trending upward
    #[pyo3(get, set)]
    pub monitor_growth: bool,
}

#[pymethods]
impl TruncationPolicy {
    #[new]
    #[pyo3(signature = (target_rank=32, hard_cap=128, relative_tol=1e-8, monitor_growth=true))]
    pub fn new(
        target_rank: usize,
        hard_cap: usize,
        relative_tol: f64,
        monitor_growth: bool,
    ) -> Self {
        Self {
            target_rank,
            hard_cap,
            relative_tol,
            monitor_growth,
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "TruncationPolicy(target_rank={}, hard_cap={}, relative_tol={:.0e})",
            self.target_rank, self.hard_cap, self.relative_tol
        )
    }
}

impl Default for TruncationPolicy {
    fn default() -> Self {
        Self {
            target_rank: 32,
            hard_cap: 128,
            relative_tol: 1e-8,
            monitor_growth: true,
        }
    }
}

/// Main TCI configuration
#[pyclass]
#[derive(Clone, Debug)]
pub struct TCIConfig {
    /// Number of qubits (log2 of grid size)
    #[pyo3(get, set)]
    pub n_qubits: usize,
    
    /// Maximum rank for approximation
    #[pyo3(get, set)]
    pub max_rank: usize,
    
    /// Convergence tolerance
    #[pyo3(get, set)]
    pub tolerance: f64,
    
    /// Batch size for index generation
    #[pyo3(get, set)]
    pub batch_size: usize,
    
    /// MaxVol configuration
    #[pyo3(get, set)]
    pub maxvol_config: MaxVolConfig,
    
    /// Truncation policy
    #[pyo3(get, set)]
    pub truncation_policy: TruncationPolicy,
}

#[pymethods]
impl TCIConfig {
    #[new]
    #[pyo3(signature = (n_qubits, max_rank=64, tolerance=1e-8, batch_size=10000, maxvol_config=None, truncation_policy=None))]
    pub fn new(
        n_qubits: usize,
        max_rank: usize,
        tolerance: f64,
        batch_size: usize,
        maxvol_config: Option<MaxVolConfig>,
        truncation_policy: Option<TruncationPolicy>,
    ) -> Self {
        Self {
            n_qubits,
            max_rank,
            tolerance,
            batch_size,
            maxvol_config: maxvol_config.unwrap_or_default(),
            truncation_policy: truncation_policy.unwrap_or_default(),
        }
    }
    
    /// Total number of grid points (2^n_qubits)
    #[getter]
    pub fn n_points(&self) -> usize {
        1 << self.n_qubits
    }
    
    fn __repr__(&self) -> String {
        format!(
            "TCIConfig(n_qubits={}, max_rank={}, tolerance={:.0e}, batch_size={})",
            self.n_qubits, self.max_rank, self.tolerance, self.batch_size
        )
    }
}
