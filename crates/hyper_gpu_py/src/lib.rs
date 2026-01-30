//! HyperGPU Python Bindings
//!
//! Exposes the Rust CUDA TT evaluation pipeline to Python via PyO3.
//! This enables Python to leverage the 97M queries/sec CUDA kernel
//! for QTT point evaluation.
//!
//! # Usage from Python
//!
//! ```python
//! from hyper_gpu_py import CudaTTEvaluator
//!
//! # Initialize CUDA
//! evaluator = CudaTTEvaluator()
//!
//! # Set TT structure (cores flattened, bond dims)
//! evaluator.set_structure(cores_flat, bond_dims, physical_dim=2)
//!
//! # Evaluate at indices
//! values = evaluator.evaluate(indices)  # indices: (num_queries, num_sites)
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods, IntoPyArray};

// Conditionally import GPU types only when gpu feature is enabled
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use hyper_core::gpu::{GpuContext, CudaTTPipeline, CudaError};

/// Convert CudaError to PyErr
#[cfg(feature = "gpu")]
fn cuda_to_py(e: CudaError) -> PyErr {
    PyRuntimeError::new_err(format!("CUDA error: {}", e))
}

/// Check if CUDA is available without panicking
#[pyfunction]
fn cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        std::panic::catch_unwind(|| {
            hyper_core::gpu::GpuContext::new().is_ok()
        }).unwrap_or(false)
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Get CUDA device name
#[pyfunction]
#[pyo3(signature = (_device_id=0))]
fn cuda_device_name(_device_id: usize) -> PyResult<String> {
    #[cfg(feature = "gpu")]
    {
        let ctx = hyper_core::gpu::GpuContext::new_on_device(_device_id)
            .map_err(|e| PyRuntimeError::new_err(format!("CUDA error: {}", e)))?;
        Ok(ctx.device_name.clone())
    }
    #[cfg(not(feature = "gpu"))]
    {
        Err(PyRuntimeError::new_err("CUDA support not compiled"))
    }
}

/// Python-accessible CUDA TT Evaluator
#[pyclass]
pub struct CudaTTEvaluator {
    #[cfg(feature = "gpu")]
    ctx: Option<Arc<GpuContext>>,
    #[cfg(feature = "gpu")]
    pipeline: Option<CudaTTPipeline>,
    num_sites: usize,
}

#[pymethods]
impl CudaTTEvaluator {
    #[new]
    #[pyo3(signature = (_device_id=0))]
    fn new(_device_id: usize) -> PyResult<Self> {
        #[cfg(feature = "gpu")]
        {
            let ctx = Arc::new(
                GpuContext::new_on_device(_device_id)
                    .map_err(cuda_to_py)?
            );
            
            let pipeline = CudaTTPipeline::new(ctx.clone())
                .map_err(cuda_to_py)?;
            
            Ok(Self {
                ctx: Some(ctx),
                pipeline: Some(pipeline),
                num_sites: 0,
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled. Rebuild with --features gpu"))
        }
    }

    fn set_structure<'py>(
        &mut self,
        _cores_flat: &Bound<'py, PyArray1<f32>>,
        _bond_dims: &Bound<'py, PyArray1<u32>>,
        _physical_dim: u32,
    ) -> PyResult<()> {
        #[cfg(feature = "gpu")]
        {
            let pipeline = self.pipeline.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Pipeline not initialized"))?;
            
            let cores_data = unsafe { _cores_flat.as_slice()? };
            let bond_data = unsafe { _bond_dims.as_slice()? };
            
            pipeline.set_tt_structure(cores_data, bond_data, _physical_dim)
                .map_err(cuda_to_py)?;
            
            self.num_sites = bond_data.len() - 1;
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled"))
        }
    }

    fn evaluate<'py>(
        &mut self,
        _py: Python<'py>,
        _indices: &Bound<'py, PyArray1<u32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        #[cfg(feature = "gpu")]
        {
            let pipeline = self.pipeline.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Pipeline not initialized"))?;
            
            let indices_data = unsafe { _indices.as_slice()? };
            
            let result = pipeline.evaluate(indices_data)
                .map_err(cuda_to_py)?;
            
            Ok(result.into_pyarray(_py))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled"))
        }
    }

    fn evaluate_morton<'py>(
        &mut self,
        _py: Python<'py>,
        _morton_indices: &Bound<'py, PyArray1<i64>>,
        _n_qubits: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        #[cfg(feature = "gpu")]
        {
            let pipeline = self.pipeline.as_mut()
                .ok_or_else(|| PyRuntimeError::new_err("Pipeline not initialized"))?;
            
            let morton_data = unsafe { _morton_indices.as_slice()? };
            let num_queries = morton_data.len();
            
            let mut indices = Vec::with_capacity(num_queries * _n_qubits);
            
            for &z in morton_data {
                for k in 0.._n_qubits {
                    let bit_pos = _n_qubits - 1 - k;
                    let bit = ((z >> bit_pos) & 1) as u32;
                    indices.push(bit);
                }
            }
            
            let result = pipeline.evaluate(&indices)
                .map_err(cuda_to_py)?;
            
            Ok(result.into_pyarray(_py))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled"))
        }
    }

    fn cache_stats(&self) -> PyResult<(u64, u64, u64)> {
        #[cfg(feature = "gpu")]
        {
            let pipeline = self.pipeline.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Pipeline not initialized"))?;
            Ok(pipeline.cache_stats())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Ok((0, 0, 0))
        }
    }

    fn cache_hit_rate(&self) -> PyResult<f64> {
        #[cfg(feature = "gpu")]
        {
            let pipeline = self.pipeline.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Pipeline not initialized"))?;
            Ok(pipeline.cache_hit_rate())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Ok(0.0)
        }
    }

    fn transfer_stats(&self) -> PyResult<(u64, u64, u64)> {
        #[cfg(feature = "gpu")]
        {
            let pipeline = self.pipeline.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Pipeline not initialized"))?;
            Ok(pipeline.transfer_stats())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Ok((0, 0, 0))
        }
    }

    fn num_sites(&self) -> usize {
        self.num_sites
    }
}

/// Batch QTT evaluator
#[pyclass]
pub struct BatchQTTEvaluator {
    #[cfg(feature = "gpu")]
    evaluator: Option<CudaTTEvaluator>,
    n_qubits: usize,
}

#[pymethods]
impl BatchQTTEvaluator {
    #[new]
    #[pyo3(signature = (_device_id=0))]
    fn new(_device_id: usize) -> PyResult<Self> {
        #[cfg(feature = "gpu")]
        {
            Ok(Self {
                evaluator: Some(CudaTTEvaluator::new(_device_id)?),
                n_qubits: 0,
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled"))
        }
    }

    fn set_cores<'py>(
        &mut self,
        _py: Python<'py>,
        _cores: Vec<Bound<'py, PyArray2<f32>>>,
    ) -> PyResult<()> {
        #[cfg(feature = "gpu")]
        {
            if _cores.is_empty() {
                return Err(PyValueError::new_err("Empty cores list"));
            }
            
            self.n_qubits = _cores.len();
            
            let mut cores_flat: Vec<f32> = Vec::new();
            let mut bond_dims: Vec<u32> = Vec::new();
            
            let first_shape = _cores[0].shape();
            bond_dims.push(first_shape[0] as u32);
            
            for core in &_cores {
                let data = unsafe { core.as_slice()? };
                let shape = core.shape();
                
                cores_flat.extend_from_slice(data);
                bond_dims.push(shape[1] as u32);
            }
            
            let cores_np = cores_flat.into_pyarray(_py);
            let bonds_np = bond_dims.into_pyarray(_py);
            
            self.evaluator.as_mut().unwrap().set_structure(&cores_np, &bonds_np, 2)?;
            
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled"))
        }
    }

    fn evaluate<'py>(
        &mut self,
        _py: Python<'py>,
        _morton_indices: &Bound<'py, PyArray1<i64>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        #[cfg(feature = "gpu")]
        {
            self.evaluator.as_mut().unwrap().evaluate_morton(_py, _morton_indices, self.n_qubits)
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("CUDA support not compiled"))
        }
    }

    fn stats(&self) -> PyResult<String> {
        #[cfg(feature = "gpu")]
        {
            let (uploaded, downloaded, launches) = self.evaluator.as_ref().unwrap().transfer_stats()?;
            let hit_rate = self.evaluator.as_ref().unwrap().cache_hit_rate()?;
            
            Ok(format!(
                "Uploaded: {} MB, Downloaded: {} MB, Launches: {}, Cache hit: {:.1}%",
                uploaded / (1024 * 1024),
                downloaded / (1024 * 1024),
                launches,
                hit_rate
            ))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Ok("CUDA not available".to_string())
        }
    }
}

/// PyO3 module initialization - NO CUDA CALLS HERE
#[pymodule]
fn hyper_gpu_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Don't check CUDA at import time - defer to cuda_available()
    m.add("CUDA_AVAILABLE", false)?;  // Conservative default - call cuda_available() to check
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_name, m)?)?;
    m.add_class::<CudaTTEvaluator>()?;
    m.add_class::<BatchQTTEvaluator>()?;
    Ok(())
}
