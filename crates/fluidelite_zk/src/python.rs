//! Python bindings for FluidElite ZK
//!
//! Exposes the Rust ZK prover to Python via PyO3.
//!
//! # Python Usage
//!
//! ```python
//! from fluidelite_zk import FluidEliteProver, MPS, MPO, Q16
//!
//! # Create prover
//! prover = FluidEliteProver.new_with_identity_weights(num_sites=8, chi_max=16)
//!
//! # Create context
//! context = MPS.new(num_sites=8, chi_max=16)
//!
//! # Generate proof
//! proof = prover.prove(context, token_id=42)
//! print(f"Proof size: {len(proof.bytes)} bytes")
//! print(f"Proof time: {proof.generation_time_ms} ms")
//!
//! # Verify proof
//! is_valid = prover.verify(proof)
//! print(f"Valid: {is_valid}")
//! ```

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;

#[cfg(feature = "python")]
use crate::circuit::config::CircuitConfig;
#[cfg(feature = "python")]
use crate::field::Q16 as RustQ16;
#[cfg(feature = "python")]
use crate::mpo::MPO as RustMPO;
#[cfg(feature = "python")]
use crate::mps::MPS as RustMPS;
#[cfg(feature = "python")]
use crate::prover::FluidEliteProver as RustProver;

/// Python-facing MPS wrapper
#[cfg(feature = "python")]
#[pyclass(name = "MPS")]
pub struct PyMPS {
    inner: RustMPS,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMPS {
    /// Create a new MPS with given parameters
    #[new]
    #[pyo3(signature = (num_sites, chi_max, phys_dim=2))]
    fn new(num_sites: usize, chi_max: usize, phys_dim: usize) -> Self {
        Self {
            inner: RustMPS::new(num_sites, chi_max, phys_dim),
        }
    }

    /// Number of sites
    #[getter]
    fn num_sites(&self) -> usize {
        self.inner.num_sites()
    }

    /// Maximum bond dimension
    #[getter]
    fn chi_max(&self) -> usize {
        self.inner.chi_max()
    }

    /// Physical dimension
    #[getter]
    fn phys_dim(&self) -> usize {
        self.inner.phys_dim()
    }

    /// Serialize to bytes
    fn to_bytes(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let bytes = bincode::serialize(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).into())
    }

    /// Deserialize from bytes
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let inner: RustMPS = bincode::deserialize(bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "MPS(sites={}, chi={}, d={})",
            self.inner.num_sites(),
            self.inner.chi_max(),
            self.inner.phys_dim()
        )
    }
}

/// Python-facing MPO wrapper
#[cfg(feature = "python")]
#[pyclass(name = "MPO")]
pub struct PyMPO {
    inner: RustMPO,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMPO {
    /// Create an identity MPO
    #[staticmethod]
    #[pyo3(signature = (num_sites, phys_dim=2))]
    fn identity(num_sites: usize, phys_dim: usize) -> Self {
        Self {
            inner: RustMPO::identity(num_sites, phys_dim),
        }
    }

    /// Number of sites
    #[getter]
    fn num_sites(&self) -> usize {
        self.inner.num_sites()
    }

    fn __repr__(&self) -> String {
        format!("MPO(sites={})", self.inner.num_sites())
    }
}

/// Python-facing proof wrapper
#[cfg(feature = "python")]
#[pyclass(name = "Proof")]
pub struct PyProof {
    /// Raw proof bytes
    #[pyo3(get)]
    pub bytes: Vec<u8>,

    /// Public inputs as hex strings
    #[pyo3(get)]
    pub public_inputs: Vec<String>,

    /// Generation time in milliseconds
    #[pyo3(get)]
    pub generation_time_ms: u64,

    /// Token ID that was proven
    #[pyo3(get)]
    pub token_id: u64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyProof {
    /// Get proof as hex string
    fn to_hex(&self) -> String {
        hex::encode(&self.bytes)
    }

    /// Get proof size in bytes
    fn size(&self) -> usize {
        self.bytes.len()
    }

    /// Serialize to JSON
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&ProofJson {
            token_id: self.token_id,
            proof_bytes: base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &self.bytes,
            ),
            public_inputs: self.public_inputs.clone(),
            generation_time_ms: self.generation_time_ms,
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Proof(token={}, size={} bytes, time={} ms)",
            self.token_id,
            self.bytes.len(),
            self.generation_time_ms
        )
    }
}

#[cfg(feature = "python")]
#[derive(serde::Serialize)]
struct ProofJson {
    token_id: u64,
    proof_bytes: String,
    public_inputs: Vec<String>,
    generation_time_ms: u64,
}

/// Python-facing prover wrapper
#[cfg(feature = "python")]
#[pyclass(name = "FluidEliteProver")]
pub struct PyFluidEliteProver {
    inner: RustProver,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFluidEliteProver {
    /// Create a new prover with identity weights (for testing)
    #[staticmethod]
    #[pyo3(signature = (num_sites=8, chi_max=16, vocab_size=256))]
    fn new_with_identity_weights(num_sites: usize, chi_max: usize, vocab_size: usize) -> Self {
        let config = CircuitConfig {
            num_sites,
            chi_max,
            vocab_size,
            phys_dim: 2,
            k: 10,
        };

        let w_hidden = RustMPO::identity(num_sites, 2);
        let w_input = RustMPO::identity(num_sites, 2);
        let readout_weights = vec![RustQ16::from_f64(0.1); chi_max * vocab_size];

        Self {
            inner: RustProver::new(w_hidden, w_input, readout_weights, config),
        }
    }

    /// Generate a ZK proof for an inference step
    fn prove(&self, context: &PyMPS, token_id: u64) -> PyResult<PyProof> {
        let result = self
            .inner
            .prove(&context.inner, token_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(PyProof {
            bytes: result.inner.proof_bytes.clone(),
            public_inputs: result
                .inner
                .public_inputs
                .iter()
                .map(|x| format!("{:?}", x))
                .collect(),
            generation_time_ms: result.inner.generation_time_ms,
            token_id,
        })
    }

    /// Get prover statistics
    fn stats(&self) -> PyResult<String> {
        Ok(format!(
            "Prover(sites={}, chi={})",
            self.inner.config().num_sites,
            self.inner.config().chi_max
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "FluidEliteProver(sites={}, chi={})",
            self.inner.config().num_sites,
            self.inner.config().chi_max
        )
    }
}

/// Circuit configuration for Python
#[cfg(feature = "python")]
#[pyclass(name = "CircuitConfig")]
#[derive(Clone)]
pub struct PyCircuitConfig {
    #[pyo3(get, set)]
    pub num_sites: usize,
    #[pyo3(get, set)]
    pub chi_max: usize,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub phys_dim: usize,
    #[pyo3(get, set)]
    pub k: u32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCircuitConfig {
    #[new]
    #[pyo3(signature = (num_sites=8, chi_max=16, vocab_size=256, phys_dim=2, k=10))]
    fn new(num_sites: usize, chi_max: usize, vocab_size: usize, phys_dim: usize, k: u32) -> Self {
        Self {
            num_sites,
            chi_max,
            vocab_size,
            phys_dim,
            k,
        }
    }

    fn estimate_constraints(&self) -> usize {
        self.num_sites * 64 + self.chi_max * self.vocab_size
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitConfig(sites={}, chi={}, vocab={}, k={})",
            self.num_sites, self.chi_max, self.vocab_size, self.k
        )
    }
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn fluidelite_zk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMPS>()?;
    m.add_class::<PyMPO>()?;
    m.add_class::<PyProof>()?;
    m.add_class::<PyFluidEliteProver>()?;
    m.add_class::<PyCircuitConfig>()?;

    // Module-level functions
    #[pyfn(m)]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[pyfn(m)]
    fn prove_inference(
        context_bytes: &[u8],
        token_id: u64,
        num_sites: usize,
        chi_max: usize,
    ) -> PyResult<PyProof> {
        let context: RustMPS = bincode::deserialize(context_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let prover = PyFluidEliteProver::new_with_identity_weights(num_sites, chi_max, 256);
        let py_context = PyMPS { inner: context };

        prover.prove(&py_context, token_id)
    }

    Ok(())
}
