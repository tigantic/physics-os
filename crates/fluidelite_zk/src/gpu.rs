//! GPU-accelerated ZK proving using ICICLE
//!
//! This module provides GPU-accelerated cryptographic operations for Halo2
//! proof generation using Ingonyama's ICICLE library.
//!
//! # Features
//!
//! - GPU-accelerated Multi-Scalar Multiplication (MSM)
//! - GPU-accelerated Number Theoretic Transform (NTT)
//! - Automatic device selection (CUDA/CPU fallback)
//! - Memory-efficient batch processing
//!
//! # Requirements
//!
//! - CUDA 12.0+ for NVIDIA GPUs
//! - ICICLE CUDA backend installed (download from Ingonyama releases)
//!
//! # Usage
//!
//! ```rust,ignore
//! use fluidelite_zk::gpu::GpuAccelerator;
//!
//! // Initialize GPU (loads CUDA backend)
//! let gpu = GpuAccelerator::new()?;
//!
//! // Check device info
//! println!("Using device: {}", gpu.device_name());
//!
//! // Run MSM benchmark
//! let duration = gpu.benchmark_msm(1024)?;
//! ```

#[cfg(feature = "gpu")]
use icicle_runtime::runtime;
#[cfg(feature = "gpu")]
use icicle_runtime::Device;
#[cfg(feature = "gpu")]
use icicle_core::msm::{msm, MSMConfig};
#[cfg(feature = "gpu")]
use icicle_core::ntt::{ntt, NTTConfig, NTTDir};
#[cfg(feature = "gpu")]
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
#[cfg(feature = "gpu")]
use icicle_core::traits::GenerateRandom;
#[cfg(feature = "gpu")]
use icicle_core::projective::Projective;
#[cfg(feature = "gpu")]
use icicle_runtime::memory::HostSlice;

/// GPU accelerator for ZK operations
#[cfg(feature = "gpu")]
pub struct GpuAccelerator {
    device: Device,
    device_name: String,
    is_cuda: bool,
}

#[cfg(feature = "gpu")]
impl GpuAccelerator {
    /// Initialize GPU acceleration
    ///
    /// Attempts to load the CUDA backend from:
    /// 1. Environment variable `ICICLE_BACKEND_INSTALL_DIR`
    /// 2. Default path `/opt/icicle/lib/backend`
    /// 3. Falls back to CPU if no GPU available
    pub fn new() -> Result<Self, String> {
        Self::with_device_id(0)
    }

    /// Initialize with specific GPU device ID
    pub fn with_device_id(gpu_id: usize) -> Result<Self, String> {
        // Try to load CUDA backend
        let backend_loaded = runtime::load_backend_from_env_or_default();
        
        if backend_loaded.is_err() {
            tracing::warn!("CUDA backend not found, falling back to CPU");
            return Self::cpu_fallback();
        }

        // Try to create CUDA device
        let device = Device::new("CUDA", gpu_id as i32);
        
        match icicle_runtime::set_device(&device) {
            Ok(_) => {
                let device_name = format!("CUDA:{}", gpu_id);
                
                tracing::info!("GPU acceleration enabled: {}", device_name);

                Ok(Self {
                    device,
                    device_name,
                    is_cuda: true,
                })
            }
            Err(e) => {
                tracing::warn!("Failed to set CUDA device: {:?}, falling back to CPU", e);
                Self::cpu_fallback()
            }
        }
    }

    /// Create CPU fallback accelerator
    fn cpu_fallback() -> Result<Self, String> {
        let device = Device::new("CPU", 0);
        icicle_runtime::set_device(&device)
            .map_err(|e| format!("Failed to set CPU device: {:?}", e))?;
        
        Ok(Self {
            device,
            device_name: "CPU".to_string(),
            is_cuda: false,
        })
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Check if using GPU
    pub fn is_gpu(&self) -> bool {
        self.is_cuda
    }

    /// Print GPU status
    pub fn print_status(&self) {
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║           FluidElite GPU Acceleration Status             ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║ Device: {:<48} ║", self.device_name);
        println!("║ Mode:   {:<48} ║", if self.is_cuda { "GPU (CUDA)" } else { "CPU Fallback" });
        println!("╚══════════════════════════════════════════════════════════╝");
    }

    /// GPU-accelerated Multi-Scalar Multiplication (MSM)
    ///
    /// Computes: sum(scalars[i] * points[i]) for i in 0..n
    ///
    /// This is the core operation for KZG polynomial commitments and
    /// is heavily accelerated on GPU (typically 10-100x speedup).
    pub fn msm_bn254(
        &self,
        points: &[G1Affine],
        scalars: &[ScalarField],
    ) -> Result<G1Projective, String> {
        if points.len() != scalars.len() {
            return Err(format!(
                "MSM: points.len()={} != scalars.len()={}",
                points.len(),
                scalars.len()
            ));
        }

        if points.is_empty() {
            return Err("MSM: empty input".to_string());
        }

        let mut result = vec![G1Projective::zero(); 1];
        
        let config = MSMConfig::default();
        
        msm(
            HostSlice::from_slice(scalars),
            HostSlice::from_slice(points),
            &config,
            HostSlice::from_mut_slice(&mut result),
        )
        .map_err(|e| format!("MSM failed: {:?}", e))?;

        Ok(result[0])
    }

    /// GPU-accelerated Number Theoretic Transform (NTT)
    ///
    /// Computes the forward NTT of a polynomial in coefficient form.
    /// Used for fast polynomial multiplication in ZK proofs.
    pub fn ntt_forward(
        &self,
        coeffs: &[ScalarField],
    ) -> Result<Vec<ScalarField>, String> {
        let n = coeffs.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(format!("NTT: input size {} must be power of 2", n));
        }

        let mut output = coeffs.to_vec();
        
        let config = NTTConfig::<ScalarField>::default();
        
        ntt(
            HostSlice::from_slice(coeffs),
            NTTDir::kForward,
            &config,
            HostSlice::from_mut_slice(&mut output),
        )
        .map_err(|e| format!("NTT forward failed: {:?}", e))?;

        Ok(output)
    }

    /// GPU-accelerated Inverse NTT (INTT)
    ///
    /// Computes the inverse NTT to convert from evaluation form back to coefficients.
    pub fn ntt_inverse(
        &self,
        evals: &[ScalarField],
    ) -> Result<Vec<ScalarField>, String> {
        let n = evals.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(format!("INTT: input size {} must be power of 2", n));
        }

        let mut output = evals.to_vec();
        
        let config = NTTConfig::<ScalarField>::default();
        
        ntt(
            HostSlice::from_slice(evals),
            NTTDir::kInverse,
            &config,
            HostSlice::from_mut_slice(&mut output),
        )
        .map_err(|e| format!("INTT failed: {:?}", e))?;

        Ok(output)
    }

    /// Benchmark MSM performance
    pub fn benchmark_msm(&self, size: usize) -> Result<std::time::Duration, String> {
        use std::time::Instant;
        
        // Generate random test data using ICICLE's GenerateRandom trait
        let scalars = ScalarField::generate_random(size);
        let points = G1Affine::generate_random(size);
        
        // Warm up
        let _ = self.msm_bn254(&points[..1], &scalars[..1])?;
        
        // Benchmark
        let start = Instant::now();
        let _ = self.msm_bn254(&points, &scalars)?;
        Ok(start.elapsed())
    }

    /// Benchmark NTT performance
    pub fn benchmark_ntt(&self, log_size: u32) -> Result<std::time::Duration, String> {
        use std::time::Instant;
        
        let size = 1usize << log_size;
        
        // Generate random test data
        let coeffs = ScalarField::generate_random(size);
        
        // Warm up
        if size >= 2 {
            let _ = self.ntt_forward(&coeffs[..2])?;
        }
        
        // Benchmark
        let start = Instant::now();
        let _ = self.ntt_forward(&coeffs)?;
        Ok(start.elapsed())
    }
}

// Stub implementation when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub struct GpuAccelerator;

#[cfg(not(feature = "gpu"))]
impl GpuAccelerator {
    /// Create new GPU accelerator (stub - returns error)
    pub fn new() -> Result<Self, String> {
        Err("GPU acceleration requires 'gpu' feature. Build with: cargo build --features gpu".to_string())
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        "N/A"
    }

    /// Check if using GPU
    pub fn is_gpu(&self) -> bool {
        false
    }

    /// Print status
    pub fn print_status(&self) {
        println!("GPU acceleration not enabled. Build with --features gpu");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_stub() {
        #[cfg(not(feature = "gpu"))]
        {
            let result = GpuAccelerator::new();
            assert!(result.is_err());
        }
    }
}
