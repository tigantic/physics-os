//! Sovereign RAM Bridge: Legacy protocol for telemetry and tensor data
//!
//! This is the original v1 protocol used for system telemetry and vector fields.
//! It coexists with the newer TensorBridge protocol (v2) for heatmap streaming.
//!
//! # Memory Layout (256-byte header + 128-byte telemetry)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ SovereignHeader (256 bytes)                                     │
//! │ ├── magic: u32 = 0x48545342 ("HTSB")                           │
//! │ ├── version: u32                                                │
//! │ ├── flags: u32                                                  │
//! │ ├── _pad0: u32                                                  │
//! │ ├── frame_index: u64                                            │
//! │ ├── timestamp_us: u64                                           │
//! │ ├── simulation_time: f32                                        │
//! │ ├── grid_width, grid_height, grid_depth: u32                    │
//! │ └── _reserved: [u8; 200]                                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Telemetry (128 bytes)                                           │
//! │ ├── Core utilization, memory, GPU metrics                       │
//! │ ├── Frame timing statistics                                     │
//! │ └── QTT compression metrics                                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Tensor Data (grid_w × grid_h × grid_d × 4 bytes)                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Vector Field Data (grid_w × grid_h × grid_d × 12 bytes)         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::fs::OpenOptions;
use std::path::Path;

use crate::{BridgeError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Protocol Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic number for protocol validation: "HTSB" (Ontic Sovereign Bridge)
pub const SOVEREIGN_MAGIC: u32 = 0x48545342;

/// Protocol version
pub const SOVEREIGN_VERSION: u32 = 1;

/// Header size (power-of-2)
pub const SOVEREIGN_HEADER_SIZE: usize = 256;

/// Telemetry block size (power-of-2)
pub const SOVEREIGN_TELEMETRY_SIZE: usize = 128;

/// Default shared memory path
pub const SOVEREIGN_SHM_PATH: &str = "/dev/shm/sovereign_bridge";

// ─────────────────────────────────────────────────────────────────────────────
// Sovereign Header Structure
// ─────────────────────────────────────────────────────────────────────────────

/// Sovereign Bridge header for shared memory IPC.
///
/// Fixed size: 256 bytes (power-of-2, cache-friendly)
#[repr(C, align(256))]
#[derive(Debug, Clone, Copy)]
pub struct SovereignHeader {
    // ─── Identification (16 bytes) ───────────────────────────────────────────
    /// Magic number: 0x48545342 ("HTSB")
    pub magic: u32,
    /// Protocol version
    pub version: u32,
    /// Flags bitfield:
    /// - bit 0: is_ready (data valid)
    /// - bit 1: has_tensor_data
    /// - bit 2: has_vector_data
    /// - bit 3: is_paused
    pub flags: u32,
    /// Padding for alignment
    pub _pad0: u32,
    
    // ─── Timing (24 bytes) ───────────────────────────────────────────────────
    /// Monotonic frame counter
    pub frame_index: u64,
    /// Producer timestamp (microseconds since epoch)
    pub timestamp_us: u64,
    /// Simulation time in seconds
    pub simulation_time: f32,
    /// Delta time since last frame (seconds)
    pub delta_time: f32,
    
    // ─── Grid Dimensions (16 bytes) ──────────────────────────────────────────
    /// Grid width (X dimension)
    pub grid_width: u32,
    /// Grid height (Y dimension)
    pub grid_height: u32,
    /// Grid depth (Z dimension)
    pub grid_depth: u32,
    /// Reserved
    pub _pad1: u32,
    
    // ─── Data Layout (16 bytes) ──────────────────────────────────────────────
    /// Offset to tensor data (bytes from file start)
    pub tensor_data_offset: u32,
    /// Size of tensor data in bytes
    pub tensor_data_size: u32,
    /// Offset to vector field data
    pub vector_data_offset: u32,
    /// Size of vector field data in bytes
    pub vector_data_size: u32,
    
    // ─── Reserved for Future Use (184 bytes) ─────────────────────────────────
    pub _reserved: [u8; 184],
}

// SAFETY: SovereignHeader is repr(C) with only primitive types
unsafe impl Pod for SovereignHeader {}
unsafe impl Zeroable for SovereignHeader {}

// Compile-time size and alignment assertions
const _: () = {
    assert!(std::mem::size_of::<SovereignHeader>() == SOVEREIGN_HEADER_SIZE);
    assert!(SOVEREIGN_HEADER_SIZE == 256);
    assert!(SOVEREIGN_HEADER_SIZE.is_power_of_two());
    assert!(std::mem::align_of::<SovereignHeader>() == 256);
};

impl Default for SovereignHeader {
    fn default() -> Self {
        Self {
            magic: SOVEREIGN_MAGIC,
            version: SOVEREIGN_VERSION,
            flags: 0,
            _pad0: 0,
            frame_index: 0,
            timestamp_us: 0,
            simulation_time: 0.0,
            delta_time: 0.0,
            grid_width: 0,
            grid_height: 0,
            grid_depth: 0,
            _pad1: 0,
            tensor_data_offset: (SOVEREIGN_HEADER_SIZE + SOVEREIGN_TELEMETRY_SIZE) as u32,
            tensor_data_size: 0,
            vector_data_offset: 0,
            vector_data_size: 0,
            _reserved: [0; 184],
        }
    }
}

impl SovereignHeader {
    /// Validate the header
    pub fn validate(&self) -> Result<()> {
        if self.magic != SOVEREIGN_MAGIC {
            return Err(BridgeError::InvalidMagic {
                expected: SOVEREIGN_MAGIC.to_le_bytes(),
                actual: self.magic.to_le_bytes(),
            });
        }
        
        if self.version != SOVEREIGN_VERSION {
            return Err(BridgeError::UnsupportedVersion(self.version));
        }
        
        Ok(())
    }
    
    /// Check if data is ready
    pub fn is_ready(&self) -> bool {
        self.flags & 0x01 != 0
    }
    
    /// Check if tensor data is available
    pub fn has_tensor_data(&self) -> bool {
        self.flags & 0x02 != 0
    }
    
    /// Check if vector field data is available
    pub fn has_vector_data(&self) -> bool {
        self.flags & 0x04 != 0
    }
    
    /// Check if simulation is paused
    pub fn is_paused(&self) -> bool {
        self.flags & 0x08 != 0
    }
    
    /// Get total grid elements
    pub fn grid_elements(&self) -> usize {
        (self.grid_width as usize) * (self.grid_height as usize) * (self.grid_depth as usize)
    }
    
    /// Get total size of file (header + telemetry + data)
    pub fn total_size(&self) -> usize {
        let base = SOVEREIGN_HEADER_SIZE + SOVEREIGN_TELEMETRY_SIZE;
        let tensor = self.tensor_data_size as usize;
        let vector = self.vector_data_size as usize;
        base + tensor + vector
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry Structure
// ─────────────────────────────────────────────────────────────────────────────

/// System telemetry from simulation.
///
/// Fixed size: 128 bytes (power-of-2, cache-friendly)
///
/// Tracks CPU, GPU, memory utilization and QTT compression metrics.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct Telemetry {
    // ─── CPU Metrics (16 bytes) ──────────────────────────────────────────────
    /// P-core (performance) utilization (0.0 - 1.0)
    pub p_core_utilization: f32,
    /// E-core (efficiency) utilization (0.0 - 1.0)
    pub e_core_utilization: f32,
    /// Total CPU utilization (0.0 - 1.0)
    pub total_cpu_utilization: f32,
    /// Number of active threads
    pub active_threads: u32,
    
    // ─── Memory Metrics (16 bytes) ───────────────────────────────────────────
    /// System memory usage in GB
    pub memory_usage_gb: f32,
    /// Peak memory usage in GB
    pub peak_memory_gb: f32,
    /// Memory bandwidth utilization (0.0 - 1.0)
    pub memory_bandwidth: f32,
    /// Reserved
    pub _pad0: u32,
    
    // ─── GPU Metrics (16 bytes) ──────────────────────────────────────────────
    /// GPU compute utilization (0.0 - 1.0)
    pub gpu_utilization: f32,
    /// GPU memory usage in GB
    pub gpu_memory_gb: f32,
    /// GPU temperature in Celsius
    pub gpu_temperature: f32,
    /// GPU power draw in Watts
    pub gpu_power_watts: f32,
    
    // ─── Frame Timing (16 bytes) ─────────────────────────────────────────────
    /// Mean frame time in milliseconds
    pub mean_frame_time_ms: f32,
    /// Maximum frame time in milliseconds
    pub max_frame_time_ms: f32,
    /// Minimum frame time in milliseconds
    pub min_frame_time_ms: f32,
    /// Frame time standard deviation
    pub frame_time_std_ms: f32,
    
    // ─── Simulation Quality (16 bytes) ───────────────────────────────────────
    /// Stability score (0.0 = unstable, 1.0 = stable)
    pub stability_score: f32,
    /// Energy conservation error (should be near 0)
    pub energy_error: f32,
    /// CFL number (Courant-Friedrichs-Lewy condition)
    pub cfl_number: f32,
    /// Residual norm
    pub residual_norm: f32,
    
    // ─── QTT Compression Metrics (16 bytes) ──────────────────────────────────
    /// QTT compression ratio (original / compressed)
    pub qtt_compression_ratio: f32,
    /// Mean tensor rank across all QTT tensors
    pub mean_tensor_rank: f32,
    /// Maximum bond dimension used
    pub max_bond_dim: f32,
    /// Truncation error from rSVD
    pub truncation_error: f32,
    
    // ─── Reserved (32 bytes) ─────────────────────────────────────────────────
    pub _reserved: [u8; 32],
}

// SAFETY: Telemetry is repr(C) with only primitive types
unsafe impl Pod for Telemetry {}
unsafe impl Zeroable for Telemetry {}

// Compile-time size and alignment assertions
const _: () = {
    assert!(std::mem::size_of::<Telemetry>() == SOVEREIGN_TELEMETRY_SIZE);
    assert!(SOVEREIGN_TELEMETRY_SIZE == 128);
    assert!(SOVEREIGN_TELEMETRY_SIZE.is_power_of_two());
    assert!(std::mem::align_of::<Telemetry>() == 128);
};

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            p_core_utilization: 0.0,
            e_core_utilization: 0.0,
            total_cpu_utilization: 0.0,
            active_threads: 0,
            memory_usage_gb: 0.0,
            peak_memory_gb: 0.0,
            memory_bandwidth: 0.0,
            _pad0: 0,
            gpu_utilization: 0.0,
            gpu_memory_gb: 0.0,
            gpu_temperature: 0.0,
            gpu_power_watts: 0.0,
            mean_frame_time_ms: 0.0,
            max_frame_time_ms: 0.0,
            min_frame_time_ms: 0.0,
            frame_time_std_ms: 0.0,
            stability_score: 1.0,
            energy_error: 0.0,
            cfl_number: 0.0,
            residual_norm: 0.0,
            qtt_compression_ratio: 1.0,
            mean_tensor_rank: 0.0,
            max_bond_dim: 0.0,
            truncation_error: 0.0,
            _reserved: [0; 32],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sovereign Bridge Reader
// ─────────────────────────────────────────────────────────────────────────────

/// Sovereign RAM Bridge Reader
///
/// Reads telemetry, tensor fields, and vector data from shared memory.
/// This is the v1 protocol for system metrics and field data.
///
/// # Example
///
/// ```rust,ignore
/// let bridge = SovereignBridge::connect("/dev/shm/sovereign_bridge")?;
/// 
/// let header = bridge.header();
/// println!("Frame: {}, Grid: {}x{}x{}", 
///     header.frame_index, header.grid_width, header.grid_height, header.grid_depth);
///
/// let telemetry = bridge.telemetry();
/// println!("QTT Compression: {:.1}x", telemetry.qtt_compression_ratio);
/// ```
pub struct SovereignBridge {
    /// Memory-mapped file handle
    mmap: Mmap,
}

impl SovereignBridge {
    /// Connect to the Sovereign RAM bridge
    ///
    /// # Arguments
    /// * `path` - Path to shared memory file (e.g., "/dev/shm/sovereign_bridge")
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        
        // On Windows, translate to WSL path
        #[cfg(target_os = "windows")]
        let path_str = {
            let p = path_ref.to_string_lossy();
            if !p.starts_with("\\\\wsl") {
                format!("\\\\wsl$\\Ubuntu{}", p)
            } else {
                p.to_string()
            }
        };
        
        #[cfg(not(target_os = "windows"))]
        let path_str = path_ref.to_string_lossy().to_string();
        
        let file = OpenOptions::new()
            .read(true)
            .open(&path_str)
            .map_err(|e| BridgeError::NotAvailable(format!("{}: {}", path_str, e)))?;
        
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| BridgeError::NotAvailable(e.to_string()))?
        };
        
        // Validate minimum size
        let min_size = SOVEREIGN_HEADER_SIZE + SOVEREIGN_TELEMETRY_SIZE;
        if mmap.len() < min_size {
            return Err(BridgeError::BufferOverflow {
                expected: min_size,
                actual: mmap.len(),
            });
        }
        
        // Read and validate header
        let header = Self::read_header_from_mmap(&mmap)?;
        header.validate()?;
        
        Ok(Self { mmap })
    }
    
    /// Read header from memory-mapped buffer
    fn read_header_from_mmap(mmap: &[u8]) -> Result<SovereignHeader> {
        if mmap.len() < SOVEREIGN_HEADER_SIZE {
            return Err(BridgeError::BufferOverflow {
                expected: SOVEREIGN_HEADER_SIZE,
                actual: mmap.len(),
            });
        }
        
        // Use try_pod_read_unaligned to handle potential misalignment
        let header: SovereignHeader = bytemuck::try_pod_read_unaligned(
            &mmap[..SOVEREIGN_HEADER_SIZE]
        ).map_err(|e| BridgeError::NotAvailable(format!("Header read error: {:?}", e)))?;
        
        Ok(header)
    }
    
    /// Get current header
    pub fn header(&self) -> Result<SovereignHeader> {
        Self::read_header_from_mmap(&self.mmap)
    }
    
    /// Get current telemetry
    pub fn telemetry(&self) -> Result<Telemetry> {
        let offset = SOVEREIGN_HEADER_SIZE;
        let end = offset + SOVEREIGN_TELEMETRY_SIZE;
        
        if self.mmap.len() < end {
            return Err(BridgeError::BufferOverflow {
                expected: end,
                actual: self.mmap.len(),
            });
        }
        
        let telemetry: Telemetry = bytemuck::try_pod_read_unaligned(
            &self.mmap[offset..end]
        ).map_err(|e| BridgeError::NotAvailable(format!("Telemetry read error: {:?}", e)))?;
        
        Ok(telemetry)
    }
    
    /// Read current frame index
    pub fn read_frame_index(&self) -> Result<u64> {
        Ok(self.header()?.frame_index)
    }
    
    /// Read simulation time in seconds
    pub fn read_simulation_time(&self) -> Result<f32> {
        Ok(self.header()?.simulation_time)
    }
    
    /// Get grid dimensions
    pub fn grid_dimensions(&self) -> Result<(u32, u32, u32)> {
        let h = self.header()?;
        Ok((h.grid_width, h.grid_height, h.grid_depth))
    }
    
    /// Read tensor data as f32 slice (zero-copy)
    ///
    /// # Safety
    /// The returned slice is only valid while the SovereignBridge exists.
    pub fn tensor_data(&self) -> Result<&[f32]> {
        let header = self.header()?;
        
        if !header.has_tensor_data() || header.tensor_data_size == 0 {
            return Ok(&[]);
        }
        
        let offset = header.tensor_data_offset as usize;
        let size = header.tensor_data_size as usize;
        let end = offset + size;
        
        if end > self.mmap.len() {
            return Err(BridgeError::BufferOverflow {
                expected: end,
                actual: self.mmap.len(),
            });
        }
        
        let ptr = self.mmap[offset..end].as_ptr();
        debug_assert!(
            ptr as usize % std::mem::align_of::<f32>() == 0,
            "Misaligned tensor data at offset {}",
            offset
        );
        
        let count = size / std::mem::size_of::<f32>();
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const f32, count) })
    }
    
    /// Read vector field data as (x, y, z) tuples (zero-copy)
    ///
    /// Returns a slice of [f32; 3] arrays representing 3D vectors.
    pub fn vector_data(&self) -> Result<&[[f32; 3]]> {
        let header = self.header()?;
        
        if !header.has_vector_data() || header.vector_data_size == 0 {
            return Ok(&[]);
        }
        
        let offset = header.vector_data_offset as usize;
        let size = header.vector_data_size as usize;
        let end = offset + size;
        
        if end > self.mmap.len() {
            return Err(BridgeError::BufferOverflow {
                expected: end,
                actual: self.mmap.len(),
            });
        }
        
        let ptr = self.mmap[offset..end].as_ptr();
        debug_assert!(
            ptr as usize % std::mem::align_of::<f32>() == 0,
            "Misaligned vector data at offset {}",
            offset
        );
        
        let count = size / (3 * std::mem::size_of::<f32>());
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const [f32; 3], count) })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_header_size() {
        assert_eq!(
            std::mem::size_of::<SovereignHeader>(),
            SOVEREIGN_HEADER_SIZE,
            "SovereignHeader size mismatch"
        );
        assert_eq!(SOVEREIGN_HEADER_SIZE, 256);
        assert!(SOVEREIGN_HEADER_SIZE.is_power_of_two());
    }
    
    #[test]
    fn test_header_alignment() {
        assert_eq!(
            std::mem::align_of::<SovereignHeader>(),
            256,
            "SovereignHeader must be 256-byte aligned"
        );
    }
    
    #[test]
    fn test_telemetry_size() {
        assert_eq!(
            std::mem::size_of::<Telemetry>(),
            SOVEREIGN_TELEMETRY_SIZE,
            "Telemetry size mismatch"
        );
        assert_eq!(SOVEREIGN_TELEMETRY_SIZE, 128);
        assert!(SOVEREIGN_TELEMETRY_SIZE.is_power_of_two());
    }
    
    #[test]
    fn test_telemetry_alignment() {
        assert_eq!(
            std::mem::align_of::<Telemetry>(),
            128,
            "Telemetry must be 128-byte aligned"
        );
    }
    
    #[test]
    fn test_magic_number() {
        assert_eq!(SOVEREIGN_MAGIC, 0x48545342);
        let bytes = SOVEREIGN_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"BSTH"); // Little-endian: reversed
    }
    
    #[test]
    fn test_header_default() {
        let header = SovereignHeader::default();
        assert_eq!(header.magic, SOVEREIGN_MAGIC);
        assert_eq!(header.version, SOVEREIGN_VERSION);
        assert_eq!(header.tensor_data_offset, (SOVEREIGN_HEADER_SIZE + SOVEREIGN_TELEMETRY_SIZE) as u32);
    }
    
    #[test]
    fn test_header_validate() {
        let header = SovereignHeader::default();
        assert!(header.validate().is_ok());
        
        let mut bad_magic = header;
        bad_magic.magic = 0;
        assert!(bad_magic.validate().is_err());
    }
    
    #[test]
    fn test_telemetry_default() {
        let t = Telemetry::default();
        assert_eq!(t.stability_score, 1.0);
        assert_eq!(t.qtt_compression_ratio, 1.0);
    }
    
    #[test]
    fn test_connect_missing() {
        let result = SovereignBridge::connect("/tmp/nonexistent_sovereign_bridge");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_header_flags() {
        let mut header = SovereignHeader::default();
        
        assert!(!header.is_ready());
        assert!(!header.has_tensor_data());
        assert!(!header.has_vector_data());
        assert!(!header.is_paused());
        
        header.flags = 0x0F;
        assert!(header.is_ready());
        assert!(header.has_tensor_data());
        assert!(header.has_vector_data());
        assert!(header.is_paused());
    }
}
