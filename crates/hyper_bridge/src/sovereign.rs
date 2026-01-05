//! Sovereign RAM Bridge: Legacy protocol for telemetry and tensor data
//!
//! This is the original v1 protocol used for system telemetry and vector fields.
//! It coexists with the newer TensorBridge protocol (v2) for heatmap streaming.

use memmap2::Mmap;
use std::fs::OpenOptions;
use crate::{BridgeError, Result};

const MAGIC_NUMBER: u32 = 0x48545342; // "HTSB" (little-endian)
const VERSION: u32 = 1;

/// System telemetry from simulation
#[derive(Debug, Clone, Copy, Default)]
pub struct Telemetry {
    pub p_core_utilization: f32,
    pub e_core_utilization: f32,
    pub memory_usage_gb: f32,
    pub gpu_utilization: f32,
    pub mean_frame_time_ms: f32,
    pub max_frame_time_ms: f32,
    pub stability_score: f32,
    pub qtt_compression_ratio: f32,
    pub mean_tensor_rank: f32,
}

/// Sovereign RAM Bridge Reader
///
/// Reads telemetry, tensor fields, and vector data from shared memory.
/// This is the v1 protocol for system metrics and field data.
pub struct SovereignBridge {
    /// Memory-mapped file handle
    mmap: Mmap,
    /// Grid dimensions
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
}

impl SovereignBridge {
    /// Connect to the Sovereign RAM bridge
    ///
    /// # Arguments
    /// * `path` - Path to shared memory file (e.g., "/dev/shm/sovereign_bridge")
    pub fn connect(path: &str) -> Result<Self> {
        // On Windows, translate to WSL path
        #[cfg(target_os = "windows")]
        let path = if !path.starts_with("\\\\wsl") {
            format!("\\\\wsl$\\Ubuntu{}", path)
        } else {
            path.to_string()
        };
        
        #[cfg(not(target_os = "windows"))]
        let path = path.to_string();
        
        let file = OpenOptions::new()
            .read(true)
            .open(&path)
            .map_err(|e| BridgeError::NotAvailable(format!("{}: {}", path, e)))?;
        
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| BridgeError::NotAvailable(e.to_string()))?
        };
        
        // Validate magic number
        if mmap.len() < 48 {
            return Err(BridgeError::BufferOverflow {
                expected: 48,
                actual: mmap.len(),
            });
        }
        
        let magic = Self::read_u32(&mmap, 0x00);
        if magic != MAGIC_NUMBER {
            return Err(BridgeError::InvalidMagic {
                expected: MAGIC_NUMBER.to_le_bytes(),
                actual: magic.to_le_bytes(),
            });
        }
        
        let version = Self::read_u32(&mmap, 0x04);
        if version != VERSION {
            return Err(BridgeError::UnsupportedVersion(version));
        }
        
        // Read grid dimensions
        let grid_width = Self::read_u32(&mmap, 0x24);
        let grid_height = Self::read_u32(&mmap, 0x28);
        let grid_depth = Self::read_u32(&mmap, 0x2C);
        
        Ok(Self {
            mmap,
            grid_width,
            grid_height,
            grid_depth,
        })
    }
    
    /// Read current frame index
    pub fn read_frame_index(&self) -> u64 {
        Self::read_u64(&self.mmap, 0x10)
    }
    
    /// Read simulation time in seconds
    pub fn read_simulation_time(&self) -> f32 {
        Self::read_f32(&self.mmap, 0x20)
    }
    
    /// Read telemetry data
    pub fn read_telemetry(&self) -> Telemetry {
        let offset = 256; // After header
        
        Telemetry {
            p_core_utilization: Self::read_f32(&self.mmap, offset),
            e_core_utilization: Self::read_f32(&self.mmap, offset + 0x04),
            memory_usage_gb: Self::read_f32(&self.mmap, offset + 0x08),
            gpu_utilization: Self::read_f32(&self.mmap, offset + 0x10),
            mean_frame_time_ms: Self::read_f32(&self.mmap, offset + 0x20),
            max_frame_time_ms: Self::read_f32(&self.mmap, offset + 0x24),
            stability_score: Self::read_f32(&self.mmap, offset + 0x2C),
            qtt_compression_ratio: Self::read_f32(&self.mmap, offset + 0x48),
            mean_tensor_rank: Self::read_f32(&self.mmap, offset + 0x4C),
        }
    }
    
    /// Get grid dimensions
    pub fn grid_dimensions(&self) -> (u32, u32, u32) {
        (self.grid_width, self.grid_height, self.grid_depth)
    }
    
    /// Read tensor data as flat array
    pub fn read_tensor_data(&self) -> Vec<f32> {
        let offset = 512; // After header + telemetry
        let count = (self.grid_width * self.grid_height * self.grid_depth) as usize;
        
        (0..count)
            .map(|i| Self::read_f32(&self.mmap, offset + i * 4))
            .collect()
    }
    
    /// Read vector field data as flat array of (x, y, z) tuples
    pub fn read_vector_data(&self) -> Vec<(f32, f32, f32)> {
        let tensor_size = (self.grid_width * self.grid_height * self.grid_depth * 4) as usize;
        let offset = 512 + tensor_size;
        let count = (self.grid_width * self.grid_height * self.grid_depth) as usize;
        
        (0..count)
            .map(|i| {
                let base = offset + i * 12;
                (
                    Self::read_f32(&self.mmap, base),
                    Self::read_f32(&self.mmap, base + 4),
                    Self::read_f32(&self.mmap, base + 8),
                )
            })
            .collect()
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Helper functions for reading little-endian values
    // ─────────────────────────────────────────────────────────────────────────
    
    fn read_u32(data: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
    
    fn read_u64(data: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ])
    }
    
    fn read_f32(data: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_magic_number() {
        assert_eq!(MAGIC_NUMBER, 0x48545342);
        let bytes = MAGIC_NUMBER.to_le_bytes();
        assert_eq!(&bytes, b"BSTH"); // Little-endian: reversed
    }
    
    #[test]
    fn test_telemetry_default() {
        let t = Telemetry::default();
        assert_eq!(t.stability_score, 0.0);
    }
    
    #[test]
    fn test_connect_missing() {
        let result = SovereignBridge::connect("/tmp/nonexistent_sovereign_bridge");
        assert!(result.is_err());
    }
}
