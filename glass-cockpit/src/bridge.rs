/*!
 * Sovereign RAM Bridge Reader
 * 
 * Doctrine 2: RAM Bridge Protocol
 * 
 * Zero-copy, lock-free shared memory reader for HyperTensor simulation data.
 * 
 * Constitutional Compliance:
 * - Read-only access (no bidirectional communication)
 * - Stale reads acceptable (no synchronization)
 * - Binary format (no serialization)
 */

use anyhow::{Result, Context, bail};
use memmap2::Mmap;
use std::fs::OpenOptions;

const MAGIC_NUMBER: u32 = 0x48545342; // "HTSB"
const VERSION: u32 = 1;

/// System telemetry from simulation
#[derive(Debug, Clone, Copy)]
pub struct Telemetry {
    pub p_core_utilization: f32,
    pub e_core_utilization: f32,
    pub memory_usage_gb: f32,
    // Phase 2 scaffolding: Telemetry overlay will display these metrics
    pub _gpu_utilization: f32,
    pub _mean_frame_time_ms: f32,
    pub _max_frame_time_ms: f32,
    pub stability_score: f32,  // Used by overlay.rs
    pub _qtt_compression_ratio: f32,
    pub _mean_tensor_rank: f32,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            p_core_utilization: 0.0,
            e_core_utilization: 0.0,
            memory_usage_gb: 0.0,
            _gpu_utilization: 0.0,
            _mean_frame_time_ms: 0.0,
            _max_frame_time_ms: 0.0,
            stability_score: 1.0,
            _qtt_compression_ratio: 1.0,
            _mean_tensor_rank: 0.0,
        }
    }
}

/// RAM bridge reader
pub struct SovereignBridge {
    mmap: Mmap,
    // Phase 2-3 scaffolding: Grid dimensions for tensor field extraction
    _grid_width: u32,
    _grid_height: u32,
    _grid_depth: u32,
}

impl SovereignBridge {
    /// Connect to the RAM bridge
    /// 
    /// Path should be `/dev/shm/sovereign_bridge` on Linux/WSL
    /// Windows will access via `\\wsl$\Ubuntu\dev\shm\sovereign_bridge`
    pub fn connect(path: &str) -> Result<Self> {
        // On Windows, translate to WSL path
        #[cfg(target_os = "windows")]
        let path = if !path.starts_with("\\\\wsl") {
            format!("\\\\wsl$\\Ubuntu{}", path)
        } else {
            path.to_string()
        };
        
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .context(format!("Failed to open RAM bridge at {}", path))?;
        
        let mmap = unsafe { 
            memmap2::MmapOptions::new()
                .map(&file)
                .context("Failed to memory-map RAM bridge")?
        };
        
        // Validate magic number
        let magic = Self::read_u32(&mmap, 0x00);
        if magic != MAGIC_NUMBER {
            bail!("Invalid magic number: expected 0x{:08X}, got 0x{:08X}", 
                  MAGIC_NUMBER, magic);
        }
        
        // Validate version
        let version = Self::read_u32(&mmap, 0x04);
        if version != VERSION {
            bail!("Unsupported version: expected {}, got {}", VERSION, version);
        }
        
        // Read grid dimensions
        let grid_width = Self::read_u32(&mmap, 0x24);
        let grid_height = Self::read_u32(&mmap, 0x28);
        let grid_depth = Self::read_u32(&mmap, 0x2C);
        
        Ok(Self {
            mmap,
            _grid_width: grid_width,
            _grid_height: grid_height,
            _grid_depth: grid_depth,
        })
    }
    
    /// Read current frame index
    pub fn read_frame_index(&self) -> u64 {
        Self::read_u64(&self.mmap, 0x10)
    }
    
    // Phase 2 scaffolding: RAM bridge access methods for telemetry overlay and tensor viz
    #[allow(dead_code)]
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
            _gpu_utilization: Self::read_f32(&self.mmap, offset + 0x10),
            _mean_frame_time_ms: Self::read_f32(&self.mmap, offset + 0x20),
            _max_frame_time_ms: Self::read_f32(&self.mmap, offset + 0x24),
            stability_score: Self::read_f32(&self.mmap, offset + 0x2C),
            _qtt_compression_ratio: Self::read_f32(&self.mmap, offset + 0x48),
            _mean_tensor_rank: Self::read_f32(&self.mmap, offset + 0x4C),
        }
    }
    
    #[allow(dead_code)]
    /// Get grid dimensions
    pub fn grid_dimensions(&self) -> (u32, u32, u32) {
        (self._grid_width, self._grid_height, self._grid_depth)
    }
    
    #[allow(dead_code)]
    /// Read tensor data as flat array
    pub fn read_tensor_data(&self) -> Vec<f32> {
        let offset = 512; // After header + telemetry
        let count = (self._grid_width * self._grid_height * self._grid_depth) as usize;
        
        (0..count)
            .map(|i| Self::read_f32(&self.mmap, offset + i * 4))
            .collect()
    }
    
    #[allow(dead_code)]
    /// Read vector field data as flat array of (x, y, z) tuples
    pub fn read_vector_data(&self) -> Vec<(f32, f32, f32)> {
        let tensor_size = (self._grid_width * self._grid_height * self._grid_depth * 4) as usize;
        let offset = 512 + tensor_size;
        let count = (self._grid_width * self._grid_height * self._grid_depth) as usize;
        
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
    
    // Helper functions for reading little-endian values
    
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
        assert_eq!(t.stability_score, 1.0);
        assert_eq!(t.qtt_compression_ratio, 1.0);
    }
}
