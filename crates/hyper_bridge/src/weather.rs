//! Weather Bridge Protocol - Global Eye Phase 1B-5
//!
//! Defines the shared memory protocol between Python (producer) and Rust (consumer).
//! This MUST match exactly with tensornet/sovereign/protocol.py
//!
//! # Memory Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ WeatherHeader (72 bytes)                                        │
//! │ ├── magic: u32 = 0x474C4F42 ("GLOB")                           │
//! │ ├── version: u32                                                │
//! │ ├── timestamp: u64                                              │
//! │ ├── valid_time: u64                                             │
//! │ ├── grid_w, grid_h: u32                                         │
//! │ ├── lat_min, lat_max, lon_min, lon_max: f32                    │
//! │ ├── max_wind_speed, mean_wind_speed: f32                       │
//! │ ├── frame_number: u64                                           │
//! │ ├── is_ready: u32                                               │
//! │ └── _padding: u32                                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ U-Wind Tensor (grid_h × grid_w × 4 bytes)                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ V-Wind Tensor (grid_h × grid_w × 4 bytes)                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::path::Path;
use std::fs::OpenOptions;
use memmap2::Mmap;

use crate::{BridgeError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Protocol Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic number for protocol validation: "GLOB" in little-endian
pub const WEATHER_MAGIC: u32 = 0x47_4C_4F_42;

/// Protocol version
pub const WEATHER_VERSION: u32 = 1;

/// Default shared memory path
pub const WEATHER_SHM_PATH: &str = "/dev/shm/hyper_weather_v1";

/// Header size in bytes
pub const WEATHER_HEADER_SIZE: usize = 72;

// ─────────────────────────────────────────────────────────────────────────────
// Weather Header Structure
// ─────────────────────────────────────────────────────────────────────────────

/// Weather data header for shared memory IPC.
///
/// Must match Python's `WeatherHeader` exactly.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct WeatherHeader {
    // ─── Identification ──────────────────────────────────────────────────────
    /// Magic number: 0x474C4F42 ("GLOB")
    pub magic: u32,
    /// Protocol version
    pub version: u32,
    
    // ─── Temporal ────────────────────────────────────────────────────────────
    /// Unix timestamp when data was written (seconds)
    pub timestamp: u64,
    /// Forecast valid time (Unix seconds)
    pub valid_time: u64,
    
    // ─── Grid Dimensions ─────────────────────────────────────────────────────
    /// Width of the wind tensor
    pub grid_w: u32,
    /// Height of the wind tensor
    pub grid_h: u32,
    
    // ─── Geographic Bounds ───────────────────────────────────────────────────
    /// Southern boundary latitude
    pub lat_min: f32,
    /// Northern boundary latitude
    pub lat_max: f32,
    /// Western boundary longitude
    pub lon_min: f32,
    /// Eastern boundary longitude
    pub lon_max: f32,
    
    // ─── Statistics ──────────────────────────────────────────────────────────
    /// Maximum wind speed in the grid (m/s)
    pub max_wind_speed: f32,
    /// Mean wind speed in the grid (m/s)
    pub mean_wind_speed: f32,
    
    // ─── Synchronization ─────────────────────────────────────────────────────
    /// Monotonic frame counter
    pub frame_number: u64,
    /// Ready flag: 1 = data is valid and ready
    pub is_ready: u32,
    
    // ─── Padding ─────────────────────────────────────────────────────────────
    /// Reserved for alignment
    pub _padding: u32,
}

impl WeatherHeader {
    /// Validate the header magic and version
    pub fn validate(&self) -> Result<()> {
        if self.magic != WEATHER_MAGIC {
            return Err(BridgeError::InvalidMagic {
                expected: WEATHER_MAGIC.to_le_bytes(),
                actual: self.magic.to_le_bytes(),
            });
        }
        
        if self.version != WEATHER_VERSION {
            return Err(BridgeError::UnsupportedVersion(self.version));
        }
        
        Ok(())
    }
    
    /// Check if data is ready to read
    pub fn is_data_ready(&self) -> bool {
        self.is_ready == 1
    }
    
    /// Get pixel count (total grid elements)
    pub fn pixel_count(&self) -> usize {
        (self.grid_w as usize) * (self.grid_h as usize)
    }
    
    /// Get bytes per tensor layer
    pub fn tensor_bytes(&self) -> usize {
        self.pixel_count() * std::mem::size_of::<f32>()
    }
    
    /// Get total expected file size
    pub fn total_size(&self) -> usize {
        WEATHER_HEADER_SIZE + 2 * self.tensor_bytes()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weather Frame (Zero-Copy View)
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-copy view into weather data from shared memory.
pub struct WeatherFrame {
    /// Memory-mapped file handle (keeps the mapping alive)
    #[allow(dead_code)]
    mmap: Mmap,
    /// Pointer to header (valid while mmap is alive)
    header_ptr: *const WeatherHeader,
    /// U-wind tensor start offset
    u_offset: usize,
    /// V-wind tensor start offset  
    v_offset: usize,
    /// Pixel count
    pixel_count: usize,
}

// SAFETY: WeatherFrame only contains read-only data and offsets
unsafe impl Send for WeatherFrame {}
unsafe impl Sync for WeatherFrame {}

impl WeatherFrame {
    /// Read the header
    pub fn header(&self) -> &WeatherHeader {
        // SAFETY: header_ptr is valid while mmap is alive
        unsafe { &*self.header_ptr }
    }
    
    /// Get U-wind tensor as a slice (east/west component)
    pub fn u_field(&self) -> &[f32] {
        unsafe {
            let ptr = self.mmap.as_ptr().add(self.u_offset) as *const f32;
            std::slice::from_raw_parts(ptr, self.pixel_count)
        }
    }
    
    /// Get V-wind tensor as a slice (north/south component)
    pub fn v_field(&self) -> &[f32] {
        unsafe {
            let ptr = self.mmap.as_ptr().add(self.v_offset) as *const f32;
            std::slice::from_raw_parts(ptr, self.pixel_count)
        }
    }
    
    /// Get wind magnitude at a specific index
    pub fn magnitude_at(&self, idx: usize) -> f32 {
        let u = self.u_field()[idx];
        let v = self.v_field()[idx];
        (u * u + v * v).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weather Reader
// ─────────────────────────────────────────────────────────────────────────────

/// Reader for weather data from shared memory.
pub struct WeatherReader {
    /// Path to shared memory file
    path: std::path::PathBuf,
    /// Last frame number we've seen
    last_frame: u64,
}

impl WeatherReader {
    /// Create a new reader for the default shared memory path.
    pub fn new() -> Self {
        Self::with_path(WEATHER_SHM_PATH)
    }
    
    /// Create a new reader for a specific path.
    pub fn with_path<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            last_frame: 0,
        }
    }
    
    /// Check if the shared memory file exists.
    pub fn is_available(&self) -> bool {
        self.path.exists()
    }
    
    /// Read the current weather frame.
    ///
    /// Returns `None` if no new frame is available.
    pub fn read_frame(&mut self) -> Result<Option<WeatherFrame>> {
        if !self.is_available() {
            return Err(BridgeError::NotAvailable(
                format!("Weather SHM not found: {}", self.path.display())
            ));
        }
        
        // Open and map the file
        let file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|e| BridgeError::NotAvailable(e.to_string()))?;
        
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| BridgeError::NotAvailable(e.to_string()))?
        };
        
        // Validate size
        if mmap.len() < WEATHER_HEADER_SIZE {
            return Err(BridgeError::BufferOverflow {
                expected: WEATHER_HEADER_SIZE,
                actual: mmap.len(),
            });
        }
        
        // Read header
        let header_ptr = mmap.as_ptr() as *const WeatherHeader;
        let header = unsafe { &*header_ptr };
        
        // Validate
        header.validate()?;
        
        // Check if ready
        if !header.is_data_ready() {
            return Ok(None);
        }
        
        // Check if new frame
        if header.frame_number <= self.last_frame {
            return Ok(None);
        }
        
        self.last_frame = header.frame_number;
        
        // Validate data size
        let expected_size = header.total_size();
        if mmap.len() < expected_size {
            return Err(BridgeError::BufferOverflow {
                expected: expected_size,
                actual: mmap.len(),
            });
        }
        
        let pixel_count = header.pixel_count();
        let tensor_bytes = header.tensor_bytes();
        
        Ok(Some(WeatherFrame {
            mmap,
            header_ptr,
            u_offset: WEATHER_HEADER_SIZE,
            v_offset: WEATHER_HEADER_SIZE + tensor_bytes,
            pixel_count,
        }))
    }
    
    /// Force read the current frame, ignoring frame number check.
    pub fn read_current(&self) -> Result<WeatherFrame> {
        if !self.is_available() {
            return Err(BridgeError::NotAvailable(
                format!("Weather SHM not found: {}", self.path.display())
            ));
        }
        
        let file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|e| BridgeError::NotAvailable(e.to_string()))?;
        
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| BridgeError::NotAvailable(e.to_string()))?
        };
        
        if mmap.len() < WEATHER_HEADER_SIZE {
            return Err(BridgeError::BufferOverflow {
                expected: WEATHER_HEADER_SIZE,
                actual: mmap.len(),
            });
        }
        
        let header_ptr = mmap.as_ptr() as *const WeatherHeader;
        let header = unsafe { &*header_ptr };
        
        header.validate()?;
        
        let pixel_count = header.pixel_count();
        let tensor_bytes = header.tensor_bytes();
        
        Ok(WeatherFrame {
            mmap,
            header_ptr,
            u_offset: WEATHER_HEADER_SIZE,
            v_offset: WEATHER_HEADER_SIZE + tensor_bytes,
            pixel_count,
        })
    }
}

impl Default for WeatherReader {
    fn default() -> Self {
        Self::new()
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
            std::mem::size_of::<WeatherHeader>(),
            WEATHER_HEADER_SIZE,
            "WeatherHeader size mismatch - must match Python!"
        );
    }
    
    #[test]
    fn test_header_magic() {
        // 0x474C4F42 = 'G' 'L' 'O' 'B' in big-endian (human readable)
        // In little-endian bytes: [0x42, 0x4F, 0x4C, 0x47] = "BOLG"
        let bytes = WEATHER_MAGIC.to_le_bytes();
        assert_eq!(bytes[0], b'B'); // 0x42
        assert_eq!(bytes[1], b'O'); // 0x4F
        assert_eq!(bytes[2], b'L'); // 0x4C
        assert_eq!(bytes[3], b'G'); // 0x47
    }
    
    #[test]
    fn test_reader_not_available() {
        let reader = WeatherReader::with_path("/tmp/nonexistent_weather");
        assert!(!reader.is_available());
    }
}
