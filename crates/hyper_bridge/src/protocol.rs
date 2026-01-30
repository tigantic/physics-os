//! RAM Bridge Protocol v2: Tensor data format and constants
//!
//! Memory layout:
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ HEADER (4096 bytes, cache-aligned)                              │
//! │ ├── magic: [u8; 4] = "TNSR"                                     │
//! │ ├── version: u32 = 1                                            │
//! │ ├── frame_number: u64                                           │
//! │ ├── width, height, channels: u32                                │
//! │ ├── data_offset, data_size: u32                                 │
//! │ ├── tensor_min, tensor_max, tensor_mean, tensor_std: f32        │
//! │ ├── producer_timestamp_us, consumer_timestamp_us: u64           │
//! │ └── _padding: [u8; 3960]                                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ DATA BUFFER (8MB for 1920×1080 RGBA8)                           │
//! │ └── Raw pixel/tensor data                                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

// bytemuck traits implemented manually below due to large alignment padding

/// Magic number for protocol validation: "TNSR"
pub const TENSOR_BRIDGE_MAGIC: [u8; 4] = [b'T', b'N', b'S', b'R'];

/// Protocol version
pub const TENSOR_BRIDGE_VERSION: u32 = 1;

/// Header size (cache-aligned)
pub const HEADER_SIZE: usize = 4096;

/// Default data buffer size (1920×1080×4 = 8MB)
pub const DEFAULT_DATA_SIZE: usize = 1920 * 1080 * 4;

/// RAM Bridge Protocol v2 Header
///
/// This structure is shared between Python and Rust via memory-mapped files.
/// **Critical**: Must match the Python dataclass exactly!
///
/// The header is designed to be exactly 4096 bytes for cache alignment.
/// We use manual Pod/Zeroable implementations since bytemuck doesn't support
/// large arrays in derive macros.
#[repr(C, align(4096))]
#[derive(Debug, Clone, Copy)]
pub struct TensorBridgeHeader {
    /// Magic number "TNSR" for validation
    pub magic: [u8; 4],
    
    /// Protocol version (1)
    pub version: u32,
    
    /// Monotonic frame counter (increments with each write)
    pub frame_number: u64,
    
    // ─────────────────────────────────────────────────────────────────
    // Tensor Metadata
    // ─────────────────────────────────────────────────────────────────
    
    /// Image width in pixels
    pub width: u32,
    
    /// Image height in pixels
    pub height: u32,
    
    /// Number of channels (1=grayscale, 3=RGB, 4=RGBA)
    pub channels: u32,
    
    /// Byte offset to pixel data (typically 4096)
    pub data_offset: u32,
    
    /// Byte size of pixel data
    pub data_size: u32,
    
    // ─────────────────────────────────────────────────────────────────
    // Tensor Statistics (computed by Python)
    // ─────────────────────────────────────────────────────────────────
    
    /// Global minimum tensor value
    pub tensor_min: f32,
    
    /// Global maximum tensor value
    pub tensor_max: f32,
    
    /// Global mean tensor value
    pub tensor_mean: f32,
    
    /// Global standard deviation
    pub tensor_std: f32,
    
    // ─────────────────────────────────────────────────────────────────
    // Synchronization
    // ─────────────────────────────────────────────────────────────────
    
    /// Producer (Python) timestamp in microseconds
    pub producer_timestamp_us: u64,
    
    /// Consumer (Rust) timestamp in microseconds
    pub consumer_timestamp_us: u64,
    
    // Padding is implicit due to #[repr(C, align(4096))]
    // The struct will be padded to 4096 bytes automatically
}

impl Default for TensorBridgeHeader {
    fn default() -> Self {
        Self {
            magic: TENSOR_BRIDGE_MAGIC,
            version: TENSOR_BRIDGE_VERSION,
            frame_number: 0,
            width: 1920,
            height: 1080,
            channels: 4,
            data_offset: HEADER_SIZE as u32,
            data_size: DEFAULT_DATA_SIZE as u32,
            tensor_min: 0.0,
            tensor_max: 1.0,
            tensor_mean: 0.5,
            tensor_std: 0.25,
            producer_timestamp_us: 0,
            consumer_timestamp_us: 0,
        }
    }
}

// SAFETY: TensorBridgeHeader is repr(C) with only primitive types
// and no padding within the data fields themselves (only trailing padding)
unsafe impl bytemuck::Pod for TensorBridgeHeader {}
unsafe impl bytemuck::Zeroable for TensorBridgeHeader {}

// Compile-time size and alignment assertions (Constitutional Article VIII)
const _: () = {
    assert!(std::mem::size_of::<TensorBridgeHeader>() == HEADER_SIZE);
    assert!(HEADER_SIZE == 4096);
    assert!(HEADER_SIZE.is_power_of_two());
    assert!(std::mem::align_of::<TensorBridgeHeader>() == 4096);
};

impl TensorBridgeHeader {
    /// Validate the header magic and version
    pub fn validate(&self) -> crate::Result<()> {
        if self.magic != TENSOR_BRIDGE_MAGIC {
            return Err(crate::BridgeError::InvalidMagic {
                expected: TENSOR_BRIDGE_MAGIC,
                actual: self.magic,
            });
        }
        
        if self.version != TENSOR_BRIDGE_VERSION {
            return Err(crate::BridgeError::UnsupportedVersion(self.version));
        }
        
        Ok(())
    }
    
    /// Get expected total buffer size (header + data)
    pub fn total_size(&self) -> usize {
        self.data_offset as usize + self.data_size as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    
    #[test]
    fn test_header_size() {
        assert_eq!(mem::size_of::<TensorBridgeHeader>(), HEADER_SIZE);
    }
    
    #[test]
    fn test_header_default() {
        let header = TensorBridgeHeader::default();
        assert_eq!(header.magic, TENSOR_BRIDGE_MAGIC);
        assert_eq!(header.version, TENSOR_BRIDGE_VERSION);
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.channels, 4);
        assert_eq!(header.data_offset, 4096);
    }
    
    #[test]
    fn test_header_validate() {
        let header = TensorBridgeHeader::default();
        assert!(header.validate().is_ok());
        
        let mut bad_magic = header;
        bad_magic.magic = [0, 0, 0, 0];
        assert!(bad_magic.validate().is_err());
    }
}
