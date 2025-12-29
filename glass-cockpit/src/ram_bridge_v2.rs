use std::path::PathBuf;
use std::{fs::OpenOptions, io, mem};
use memmap2::MmapMut;

// Phase 3 scaffolding: Tensor bridge protocol v2 constants
#[allow(dead_code)]
/// Magic number for protocol validation: "TNSR"
const TENSOR_BRIDGE_MAGIC: [u8; 4] = [b'T', b'N', b'S', b'R'];

#[allow(dead_code)]
/// Protocol version
const TENSOR_BRIDGE_VERSION: u32 = 1;

// Phase 3 scaffolding: Tensor bridge protocol v2 header structure
#[allow(dead_code)]
/// RAM Bridge Protocol v2: Structured tensor data streaming
///
/// Memory layout:
/// - Header (4096 bytes, cache-aligned)
/// - Data buffer (8MB for 1920×1080 RGBA8)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorBridgeHeader {
    /// Magic number "TNSR" for validation
    pub magic: [u8; 4],
    
    /// Protocol version (1)
    pub version: u32,
    
    /// Monotonic frame counter (increments with each write)
    pub frame_number: u64,
    
    // Tensor Metadata
    /// Image width in pixels (1920)
    pub width: u32,
    
    /// Image height in pixels (1080)
    pub height: u32,
    
    /// Number of channels (4 for RGBA8)
    pub channels: u32,
    
    /// Byte offset to pixel data (4096)
    pub data_offset: u32,
    
    /// Byte size of pixel data
    pub data_size: u32,
    
    // Statistics (computed by Python)
    /// Global minimum tensor value
    pub tensor_min: f32,
    
    /// Global maximum tensor value
    pub tensor_max: f32,
    
    /// Global mean tensor value
    pub tensor_mean: f32,
    
    /// Global standard deviation
    pub tensor_std: f32,
    
    // Synchronization
    /// Producer (Python) timestamp in microseconds
    pub producer_timestamp_us: u64,
    
    /// Consumer (Rust) timestamp in microseconds
    pub consumer_timestamp_us: u64,
    
    /// Padding to 4KB for cache alignment
    _padding: [u8; 3960],
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
            data_offset: 4096,
            data_size: 1920 * 1080 * 4,
            tensor_min: 0.0,
            tensor_max: 1.0,
            tensor_mean: 0.5,
            tensor_std: 0.25,
            producer_timestamp_us: 0,
            consumer_timestamp_us: 0,
            _padding: [0; 3960],
        }
    }
}

// Phase 3 scaffolding: RAM Bridge v2 reader for tensor visualization
#[allow(dead_code)]
/// RAM Bridge Reader v2
///
/// Reads structured tensor data from shared memory file written by Python.
/// Supports frame synchronization, statistics, and error detection.
pub struct RamBridgeV2 {
    /// Memory-mapped file handle
    mmap: MmapMut,
    
    /// Last frame number successfully read
    last_frame_number: u64,
    
    /// Frame drop counter (for diagnostics)
    frame_drops: u64,
}

// Phase 3 scaffolding: RAM Bridge v2 implementation for tensor streaming
#[allow(dead_code)]
impl RamBridgeV2 {
    /// Connect to RAM bridge
    ///
    /// # Arguments
    /// * `path` - Path to shared memory file (e.g., "/dev/shm/hypertensor_bridge")
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully connected
    /// * `Err(io::Error)` - Bridge not available (simulation not running)
    pub fn connect(path: PathBuf) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Validate header on connect
        let header = Self::read_header(&mmap)?;
        
        if header.magic != TENSOR_BRIDGE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic number: expected {:?}, got {:?}", 
                        TENSOR_BRIDGE_MAGIC, header.magic),
            ));
        }
        
        if header.version != TENSOR_BRIDGE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported protocol version: {}", header.version),
            ));
        }
        
        Ok(Self {
            mmap,
            last_frame_number: header.frame_number.saturating_sub(1),
            frame_drops: 0,
        })
    }
    
    /// Read header from memory-mapped buffer
    fn read_header(mmap: &[u8]) -> io::Result<TensorBridgeHeader> {
        if mmap.len() < mem::size_of::<TensorBridgeHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Buffer too small for header",
            ));
        }
        
        let header_bytes = &mmap[..mem::size_of::<TensorBridgeHeader>()];
        let header: TensorBridgeHeader = unsafe {
            std::ptr::read(header_bytes.as_ptr() as *const TensorBridgeHeader)
        };
        
        Ok(header)
    }
    
    /// Read current frame data
    ///
    /// # Returns
    /// * `Ok(Some((header, data)))` - New frame available
    /// * `Ok(None)` - No new frame (same frame number as last read)
    /// * `Err(io::Error)` - Invalid data or protocol error
    pub fn read_frame(&mut self) -> io::Result<Option<(TensorBridgeHeader, Vec<u8>)>> {
        let header = Self::read_header(&self.mmap)?;
        
        // Check if this is a new frame
        if header.frame_number <= self.last_frame_number {
            return Ok(None); // No new frame
        }
        
        // Detect frame drops
        let expected_frame = self.last_frame_number + 1;
        if header.frame_number > expected_frame {
            let drops = header.frame_number - expected_frame;
            self.frame_drops += drops;
            eprintln!(
                "⚠ Frame drop detected: expected {}, got {} (dropped {})",
                expected_frame, header.frame_number, drops
            );
        }
        
        self.last_frame_number = header.frame_number;
        
        // Read pixel data
        let data_start = header.data_offset as usize;
        let data_end = data_start + header.data_size as usize;
        
        if data_end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Data buffer overflow: expected {} bytes, file has {}",
                    data_end,
                    self.mmap.len()
                ),
            ));
        }
        
        let data = self.mmap[data_start..data_end].to_vec();
        
        Ok(Some((header, data)))
    }
    
    /// Get frame drop statistics
    pub fn frame_drops(&self) -> u64 {
        self.frame_drops
    }
    
    /// Get last successfully read frame number
    pub fn last_frame_number(&self) -> u64 {
        self.last_frame_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::slice;
    
    #[test]
    fn test_header_size() {
        // Header must be exactly 4096 bytes for cache alignment
        assert_eq!(mem::size_of::<TensorBridgeHeader>(), 4096);
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
    fn test_bridge_connect_missing_file() {
        let result = RamBridgeV2::connect(PathBuf::from("/tmp/nonexistent_bridge"));
        assert!(result.is_err());
    }
    
    // Integration test with mock data (requires tempfile crate)
    #[test]
    #[ignore] // Requires tempfile dev-dependency
    fn test_bridge_read_mock_frame() {
        // use tempfile::NamedTempFile;
        
        // Create temporary file
        let mut tmpfile = NamedTempFile::new().unwrap();
        
        // Write header
        let header = TensorBridgeHeader {
            frame_number: 42,
            ..Default::default()
        };
        
        let header_bytes = unsafe {
            slice::from_raw_parts(
                &header as *const TensorBridgeHeader as *const u8,
                mem::size_of::<TensorBridgeHeader>(),
            )
        };
        tmpfile.write_all(header_bytes).unwrap();
        
        // Write dummy pixel data (1920×1080×4 = 8MB)
        let pixel_data = vec![128u8; 1920 * 1080 * 4];
        tmpfile.write_all(&pixel_data).unwrap();
        tmpfile.flush().unwrap();
        
        // Test read
        let mut bridge = RamBridgeV2::connect(tmpfile.path().to_path_buf()).unwrap();
        
        let result = bridge.read_frame().unwrap();
        assert!(result.is_some());
        
        let (read_header, read_data) = result.unwrap();
        assert_eq!(read_header.frame_number, 42);
        assert_eq!(read_data.len(), 1920 * 1080 * 4);
        assert_eq!(read_data[0], 128);
    }
}
