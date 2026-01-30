//! QTT Bridge Protocol - Native Tensor Train Transmission
//!
//! This protocol transmits QTT cores directly WITHOUT decompression.
//! The Rust consumer evaluates the TT on GPU via WGPU compute shaders.
//!
//! # QTT Doctrine Compliance
//!
//! This protocol enforces the following QTT optimization rules:
//! - **QTT Native**: Cores transmitted in TT format, never decompressed
//! - **No Dense**: Data remains in O(χ²·d·L) format, not O(d^L)
//! - **Higher Scale = Higher Compress**: Bond dimensions adapt to data complexity
//! - **Compression validation**: Ratio must exceed 1.5x to justify QTT overhead
//!
//! # Memory Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ QTTBridgeHeader (512 bytes, cache-aligned)                      │
//! │ ├── magic: [u8; 4] = "QTTB"                                     │
//! │ ├── version: u32 = 1                                            │
//! │ ├── frame_number: u64                                           │
//! │ ├── num_sites: u32 (L)                                          │
//! │ ├── physical_dim: u32 (d)                                       │
//! │ ├── max_bond_dim: u32 (χ_max)                                   │
//! │ ├── compression_ratio: f32                                      │
//! │ ├── truncation_error: f64                                       │
//! │ ├── bond_dims: [u16; 64] (χ per bond)                           │
//! │ ├── core_offsets: [u32; 64] (byte offset per core)              │
//! │ └── timestamps, flags, padding                                  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ TT-Core Data (variable size, 16-byte aligned)                   │
//! │ └── Core[i]: f32[χ_left × d × χ_right] in row-major order       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Core Layout Convention (Constitutional Article II, Section 2.2)
//!
//! Each TT-core has shape `(χ_left, d, χ_right)` stored in row-major order:
//! - χ_left varies slowest (outermost loop)
//! - χ_right varies fastest (innermost loop)
//! - Index: `core[i, j, k] = data[i * d * χ_right + j * χ_right + k]`

use bytemuck::{Pod, Zeroable};
use std::path::Path;
use std::fs::OpenOptions;
use memmap2::Mmap;
use crc::{Crc, CRC_32_ISO_HDLC};

use crate::{BridgeError, Result};

/// CRC32 algorithm for data integrity verification
const CRC32: Crc<u32> = Crc::<u32>::new(&CRC_32_ISO_HDLC);

// ─────────────────────────────────────────────────────────────────────────────
// Protocol Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic number for QTT protocol: "QTTB"
pub const QTT_BRIDGE_MAGIC: [u8; 4] = [b'Q', b'T', b'T', b'B'];

/// Protocol version
pub const QTT_BRIDGE_VERSION: u32 = 1;

/// Maximum tensor train sites (log2 of max grid dimension)
/// For a 2^20 = 1M element grid, we need 20 sites
pub const MAX_QTT_SITES: usize = 64;

/// Header size (power of 2, 512 bytes)
pub const QTT_HEADER_SIZE: usize = 512;

/// Default shared memory path
pub const QTT_SHM_PATH: &str = "/dev/shm/hyper_qtt_v1";

/// Minimum compression ratio to justify QTT transmission
/// Below this, dense transmission is more efficient
pub const MIN_COMPRESSION_RATIO: f32 = 1.5;

/// Maximum acceptable truncation error for physics accuracy
pub const MAX_TRUNCATION_ERROR: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Data Type Enum
// ─────────────────────────────────────────────────────────────────────────────

/// Supported data types for TT-cores
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QTTDataType {
    /// 32-bit float (default, GPU-friendly)
    Float32 = 0,
    /// 64-bit float (high precision physics)
    Float64 = 1,
    /// 16-bit float (bandwidth-optimized)
    Float16 = 2,
}

impl QTTDataType {
    /// Bytes per element for this data type
    pub fn element_size(&self) -> usize {
        match self {
            QTTDataType::Float32 => 4,
            QTTDataType::Float64 => 8,
            QTTDataType::Float16 => 2,
        }
    }
    
    /// Convert from raw u8
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(QTTDataType::Float32),
            1 => Some(QTTDataType::Float64),
            2 => Some(QTTDataType::Float16),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QTT Bridge Header
// ─────────────────────────────────────────────────────────────────────────────

/// QTT Bridge Header
/// 
/// Transmits tensor train cores WITHOUT decompression.
/// Consumer evaluates TT directly on GPU.
///
/// # QTT Doctrine
/// 
/// This header enforces native QTT transmission by:
/// 1. Storing bond dimensions for each site boundary
/// 2. Storing core offsets for direct GPU buffer binding
/// 3. Validating compression ratio exceeds threshold
/// 4. Tracking truncation error for physics accuracy
#[repr(C, align(512))]
#[derive(Debug, Clone, Copy)]
pub struct QTTBridgeHeader {
    // ─── Identification (16 bytes) ───────────────────────────────────────────
    /// Magic number "QTTB"
    pub magic: [u8; 4],
    /// Protocol version
    pub version: u32,
    /// Frame counter (monotonic)
    pub frame_number: u64,
    
    // ─── TT Structure (16 bytes) ─────────────────────────────────────────────
    /// Number of TT sites (L)
    pub num_sites: u32,
    /// Physical dimension per site (d, typically 2 for QTT)
    pub physical_dim: u32,
    /// Maximum bond dimension (χ_max)
    pub max_bond_dim: u32,
    /// Actual number of cores in this frame
    pub num_cores: u32,
    
    // ─── Original Tensor Info (24 bytes) ─────────────────────────────────────
    /// Original tensor dimensions [D0, D1, D2, D3] (0 = unused)
    pub original_shape: [u32; 4],
    /// Original tensor total elements
    pub original_elements: u64,
    
    // ─── Compression Metrics (24 bytes) ──────────────────────────────────────
    /// Compression ratio: original_bytes / compressed_bytes
    /// MUST be > MIN_COMPRESSION_RATIO for QTT to be beneficial
    pub compression_ratio: f32,
    /// Relative truncation error: ||A - Ã|| / ||A||
    /// MUST be < MAX_TRUNCATION_ERROR for physics accuracy
    pub truncation_error: f64,
    /// Mean bond dimension across all bonds
    pub mean_bond_dim: f32,
    /// Maximum singular value (for scaling)
    pub max_singular_value: f32,
    
    // ─── Bond Dimensions (128 bytes) ─────────────────────────────────────────
    /// Bond dimensions: χ[i] = bond dimension between site i and i+1
    /// Length: num_sites - 1 valid entries, remaining are 0
    /// For site i, core shape is (bond_dims[i-1], d, bond_dims[i])
    /// Note: bond_dims[0] is χ between sites 0 and 1
    pub bond_dims: [u16; MAX_QTT_SITES],
    
    // ─── Core Offsets (256 bytes) ────────────────────────────────────────────
    /// Byte offset to each core (relative to data section start)
    /// Core[i] starts at: QTT_HEADER_SIZE + core_offsets[i]
    /// Core[i] size in bytes: core_size(i) method
    pub core_offsets: [u32; MAX_QTT_SITES],
    
    // ─── Flags and Metadata (16 bytes) ───────────────────────────────────────
    /// Flags bitfield:
    /// - bit 0: is_complex (f32 pairs instead of f32)
    /// - bit 1: is_canonical (left-canonical form, optimal for contraction)
    /// - bit 2: has_norm (norm stored separately)
    /// - bit 3: is_periodic (periodic boundary conditions)
    /// - bit 4: is_ready (data is valid and ready to read)
    pub flags: u32,
    
    /// Data type: 0=f32, 1=f64, 2=f16
    pub dtype: u8,
    
    /// rSVD oversampling parameter used during compression
    pub rsvd_oversampling: u8,
    
    /// Number of power iterations in rSVD
    pub rsvd_power_iters: u8,
    
    /// Reserved
    pub _reserved0: u8,
    
    /// Total data size in bytes (all cores combined)
    pub total_data_bytes: u32,
    
    /// CRC32 checksum of core data (for integrity)
    pub data_checksum: u32,
    
    // ─── Timestamps (16 bytes) ───────────────────────────────────────────────
    /// Producer timestamp (microseconds since epoch)
    pub producer_timestamp_us: u64,
    /// Consumer timestamp (set by Rust when read)
    pub consumer_timestamp_us: u64,
    
    // ─── Padding to 512 bytes ────────────────────────────────────────────────
    /// Explicit padding
    pub _padding: [u8; 16],
}

// SAFETY: QTTBridgeHeader is repr(C) with only primitive types
unsafe impl Pod for QTTBridgeHeader {}
unsafe impl Zeroable for QTTBridgeHeader {}

// Compile-time size and alignment assertions (Constitutional Article VIII)
const _: () = {
    assert!(std::mem::size_of::<QTTBridgeHeader>() == QTT_HEADER_SIZE);
    assert!(QTT_HEADER_SIZE == 512);
    assert!(QTT_HEADER_SIZE.is_power_of_two());
    assert!(std::mem::align_of::<QTTBridgeHeader>() == 512);
};

impl Default for QTTBridgeHeader {
    fn default() -> Self {
        Self {
            magic: QTT_BRIDGE_MAGIC,
            version: QTT_BRIDGE_VERSION,
            frame_number: 0,
            num_sites: 0,
            physical_dim: 2, // Standard QTT
            max_bond_dim: 64,
            num_cores: 0,
            original_shape: [0; 4],
            original_elements: 0,
            compression_ratio: 1.0,
            truncation_error: 0.0,
            mean_bond_dim: 0.0,
            max_singular_value: 1.0,
            bond_dims: [0; MAX_QTT_SITES],
            core_offsets: [0; MAX_QTT_SITES],
            flags: 0,
            dtype: 0,
            rsvd_oversampling: 10,
            rsvd_power_iters: 2,
            _reserved0: 0,
            total_data_bytes: 0,
            data_checksum: 0,
            producer_timestamp_us: 0,
            consumer_timestamp_us: 0,
            _padding: [0; 16],
        }
    }
}

impl QTTBridgeHeader {
    /// Validate header magic and version
    pub fn validate(&self) -> Result<()> {
        if self.magic != QTT_BRIDGE_MAGIC {
            return Err(BridgeError::InvalidMagic {
                expected: QTT_BRIDGE_MAGIC,
                actual: self.magic,
            });
        }
        
        if self.version != QTT_BRIDGE_VERSION {
            return Err(BridgeError::UnsupportedVersion(self.version));
        }
        
        if self.num_sites as usize > MAX_QTT_SITES {
            return Err(BridgeError::BufferOverflow {
                expected: MAX_QTT_SITES,
                actual: self.num_sites as usize,
            });
        }
        
        if self.physical_dim == 0 {
            return Err(BridgeError::NotAvailable(
                "Physical dimension cannot be 0".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Validate QTT doctrine compliance
    /// 
    /// Ensures the QTT transmission is actually beneficial compared to dense.
    /// Returns error if compression ratio is too low or truncation error too high.
    pub fn validate_doctrine(&self) -> Result<()> {
        // Rule: No benefit if ratio < MIN_COMPRESSION_RATIO
        if self.compression_ratio < MIN_COMPRESSION_RATIO {
            return Err(BridgeError::NotAvailable(format!(
                "QTT DOCTRINE VIOLATION: compression_ratio {:.2} < {:.1}, use dense instead",
                self.compression_ratio, MIN_COMPRESSION_RATIO
            )));
        }
        
        // Rule: Truncation error must preserve physics accuracy
        if self.truncation_error > MAX_TRUNCATION_ERROR {
            return Err(BridgeError::NotAvailable(format!(
                "QTT DOCTRINE VIOLATION: truncation_error {:.2e} > {:.0e}",
                self.truncation_error, MAX_TRUNCATION_ERROR
            )));
        }
        
        Ok(())
    }
    
    /// Check if data is complex-valued
    pub fn is_complex(&self) -> bool {
        self.flags & 0x01 != 0
    }
    
    /// Check if in left-canonical form
    pub fn is_canonical(&self) -> bool {
        self.flags & 0x02 != 0
    }
    
    /// Check if norm is stored separately
    pub fn has_norm(&self) -> bool {
        self.flags & 0x04 != 0
    }
    
    /// Check if periodic boundary conditions
    pub fn is_periodic(&self) -> bool {
        self.flags & 0x08 != 0
    }
    
    /// Check if data is ready to read
    pub fn is_ready(&self) -> bool {
        self.flags & 0x10 != 0
    }
    
    /// Get data type
    pub fn data_type(&self) -> Option<QTTDataType> {
        QTTDataType::from_u8(self.dtype)
    }
    
    /// Get total header + data size
    pub fn total_size(&self) -> usize {
        QTT_HEADER_SIZE + self.total_data_bytes as usize
    }
    
    /// Get left bond dimension for site i
    /// 
    /// For site 0, χ_left = 1 (boundary condition)
    /// For site i > 0, χ_left = bond_dims[i-1]
    pub fn chi_left(&self, site: usize) -> usize {
        if site == 0 {
            1
        } else if site < self.num_sites as usize {
            self.bond_dims[site - 1] as usize
        } else {
            0
        }
    }
    
    /// Get right bond dimension for site i
    /// 
    /// For last site, χ_right = 1 (boundary condition)
    /// For site i < L-1, χ_right = bond_dims[i]
    pub fn chi_right(&self, site: usize) -> usize {
        if site >= self.num_sites as usize {
            0
        } else if site == self.num_sites as usize - 1 {
            1
        } else {
            self.bond_dims[site] as usize
        }
    }
    
    /// Calculate expected core size in bytes for site i
    /// 
    /// Core[i] shape: (χ_left, d, χ_right)
    /// Size = χ_left × d × χ_right × element_size
    pub fn core_size(&self, site: usize) -> usize {
        if site >= self.num_sites as usize {
            return 0;
        }
        
        let chi_left = self.chi_left(site);
        let chi_right = self.chi_right(site);
        let d = self.physical_dim as usize;
        
        let element_size = self.data_type()
            .map(|dt| dt.element_size())
            .unwrap_or(4);
        
        chi_left * d * chi_right * element_size
    }
    
    /// Calculate expected core element count for site i
    pub fn core_elements(&self, site: usize) -> usize {
        if site >= self.num_sites as usize {
            return 0;
        }
        
        let chi_left = self.chi_left(site);
        let chi_right = self.chi_right(site);
        let d = self.physical_dim as usize;
        
        chi_left * d * chi_right
    }
    
    /// Verify compression ratio is beneficial
    /// 
    /// Returns true if QTT transmission saves bandwidth vs dense
    pub fn is_compression_beneficial(&self) -> bool {
        self.compression_ratio >= MIN_COMPRESSION_RATIO
    }
    
    /// Calculate theoretical compression based on bond dims
    pub fn calculate_compression_ratio(&self) -> f32 {
        if self.num_sites == 0 || self.original_elements == 0 {
            return 1.0;
        }
        
        let element_size = self.data_type()
            .map(|dt| dt.element_size())
            .unwrap_or(4);
        
        let original_bytes = self.original_elements as usize * element_size;
        let compressed_bytes = self.total_data_bytes as usize;
        
        if compressed_bytes == 0 {
            return 1.0;
        }
        
        original_bytes as f32 / compressed_bytes as f32
    }
    
    /// Compute CRC32 checksum of the given data
    pub fn compute_checksum(data: &[u8]) -> u32 {
        CRC32.checksum(data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QTT Frame (Zero-Copy View)
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-copy view into QTT data from shared memory.
/// 
/// Provides direct access to TT-cores without decompression.
/// Use `core(i)` to get a slice of core tensor data for GPU upload.
pub struct QTTFrame {
    /// Memory-mapped file handle (keeps the mapping alive)
    #[allow(dead_code)]
    mmap: Mmap,
    /// Copy of header for fast access
    header: QTTBridgeHeader,
}

// SAFETY: QTTFrame only contains read-only data
unsafe impl Send for QTTFrame {}
unsafe impl Sync for QTTFrame {}

impl QTTFrame {
    /// Get the header
    pub fn header(&self) -> &QTTBridgeHeader {
        &self.header
    }
    
    /// Get TT-core data for site i as a byte slice
    /// 
    /// Returns None if site index is out of range.
    /// The returned slice can be directly uploaded to GPU.
    pub fn core_bytes(&self, site: usize) -> Option<&[u8]> {
        if site >= self.header.num_sites as usize {
            return None;
        }
        
        let offset = QTT_HEADER_SIZE + self.header.core_offsets[site] as usize;
        let size = self.header.core_size(site);
        let end = offset + size;
        
        if end > self.mmap.len() {
            return None;
        }
        
        Some(&self.mmap[offset..end])
    }
    
    /// Get TT-core data for site i as f32 slice
    /// 
    /// Returns None if site index is out of range or dtype is not f32.
    /// 
    /// # Panics
    /// 
    /// Debug panics if core data is misaligned.
    pub fn core_f32(&self, site: usize) -> Option<&[f32]> {
        if self.header.dtype != QTTDataType::Float32 as u8 {
            return None;
        }
        
        let bytes = self.core_bytes(site)?;
        let ptr = bytes.as_ptr();
        
        debug_assert!(
            ptr as usize % std::mem::align_of::<f32>() == 0,
            "FATAL: Misaligned f32 access in core {} at offset {}",
            site, self.header.core_offsets[site]
        );
        
        let count = self.header.core_elements(site);
        Some(unsafe { std::slice::from_raw_parts(ptr as *const f32, count) })
    }
    
    /// Get all cores as contiguous byte slice
    /// 
    /// Useful for single GPU buffer upload.
    pub fn all_cores_bytes(&self) -> &[u8] {
        let start = QTT_HEADER_SIZE;
        let end = start + self.header.total_data_bytes as usize;
        
        if end > self.mmap.len() {
            return &[];
        }
        
        &self.mmap[start..end]
    }
    
    /// Get number of sites
    pub fn num_sites(&self) -> usize {
        self.header.num_sites as usize
    }
    
    /// Get physical dimension
    pub fn physical_dim(&self) -> usize {
        self.header.physical_dim as usize
    }
    
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.header.compression_ratio
    }
    
    /// Get truncation error
    pub fn truncation_error(&self) -> f64 {
        self.header.truncation_error
    }
    
    /// Verify data integrity using CRC32 checksum
    ///
    /// Returns Ok(()) if checksum matches, Err if corrupted or checksum is 0 (not set).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let frame = reader.read_frame()?.unwrap();
    /// if frame.verify_checksum().is_err() {
    ///     eprintln!("WARNING: QTT data corruption detected!");
    /// }
    /// ```
    pub fn verify_checksum(&self) -> Result<()> {
        let stored_checksum = self.header.data_checksum;
        
        // If checksum is 0, it wasn't set by the producer
        if stored_checksum == 0 {
            return Err(BridgeError::NotAvailable(
                "CRC32 checksum not set by producer".to_string()
            ));
        }
        
        let data = self.all_cores_bytes();
        if data.is_empty() {
            return Err(BridgeError::NotAvailable(
                "No core data to verify".to_string()
            ));
        }
        
        let computed_checksum = QTTBridgeHeader::compute_checksum(data);
        
        if computed_checksum != stored_checksum {
            return Err(BridgeError::NotAvailable(format!(
                "CRC32 checksum mismatch: stored={:#010x}, computed={:#010x}",
                stored_checksum, computed_checksum
            )));
        }
        
        Ok(())
    }
    
    /// Compute CRC32 checksum of core data (for debugging/validation)
    pub fn compute_checksum(&self) -> u32 {
        QTTBridgeHeader::compute_checksum(self.all_cores_bytes())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QTT Reader
// ─────────────────────────────────────────────────────────────────────────────

/// Reader for QTT data from shared memory.
/// 
/// # Example
/// 
/// ```rust,ignore
/// let mut reader = QTTReader::new();
/// 
/// if let Some(frame) = reader.read_frame()? {
///     println!("Sites: {}, Compression: {:.1}x", 
///         frame.num_sites(), frame.compression_ratio());
///     
///     // Get core 0 for GPU upload
///     if let Some(core_data) = frame.core_f32(0) {
///         // Upload to GPU buffer
///     }
/// }
/// ```
pub struct QTTReader {
    /// Path to shared memory file
    path: std::path::PathBuf,
    /// Last frame number we've seen
    last_frame: u64,
}

impl QTTReader {
    /// Create a new reader for the default shared memory path.
    pub fn new() -> Self {
        Self::with_path(QTT_SHM_PATH)
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
    
    /// Read the current QTT frame.
    /// 
    /// Returns `None` if no new frame is available.
    /// Validates both protocol and QTT doctrine compliance.
    pub fn read_frame(&mut self) -> Result<Option<QTTFrame>> {
        if !self.is_available() {
            return Err(BridgeError::NotAvailable(
                format!("QTT SHM not found: {}", self.path.display())
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
        if mmap.len() < QTT_HEADER_SIZE {
            return Err(BridgeError::BufferOverflow {
                expected: QTT_HEADER_SIZE,
                actual: mmap.len(),
            });
        }
        
        // Read header
        let header: QTTBridgeHeader = *bytemuck::from_bytes(
            &mmap[..std::mem::size_of::<QTTBridgeHeader>()]
        );
        
        // Validate protocol
        header.validate()?;
        
        // Check if ready
        if !header.is_ready() {
            return Ok(None);
        }
        
        // Check if new frame
        if header.frame_number <= self.last_frame {
            return Ok(None);
        }
        
        // Validate QTT doctrine (compression must be beneficial)
        header.validate_doctrine()?;
        
        self.last_frame = header.frame_number;
        
        // Validate data size
        let expected_size = header.total_size();
        if mmap.len() < expected_size {
            return Err(BridgeError::BufferOverflow {
                expected: expected_size,
                actual: mmap.len(),
            });
        }
        
        Ok(Some(QTTFrame { mmap, header }))
    }
    
    /// Force read the current frame, ignoring frame number check.
    /// 
    /// Still validates protocol and doctrine compliance.
    pub fn read_current(&self) -> Result<QTTFrame> {
        if !self.is_available() {
            return Err(BridgeError::NotAvailable(
                format!("QTT SHM not found: {}", self.path.display())
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
        
        if mmap.len() < QTT_HEADER_SIZE {
            return Err(BridgeError::BufferOverflow {
                expected: QTT_HEADER_SIZE,
                actual: mmap.len(),
            });
        }
        
        let header: QTTBridgeHeader = *bytemuck::from_bytes(
            &mmap[..std::mem::size_of::<QTTBridgeHeader>()]
        );
        
        header.validate()?;
        header.validate_doctrine()?;
        
        Ok(QTTFrame { mmap, header })
    }
    
    /// Reset frame tracking (use after Python restart)
    pub fn reset(&mut self) {
        self.last_frame = 0;
    }
}

impl Default for QTTReader {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming Support for Large QTT (>1GB)
// ─────────────────────────────────────────────────────────────────────────────

/// Chunk size for streaming large QTT payloads (64 MB)
pub const STREAM_CHUNK_SIZE: usize = 64 * 1024 * 1024;

/// Maximum recommended QTT size for non-streaming access (1 GB)
pub const MAX_NON_STREAMING_SIZE: usize = 1024 * 1024 * 1024;

/// Streaming iterator over TT-cores for large QTT payloads.
///
/// For payloads larger than 1GB, this provides memory-efficient
/// iteration over cores without loading all data at once.
///
/// # Example
///
/// ```rust,ignore
/// let reader = QTTReader::new();
/// let frame = reader.read_current()?;
///
/// // Stream cores to GPU
/// for chunk in QTTStreamingIterator::new(&frame)? {
///     let (core_idx, data) = chunk?;
///     gpu.upload_core(core_idx, data);
/// }
/// ```
pub struct QTTStreamingIterator<'a> {
    /// Reference to QTT frame
    frame: &'a QTTFrame,
    /// Current core index
    current_core: usize,
    /// Total cores to iterate
    total_cores: usize,
}

impl<'a> QTTStreamingIterator<'a> {
    /// Create a new streaming iterator over QTT cores.
    pub fn new(frame: &'a QTTFrame) -> Self {
        Self {
            frame,
            current_core: 0,
            total_cores: frame.header.num_sites as usize,
        }
    }
    
    /// Get total size of all cores in bytes.
    pub fn total_bytes(&self) -> usize {
        self.frame.header.total_data_bytes as usize
    }
    
    /// Check if streaming is recommended for this payload size.
    pub fn should_stream(&self) -> bool {
        self.total_bytes() > MAX_NON_STREAMING_SIZE
    }
    
    /// Get progress as fraction (0.0 - 1.0)
    pub fn progress(&self) -> f32 {
        if self.total_cores == 0 {
            return 1.0;
        }
        self.current_core as f32 / self.total_cores as f32
    }
    
    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.current_core = 0;
    }
}

/// A single TT-core chunk for streaming upload.
#[derive(Debug)]
pub struct CoreChunk<'a> {
    /// Core index (0 to L-1)
    pub core_index: usize,
    /// Left bond dimension
    pub chi_left: usize,
    /// Right bond dimension
    pub chi_right: usize,
    /// Physical dimension
    pub physical_dim: usize,
    /// Raw byte data
    pub data: &'a [u8],
    /// Byte offset from start of data section
    pub offset: usize,
}

impl<'a> CoreChunk<'a> {
    /// Get core data as f32 slice (if dtype is Float32).
    ///
    /// Returns None if dtype is not Float32 or data is misaligned.
    pub fn as_f32(&self) -> Option<&[f32]> {
        let ptr = self.data.as_ptr();
        
        // Check alignment
        if ptr as usize % std::mem::align_of::<f32>() != 0 {
            return None;
        }
        
        let count = self.chi_left * self.physical_dim * self.chi_right;
        if self.data.len() < count * 4 {
            return None;
        }
        
        Some(unsafe { std::slice::from_raw_parts(ptr as *const f32, count) })
    }
    
    /// Get core shape as (χ_left, d, χ_right).
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.chi_left, self.physical_dim, self.chi_right)
    }
    
    /// Get number of elements in this core.
    pub fn num_elements(&self) -> usize {
        self.chi_left * self.physical_dim * self.chi_right
    }
}

impl<'a> Iterator for QTTStreamingIterator<'a> {
    type Item = Result<CoreChunk<'a>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_core >= self.total_cores {
            return None;
        }
        
        let idx = self.current_core;
        self.current_core += 1;
        
        let data = match self.frame.core_bytes(idx) {
            Some(d) => d,
            None => return Some(Err(BridgeError::BufferOverflow {
                expected: self.frame.header.core_size(idx),
                actual: 0,
            })),
        };
        
        Some(Ok(CoreChunk {
            core_index: idx,
            chi_left: self.frame.header.chi_left(idx),
            chi_right: self.frame.header.chi_right(idx),
            physical_dim: self.frame.header.physical_dim as usize,
            data,
            offset: self.frame.header.core_offsets[idx] as usize,
        }))
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_cores - self.current_core;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for QTTStreamingIterator<'a> {}

/// Batch streaming iterator for efficient GPU upload.
///
/// Groups multiple small cores into larger batches to minimize
/// GPU upload overhead.
pub struct QTTBatchIterator<'a> {
    /// Reference to QTT frame
    frame: &'a QTTFrame,
    /// Current core index
    current_core: usize,
    /// Total cores
    total_cores: usize,
    /// Target batch size in bytes
    target_batch_bytes: usize,
}

impl<'a> QTTBatchIterator<'a> {
    /// Create a new batch iterator with default batch size (64 MB).
    pub fn new(frame: &'a QTTFrame) -> Self {
        Self::with_batch_size(frame, STREAM_CHUNK_SIZE)
    }
    
    /// Create a new batch iterator with custom batch size.
    pub fn with_batch_size(frame: &'a QTTFrame, batch_bytes: usize) -> Self {
        Self {
            frame,
            current_core: 0,
            total_cores: frame.header.num_sites as usize,
            target_batch_bytes: batch_bytes,
        }
    }
}

/// A batch of cores for bulk GPU upload.
#[derive(Debug)]
pub struct CoreBatch {
    /// Start core index (inclusive)
    pub start_core: usize,
    /// End core index (exclusive)
    pub end_core: usize,
    /// Total bytes in this batch
    pub total_bytes: usize,
    /// Core offsets relative to batch start (for GPU buffer binding)
    pub offsets: Vec<usize>,
}

impl<'a> Iterator for QTTBatchIterator<'a> {
    type Item = CoreBatch;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_core >= self.total_cores {
            return None;
        }
        
        let start = self.current_core;
        let mut bytes = 0usize;
        let mut offsets = Vec::new();
        
        // Collect cores until we hit batch size
        while self.current_core < self.total_cores {
            let core_size = self.frame.header.core_size(self.current_core);
            
            // Always include at least one core
            if bytes > 0 && bytes + core_size > self.target_batch_bytes {
                break;
            }
            
            offsets.push(bytes);
            bytes += core_size;
            self.current_core += 1;
        }
        
        Some(CoreBatch {
            start_core: start,
            end_core: self.current_core,
            total_bytes: bytes,
            offsets,
        })
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
            std::mem::size_of::<QTTBridgeHeader>(),
            QTT_HEADER_SIZE,
            "QTTBridgeHeader size mismatch"
        );
        assert_eq!(QTT_HEADER_SIZE, 512);
        assert!(QTT_HEADER_SIZE.is_power_of_two());
    }
    
    #[test]
    fn test_header_alignment() {
        assert_eq!(
            std::mem::align_of::<QTTBridgeHeader>(),
            512,
            "QTTBridgeHeader must be 512-byte aligned"
        );
    }
    
    #[test]
    fn test_header_default() {
        let header = QTTBridgeHeader::default();
        assert_eq!(header.magic, QTT_BRIDGE_MAGIC);
        assert_eq!(header.version, QTT_BRIDGE_VERSION);
        assert_eq!(header.physical_dim, 2);
    }
    
    #[test]
    fn test_header_validate() {
        let header = QTTBridgeHeader::default();
        assert!(header.validate().is_ok());
        
        let mut bad_magic = header;
        bad_magic.magic = [0, 0, 0, 0];
        assert!(bad_magic.validate().is_err());
    }
    
    #[test]
    fn test_doctrine_validation() {
        let mut header = QTTBridgeHeader::default();
        header.compression_ratio = 2.0;
        header.truncation_error = 1e-8;
        assert!(header.validate_doctrine().is_ok());
        
        // Low compression ratio should fail
        let mut low_ratio = header;
        low_ratio.compression_ratio = 1.0;
        assert!(low_ratio.validate_doctrine().is_err());
        
        // High truncation error should fail
        let mut high_error = header;
        high_error.compression_ratio = 2.0;
        high_error.truncation_error = 1e-3;
        assert!(high_error.validate_doctrine().is_err());
    }
    
    #[test]
    fn test_bond_dimensions() {
        let mut header = QTTBridgeHeader::default();
        header.num_sites = 4;
        header.physical_dim = 2;
        header.bond_dims[0] = 8;  // χ between sites 0-1
        header.bond_dims[1] = 16; // χ between sites 1-2
        header.bond_dims[2] = 8;  // χ between sites 2-3
        
        // Site 0: χ_left=1, χ_right=8
        assert_eq!(header.chi_left(0), 1);
        assert_eq!(header.chi_right(0), 8);
        
        // Site 1: χ_left=8, χ_right=16
        assert_eq!(header.chi_left(1), 8);
        assert_eq!(header.chi_right(1), 16);
        
        // Site 2: χ_left=16, χ_right=8
        assert_eq!(header.chi_left(2), 16);
        assert_eq!(header.chi_right(2), 8);
        
        // Site 3 (last): χ_left=8, χ_right=1
        assert_eq!(header.chi_left(3), 8);
        assert_eq!(header.chi_right(3), 1);
    }
    
    #[test]
    fn test_core_size_calculation() {
        let mut header = QTTBridgeHeader::default();
        header.num_sites = 3;
        header.physical_dim = 2;
        header.dtype = QTTDataType::Float32 as u8;
        header.bond_dims[0] = 4;  // χ between sites 0-1
        header.bond_dims[1] = 4;  // χ between sites 1-2
        
        // Site 0: (1, 2, 4) = 8 elements × 4 bytes = 32 bytes
        assert_eq!(header.core_size(0), 32);
        assert_eq!(header.core_elements(0), 8);
        
        // Site 1: (4, 2, 4) = 32 elements × 4 bytes = 128 bytes
        assert_eq!(header.core_size(1), 128);
        assert_eq!(header.core_elements(1), 32);
        
        // Site 2: (4, 2, 1) = 8 elements × 4 bytes = 32 bytes
        assert_eq!(header.core_size(2), 32);
        assert_eq!(header.core_elements(2), 8);
    }
    
    #[test]
    fn test_reader_not_available() {
        let reader = QTTReader::with_path("/tmp/nonexistent_qtt");
        assert!(!reader.is_available());
    }
    
    #[test]
    fn test_data_type_sizes() {
        assert_eq!(QTTDataType::Float32.element_size(), 4);
        assert_eq!(QTTDataType::Float64.element_size(), 8);
        assert_eq!(QTTDataType::Float16.element_size(), 2);
    }
    
    #[test]
    fn test_crc32_checksum() {
        // Test that CRC32 produces consistent results
        let data = b"Hello, QTT Bridge!";
        let checksum1 = QTTBridgeHeader::compute_checksum(data);
        let checksum2 = QTTBridgeHeader::compute_checksum(data);
        assert_eq!(checksum1, checksum2, "CRC32 should be deterministic");
        
        // Different data should produce different checksum
        let data2 = b"Hello, QTT Bridge?";
        let checksum3 = QTTBridgeHeader::compute_checksum(data2);
        assert_ne!(checksum1, checksum3, "Different data should have different CRC32");
        
        // Known value test (CRC32-ISO-HDLC)
        let known_data = b"123456789";
        let known_checksum = QTTBridgeHeader::compute_checksum(known_data);
        assert_eq!(known_checksum, 0xCBF43926, "CRC32-ISO-HDLC known value");
    }
}
