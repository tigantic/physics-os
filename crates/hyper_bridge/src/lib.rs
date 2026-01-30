//! HyperBridge: RAM Bridge IPC Protocol for Python ↔ Rust Tensor Streaming
//!
//! This crate provides zero-copy shared memory communication between:
//! - **Python (TensorNet)**: Physics simulation, QTT compression, heatmap generation
//! - **Rust (Glass Cockpit/Global Eye)**: Real-time visualization at 60+ FPS
//!
//! # Protocol Overview
//!
//! The RAM Bridge uses a memory-mapped file (`/dev/shm/hypertensor_bridge`) with:
//! - **Header (4KB)**: Metadata, frame counter, tensor statistics, synchronization
//! - **Data Buffer (8MB+)**: Raw tensor data (RGBA8, grayscale, or raw float)
//!
//! # Usage
//!
//! ```rust,ignore
//! use hyper_bridge::{RamBridgeReader, TensorBridgeHeader};
//!
//! let mut bridge = RamBridgeReader::connect("/dev/shm/hypertensor_bridge")?;
//!
//! loop {
//!     if let Some((header, data)) = bridge.read_frame()? {
//!         println!("Frame {}: {}x{}", header.frame_number, header.width, header.height);
//!         // Process tensor data...
//!     }
//! }
//! ```
//!
//! # Thread Safety
//!
//! The bridge is **not** thread-safe by design. Each consumer should have its own
//! `RamBridgeReader` instance, or use external synchronization.

mod protocol;
mod reader;
mod writer;
mod sovereign;
mod sovereign_v2;
mod weather;
pub mod trajectory;
pub mod swarm;
pub mod qtt;
pub mod bench_real_data;

#[cfg(test)]
mod tests;

pub use protocol::{
    TensorBridgeHeader,
    TENSOR_BRIDGE_MAGIC,
    TENSOR_BRIDGE_VERSION,
    HEADER_SIZE,
    DEFAULT_DATA_SIZE,
};

pub use reader::RamBridgeReader;
pub use writer::RamBridgeWriter;

// Legacy v1 SovereignBridge (deprecated, use sovereign_v2 for new code)
#[deprecated(since = "0.2.0", note = "Use SovereignBridgeV2 for new code")]
pub use sovereign::SovereignBridge;

// ─────────────────────────────────────────────────────────────────────────────
// Sovereign Bridge V2 (Refactored with bytemuck structs)
// ─────────────────────────────────────────────────────────────────────────────

pub use sovereign_v2::{
    SovereignBridge as SovereignBridgeV2,
    SovereignHeader,
    Telemetry,
    SOVEREIGN_MAGIC,
    SOVEREIGN_VERSION,
    SOVEREIGN_HEADER_SIZE,
    SOVEREIGN_TELEMETRY_SIZE,
    SOVEREIGN_SHM_PATH,
};

// ─────────────────────────────────────────────────────────────────────────────
// Trajectory Guidance (Phase 3 - Hypersonic Solver)
// ─────────────────────────────────────────────────────────────────────────────

pub use trajectory::{
    TrajectoryHeader,
    TrajectoryData,
    Waypoint,
    TRAJECTORY_MAGIC,
    TRAJECTORY_VERSION,
    TRAJECTORY_HEADER_SIZE,
    MAX_WAYPOINTS,
};

// ─────────────────────────────────────────────────────────────────────────────
// Weather Bridge (Global Eye)
// ─────────────────────────────────────────────────────────────────────────────

pub use weather::{
    WeatherHeader,
    WeatherFrame,
    WeatherReader,
    WEATHER_MAGIC,
    WEATHER_VERSION,
    WEATHER_SHM_PATH,
    WEATHER_HEADER_SIZE,
};

// ─────────────────────────────────────────────────────────────────────────────
// QTT Bridge (Native Tensor Train - NO DECOMPRESSION)
// ─────────────────────────────────────────────────────────────────────────────

pub use qtt::{
    QTTBridgeHeader,
    QTTFrame,
    QTTReader,
    QTTDataType,
    // Streaming support for large QTT payloads (>1GB)
    QTTStreamingIterator,
    QTTBatchIterator,
    CoreChunk,
    CoreBatch,
    QTT_BRIDGE_MAGIC,
    QTT_BRIDGE_VERSION,
    QTT_SHM_PATH,
    QTT_HEADER_SIZE,
    MAX_QTT_SITES,
    MIN_COMPRESSION_RATIO,
    MAX_TRUNCATION_ERROR,
    STREAM_CHUNK_SIZE,
    MAX_NON_STREAMING_SIZE,
};

// ─────────────────────────────────────────────────────────────────────────────
// Backward Compatibility Aliases
// ─────────────────────────────────────────────────────────────────────────────

/// Protocol version for external consumers
pub const PROTOCOL_VERSION: u32 = TENSOR_BRIDGE_VERSION;

/// Legacy alias: RamBridgeV2 → RamBridgeReader
///
/// This alias exists for backward compatibility with code that used the
/// original `ram_bridge_v2` module. New code should use `RamBridgeReader`.
pub type RamBridgeV2 = RamBridgeReader;

/// Error types for RAM bridge operations
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Bridge not available: {0}")]
    NotAvailable(String),
    
    #[error("Invalid magic number: expected {expected:?}, got {actual:?}")]
    InvalidMagic { expected: [u8; 4], actual: [u8; 4] },
    
    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Buffer overflow: expected {expected} bytes, got {actual}")]
    BufferOverflow { expected: usize, actual: usize },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, BridgeError>;
