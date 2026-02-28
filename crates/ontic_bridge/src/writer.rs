//! RAM Bridge Writer: Produce tensor data to shared memory
//!
//! This is primarily used for testing from Rust.
//! In production, Python writes to the bridge.

use std::path::Path;
use std::fs::OpenOptions;
use std::time::{SystemTime, UNIX_EPOCH};
use memmap2::MmapMut;
use bytemuck;

use crate::protocol::{TensorBridgeHeader, HEADER_SIZE};
use crate::{BridgeError, Result};

/// RAM Bridge Writer
///
/// Writes tensor data to a memory-mapped file for consumers.
/// Primarily used for testing; production writes come from Python.
pub struct RamBridgeWriter {
    /// Memory-mapped file handle
    mmap: MmapMut,
    
    /// Current frame number
    frame_number: u64,
}

impl RamBridgeWriter {
    /// Create a new RAM bridge file
    ///
    /// # Arguments
    /// * `path` - Path to create shared memory file
    /// * `width` - Tensor width
    /// * `height` - Tensor height
    /// * `channels` - Number of channels (1, 3, or 4)
    pub fn create<P: AsRef<Path>>(
        path: P,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Self> {
        let data_size = (width * height * channels) as usize;
        let total_size = HEADER_SIZE + data_size;
        
        // Create or truncate file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())
            .map_err(|e| BridgeError::NotAvailable(e.to_string()))?;
        
        file.set_len(total_size as u64)
            .map_err(|e| BridgeError::Io(e))?;
        
        let mut mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| BridgeError::NotAvailable(e.to_string()))?
        };
        
        // Initialize header
        let header = TensorBridgeHeader {
            width,
            height,
            channels,
            data_size: data_size as u32,
            ..Default::default()
        };
        
        let header_bytes: &[u8] = bytemuck::bytes_of(&header);
        mmap[..header_bytes.len()].copy_from_slice(header_bytes);
        
        Ok(Self {
            mmap,
            frame_number: 0,
        })
    }
    
    /// Write a new frame
    ///
    /// # Arguments
    /// * `data` - Raw tensor data (must match width × height × channels)
    /// * `stats` - Optional (min, max, mean, std) statistics
    pub fn write_frame(
        &mut self,
        data: &[u8],
        stats: Option<(f32, f32, f32, f32)>,
    ) -> Result<()> {
        self.frame_number += 1;
        
        // Update header
        let mut header: TensorBridgeHeader = *bytemuck::from_bytes(
            &self.mmap[..std::mem::size_of::<TensorBridgeHeader>()]
        );
        
        if data.len() != header.data_size as usize {
            return Err(BridgeError::BufferOverflow {
                expected: header.data_size as usize,
                actual: data.len(),
            });
        }
        
        header.frame_number = self.frame_number;
        header.producer_timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        
        if let Some((min, max, mean, std)) = stats {
            header.tensor_min = min;
            header.tensor_max = max;
            header.tensor_mean = mean;
            header.tensor_std = std;
        }
        
        // Write header
        let header_bytes: &[u8] = bytemuck::bytes_of(&header);
        self.mmap[..header_bytes.len()].copy_from_slice(header_bytes);
        
        // Write data
        let data_start = header.data_offset as usize;
        self.mmap[data_start..data_start + data.len()].copy_from_slice(data);
        
        // Flush to ensure visibility
        self.mmap.flush()?;
        
        Ok(())
    }
    
    /// Get current frame number
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RamBridgeReader;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_write_read_roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();
        
        // Create writer
        let mut writer = RamBridgeWriter::create(path, 64, 64, 4).unwrap();
        
        // Write frame
        let data = vec![128u8; 64 * 64 * 4];
        writer.write_frame(&data, Some((0.0, 1.0, 0.5, 0.1))).unwrap();
        
        // Read back
        let mut reader = RamBridgeReader::connect(path).unwrap();
        let (header, read_data) = reader.read_frame().unwrap().unwrap();
        
        assert_eq!(header.frame_number, 1);
        assert_eq!(header.width, 64);
        assert_eq!(header.height, 64);
        assert_eq!(read_data, data);
    }
}
