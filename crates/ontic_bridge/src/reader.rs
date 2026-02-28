//! RAM Bridge Reader: Consume tensor data from shared memory

use std::path::Path;
use std::fs::OpenOptions;
use memmap2::MmapMut;
use bytemuck;

use crate::protocol::TensorBridgeHeader;
use crate::{BridgeError, Result};

/// RAM Bridge Reader
///
/// Reads tensor data from a memory-mapped file written by Python.
/// Provides frame synchronization and statistics access.
pub struct RamBridgeReader {
    /// Memory-mapped file handle
    mmap: MmapMut,
    
    /// Last frame number successfully read
    last_frame_number: u64,
    
    /// Frame drop counter (for diagnostics)
    frame_drops: u64,
}

impl RamBridgeReader {
    /// Connect to an existing RAM bridge
    ///
    /// # Arguments
    /// * `path` - Path to shared memory file (e.g., "/dev/shm/ontic_bridge")
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully connected
    /// * `Err(BridgeError)` - Bridge not available or invalid
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())
            .map_err(|e| BridgeError::NotAvailable(e.to_string()))?;
        
        let mmap = unsafe { 
            MmapMut::map_mut(&file)
                .map_err(|e| BridgeError::NotAvailable(e.to_string()))?
        };
        
        // Validate header on connect
        let header = Self::read_header_from_mmap(&mmap)?;
        header.validate()?;
        
        Ok(Self {
            mmap,
            last_frame_number: header.frame_number.saturating_sub(1),
            frame_drops: 0,
        })
    }
    
    /// Read header from memory-mapped buffer
    fn read_header_from_mmap(mmap: &[u8]) -> Result<TensorBridgeHeader> {
        if mmap.len() < std::mem::size_of::<TensorBridgeHeader>() {
            return Err(BridgeError::BufferOverflow {
                expected: std::mem::size_of::<TensorBridgeHeader>(),
                actual: mmap.len(),
            });
        }
        
        let header_bytes = &mmap[..std::mem::size_of::<TensorBridgeHeader>()];
        let header: TensorBridgeHeader = *bytemuck::from_bytes(header_bytes);
        
        Ok(header)
    }
    
    /// Get current header without consuming a frame
    pub fn peek_header(&self) -> Result<TensorBridgeHeader> {
        Self::read_header_from_mmap(&self.mmap)
    }
    
    /// Read current frame data
    ///
    /// # Returns
    /// * `Ok(Some((header, data)))` - New frame available
    /// * `Ok(None)` - No new frame (same frame number as last read)
    /// * `Err(BridgeError)` - Invalid data or protocol error
    pub fn read_frame(&mut self) -> Result<Option<(TensorBridgeHeader, Vec<u8>)>> {
        let header = Self::read_header_from_mmap(&self.mmap)?;
        
        // Check if this is a new frame
        if header.frame_number <= self.last_frame_number {
            return Ok(None);
        }
        
        // Detect frame drops
        let expected_frame = self.last_frame_number + 1;
        if header.frame_number > expected_frame {
            let drops = header.frame_number - expected_frame;
            self.frame_drops += drops;
            #[cfg(debug_assertions)]
            eprintln!(
                "⚠ Frame drop: expected {}, got {} (dropped {})",
                expected_frame, header.frame_number, drops
            );
        }
        
        self.last_frame_number = header.frame_number;
        
        // Read pixel data
        let data_start = header.data_offset as usize;
        let data_end = data_start + header.data_size as usize;
        
        if data_end > self.mmap.len() {
            return Err(BridgeError::BufferOverflow {
                expected: data_end,
                actual: self.mmap.len(),
            });
        }
        
        let data = self.mmap[data_start..data_end].to_vec();
        
        Ok(Some((header, data)))
    }
    
    /// Read frame data without copying (returns slice reference)
    ///
    /// # Safety
    /// The returned slice is only valid until the next Python write.
    /// Use this for zero-copy GPU upload.
    pub fn read_frame_zero_copy(&mut self) -> Result<Option<(TensorBridgeHeader, &[u8])>> {
        let header = Self::read_header_from_mmap(&self.mmap)?;
        
        if header.frame_number <= self.last_frame_number {
            return Ok(None);
        }
        
        let expected_frame = self.last_frame_number + 1;
        if header.frame_number > expected_frame {
            self.frame_drops += header.frame_number - expected_frame;
        }
        
        self.last_frame_number = header.frame_number;
        
        let data_start = header.data_offset as usize;
        let data_end = data_start + header.data_size as usize;
        
        if data_end > self.mmap.len() {
            return Err(BridgeError::BufferOverflow {
                expected: data_end,
                actual: self.mmap.len(),
            });
        }
        
        Ok(Some((header, &self.mmap[data_start..data_end])))
    }
    
    /// Get frame drop statistics
    pub fn frame_drops(&self) -> u64 {
        self.frame_drops
    }
    
    /// Get last successfully read frame number
    pub fn last_frame_number(&self) -> u64 {
        self.last_frame_number
    }
    
    /// Reset frame tracking (use after Python restart)
    pub fn reset(&mut self) {
        self.last_frame_number = 0;
        self.frame_drops = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_connect_missing_file() {
        let result = RamBridgeReader::connect("/tmp/nonexistent_ontic_bridge");
        assert!(result.is_err());
    }
}
