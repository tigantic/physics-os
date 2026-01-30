//! Block-SVD Reconstruction: High-speed block reconstruction from U, S, Vh
//!
//! This module provides O(1) per-query block reconstruction by:
//! 1. Pre-computing cumulative pointer arrays in Rust (eliminates Python loops)
//! 2. Parallel block reconstruction using Rayon
//! 3. Direct memory access via numpy arrays (zero-copy)
//!
//! # Performance Target
//! - Single block query: < 100μs
//! - Full frame reconstruction: < 10ms
//! - Batched frame reconstruction: < 1ms per frame

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s, Axis};
use rayon::prelude::*;

/// Pre-computed block pointers for O(1) lookup
pub struct BlockPointers {
    /// Cumulative sum for U array indexing
    pub cumsum_u: Vec<u64>,
    /// Cumulative sum for S array indexing
    pub cumsum_s: Vec<u64>,
    /// Cumulative sum for Vh array indexing (same as U for square blocks)
    pub cumsum_vh: Vec<u64>,
    /// Rank per block
    pub ranks: Vec<u16>,
    /// Block dimensions
    pub block_size: usize,
    /// Number of blocks per frame
    pub blocks_per_frame: usize,
    /// Frame dimensions (H, W)
    pub frame_shape: (usize, usize),
}

impl BlockPointers {
    /// Build pointer arrays from ranks array
    /// This is the O(N) operation that happens ONCE at load time
    pub fn new(ranks: &[u16], block_size: usize, frame_h: usize, frame_w: usize) -> Self {
        let n_blocks = ranks.len();
        let n_blocks_h = (frame_h + block_size - 1) / block_size;
        let n_blocks_w = (frame_w + block_size - 1) / block_size;
        let blocks_per_frame = n_blocks_h * n_blocks_w;
        
        // Pre-compute cumulative sums
        let mut cumsum_u = Vec::with_capacity(n_blocks + 1);
        let mut cumsum_s = Vec::with_capacity(n_blocks + 1);
        
        cumsum_u.push(0);
        cumsum_s.push(0);
        
        let mut u_ptr: u64 = 0;
        let mut s_ptr: u64 = 0;
        
        for &r in ranks {
            let r = r as u64;
            u_ptr += r * block_size as u64;
            s_ptr += r;
            cumsum_u.push(u_ptr);
            cumsum_s.push(s_ptr);
        }
        
        Self {
            cumsum_u: cumsum_u.clone(),
            cumsum_s: cumsum_s.clone(),
            cumsum_vh: cumsum_u, // Same for Vh
            ranks: ranks.to_vec(),
            block_size,
            blocks_per_frame,
            frame_shape: (frame_h, frame_w),
        }
    }

    /// Get pointers for a specific block (O(1) operation)
    #[inline]
    pub fn get_block_pointers(&self, block_idx: usize) -> BlockSlice {
        let rank = self.ranks[block_idx] as usize;
        BlockSlice {
            u_start: self.cumsum_u[block_idx] as usize,
            u_end: self.cumsum_u[block_idx + 1] as usize,
            s_start: self.cumsum_s[block_idx] as usize,
            s_end: self.cumsum_s[block_idx + 1] as usize,
            vh_start: self.cumsum_vh[block_idx] as usize,
            vh_end: self.cumsum_vh[block_idx + 1] as usize,
            rank,
        }
    }

    /// Get block position in frame
    #[inline]
    pub fn block_position(&self, block_idx: usize) -> (usize, usize, usize, usize) {
        let blocks_per_row = (self.frame_shape.1 + self.block_size - 1) / self.block_size;
        let bh = block_idx / blocks_per_row;
        let bw = block_idx % blocks_per_row;
        
        let y_start = bh * self.block_size;
        let x_start = bw * self.block_size;
        let y_end = (y_start + self.block_size).min(self.frame_shape.0);
        let x_end = (x_start + self.block_size).min(self.frame_shape.1);
        
        (y_start, y_end, x_start, x_end)
    }
}

/// Slice indices for a single block
pub struct BlockSlice {
    pub u_start: usize,
    pub u_end: usize,
    pub s_start: usize,
    pub s_end: usize,
    pub vh_start: usize,
    pub vh_end: usize,
    pub rank: usize,
}

/// Reconstruct a single block from U, S, Vh components
/// Returns block_size x block_size array
/// 
/// Uses optimized matrix multiply: (U @ diag(S)) @ Vh
/// Complexity: O(block_size * rank * block_size) with SIMD vectorization
pub fn reconstruct_block(
    u_data: &[f32],
    s_data: &[f32],
    vh_data: &[f32],
    slice: &BlockSlice,
    block_size: usize,
) -> Array2<f32> {
    if slice.rank == 0 {
        return Array2::zeros((block_size, block_size));
    }
    
    let rank = slice.rank;
    
    // Extract components
    let u_slice = &u_data[slice.u_start..slice.u_end];
    let s_slice = &s_data[slice.s_start..slice.s_end];
    let vh_slice = &vh_data[slice.vh_start..slice.vh_end];
    
    // Create ndarray views
    let u_arr = ArrayView2::from_shape((block_size, rank), u_slice).unwrap();
    let vh_arr = ArrayView2::from_shape((rank, block_size), vh_slice).unwrap();
    
    // U @ diag(S) - apply S scaling row-wise to U
    // Then (U * S) @ Vh via ndarray's optimized dot product
    let mut u_scaled = u_arr.to_owned();
    for (i, mut row) in u_scaled.axis_iter_mut(Axis(0)).enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            *val *= s_slice[j];
        }
    }
    
    // Use ndarray's optimized matrix multiply (uses BLAS if available)
    u_scaled.dot(&vh_arr)
}

/// Reconstruct a full frame in parallel
pub fn reconstruct_frame_parallel(
    u_data: &[f32],
    s_data: &[f32],
    vh_data: &[f32],
    pointers: &BlockPointers,
    frame_idx: usize,
) -> Array2<f32> {
    let (h, w) = pointers.frame_shape;
    let block_size = pointers.block_size;
    let n_blocks_w = (w + block_size - 1) / block_size;
    let n_blocks_h = (h + block_size - 1) / block_size;
    let n_blocks = n_blocks_h * n_blocks_w;
    
    let block_offset = frame_idx * pointers.blocks_per_frame;
    
    // Parallel block reconstruction
    let blocks: Vec<(usize, Array2<f32>)> = (0..n_blocks)
        .into_par_iter()
        .map(|bi| {
            let global_bi = block_offset + bi;
            let slice = pointers.get_block_pointers(global_bi);
            let block = reconstruct_block(u_data, s_data, vh_data, &slice, block_size);
            (bi, block)
        })
        .collect();
    
    // Assemble frame
    let mut frame = Array2::zeros((h, w));
    
    for (bi, block) in blocks {
        let (y_start, y_end, x_start, x_end) = pointers.block_position(bi);
        let block_h = y_end - y_start;
        let block_w = x_end - x_start;
        
        for i in 0..block_h {
            for j in 0..block_w {
                frame[[y_start + i, x_start + j]] = block[[i, j]];
            }
        }
    }
    
    frame
}

/// Batch reconstruct multiple frames
pub fn reconstruct_batch_parallel(
    u_data: &[f32],
    s_data: &[f32],
    vh_data: &[f32],
    pointers: &BlockPointers,
    frame_indices: &[usize],
) -> Vec<Array2<f32>> {
    frame_indices
        .par_iter()
        .map(|&fi| reconstruct_frame_parallel(u_data, s_data, vh_data, pointers, fi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_construction() {
        let ranks = vec![4u16, 8, 4, 8, 16, 4, 8, 4, 8];
        let pointers = BlockPointers::new(&ranks, 64, 192, 192);
        
        assert_eq!(pointers.blocks_per_frame, 9);
        assert_eq!(pointers.cumsum_u.len(), 10);
        
        // First block: rank 4
        let slice = pointers.get_block_pointers(0);
        assert_eq!(slice.rank, 4);
        assert_eq!(slice.u_start, 0);
        assert_eq!(slice.u_end, 4 * 64);
        
        // Second block: rank 8
        let slice = pointers.get_block_pointers(1);
        assert_eq!(slice.rank, 8);
        assert_eq!(slice.u_start, 4 * 64);
    }
    
    #[test]
    fn test_block_position() {
        let ranks = vec![4u16; 9];
        let pointers = BlockPointers::new(&ranks, 64, 192, 192);
        
        // Block 0: top-left
        let (y0, y1, x0, x1) = pointers.block_position(0);
        assert_eq!((y0, y1, x0, x1), (0, 64, 0, 64));
        
        // Block 1: top-middle
        let (y0, y1, x0, x1) = pointers.block_position(1);
        assert_eq!((y0, y1, x0, x1), (0, 64, 64, 128));
        
        // Block 3: middle-left
        let (y0, y1, x0, x1) = pointers.block_position(3);
        assert_eq!((y0, y1, x0, x1), (64, 128, 0, 64));
    }
}
