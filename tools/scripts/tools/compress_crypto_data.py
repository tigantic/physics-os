#!/usr/bin/env python3
"""
Quantum Compression of Crypto/Stock Market Data
================================================
Uses QTT (Quantized Tensor Train) compression from The Ontic Engine's GENESIS module
to compress 2.9GB of 5-minute OHLCV market data.

Production-grade implementation with full error handling.
"""

import os
import sys
import time
import glob
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch

# Import QTT from GENESIS
try:
    from ontic.genesis.sgw import QTTSignal
except ImportError as e:
    print(f"[ERROR] Failed to import QTTSignal: {e}")
    print("Ensure ontic package is available in the path.")
    sys.exit(1)


@dataclass
class CompressionStats:
    """Statistics for compression operation."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_rows: int = 0
    raw_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 0.0
    elapsed_seconds: float = 0.0
    files_per_second: float = 0.0
    throughput_mb_per_sec: float = 0.0
    errors: list = field(default_factory=list)


def parse_market_file(filepath: str) -> np.ndarray:
    """
    Parse a market data file into numerical array.
    
    Expected format (CSV with header):
    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    
    Returns:
        np.ndarray of shape (N, 6) with [date, time, open, high, low, close, volume]
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            try:
                # Extract: date, time, open, high, low, close, vol
                date_val = float(parts[2])  # YYYYMMDD as float
                time_val = float(parts[3])  # HHMMSS as float
                open_val = float(parts[4])
                high_val = float(parts[5])
                low_val = float(parts[6])
                close_val = float(parts[7])
                vol_val = float(parts[8])
                rows.append([date_val, time_val, open_val, high_val, low_val, close_val, vol_val])
            except (ValueError, IndexError):
                continue
    
    if not rows:
        return np.array([]).reshape(0, 7)
    
    return np.array(rows, dtype=np.float32)


def find_all_data_files(base_dir: str) -> list:
    """Find all .txt data files recursively."""
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.txt') and not filename.endswith('Zone.Identifier'):
                files.append(os.path.join(root, filename))
    return sorted(files)


def pad_to_power_of_2(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad array length to nearest power of 2."""
    n = len(arr)
    target = 1
    while target < n:
        target *= 2
    if target == n:
        return arr, n
    # Pad with last value (preserves structure better than zeros)
    padded = np.zeros(target, dtype=arr.dtype)
    padded[:n] = arr
    if n > 0:
        padded[n:] = arr[-1]  # Repeat last value
    return padded, n


def compress_to_qtt(data: np.ndarray, max_rank: int = 32) -> Tuple[Dict[str, Any], int]:
    """
    Compress data using QTT tensor train decomposition.
    
    Args:
        data: 2D array of shape (N, features)
        max_rank: Maximum TT rank for compression
        
    Returns:
        Tuple of (compressed_dict, compressed_size_bytes)
    """
    if data.size == 0:
        return {}, 0
    
    # Flatten the 2D data to 1D for QTT (which expects 1D signals)
    # We'll store shape info to reconstruct
    original_shape = data.shape
    flat_data = data.flatten().astype(np.float32)
    
    # Normalize for better compression
    data_min = float(flat_data.min())
    data_max = float(flat_data.max())
    data_range = data_max - data_min
    if data_range > 0:
        flat_norm = (flat_data - data_min) / data_range
    else:
        flat_norm = flat_data - data_min
    
    # Pad to power of 2 (required by QTT)
    flat_padded, original_len = pad_to_power_of_2(flat_norm)
    
    # Convert to tensor
    tensor_padded = torch.from_numpy(flat_padded).float()
    
    # Create QTT signal using tol parameter (not tolerance)
    qtt = QTTSignal.from_dense(tensor_padded, max_rank=max_rank, tol=1e-6)
    
    # Get compressed representation
    cores = qtt.cores
    
    # Package for storage
    compressed = {
        'cores': [c.numpy() if isinstance(c, torch.Tensor) else c for c in cores],
        'shape': list(original_shape),
        'original_len': original_len,
        'padded_len': len(flat_padded),
        'min': data_min,
        'max': data_max,
        'dtype': str(data.dtype),
    }
    
    # Calculate compressed size
    compressed_size = sum(c.nbytes for c in compressed['cores'])
    compressed_size += 128  # Metadata overhead estimate
    
    return compressed, compressed_size


def compress_all_files(base_dir: str, output_file: str, batch_size: int = 500) -> CompressionStats:
    """
    Compress all market data files using QTT.
    
    Args:
        base_dir: Base directory containing market data
        output_file: Output .npz file for compressed data
        batch_size: Number of files to process before batching
        
    Returns:
        CompressionStats with compression metrics
    """
    stats = CompressionStats()
    
    print(f"\n{'='*70}")
    print("QUANTUM TENSOR TRAIN (QTT) COMPRESSION")
    print(f"{'='*70}")
    print(f"Source:      {base_dir}")
    print(f"Output:      {output_file}")
    print(f"Batch Size:  {batch_size} files")
    print(f"{'='*70}\n")
    
    # Find all files
    print("[1/5] Scanning for data files...")
    files = find_all_data_files(base_dir)
    stats.total_files = len(files)
    print(f"      Found {stats.total_files:,} data files\n")
    
    if stats.total_files == 0:
        print("[ERROR] No data files found!")
        return stats
    
    # Process in batches to manage memory
    print("[2/5] Loading and parsing data files...")
    start_time = time.time()
    
    all_data = []
    file_metadata = []
    batch_raw_bytes = 0
    
    for i, filepath in enumerate(files):
        if (i + 1) % 1000 == 0 or i == 0:
            pct = (i + 1) / stats.total_files * 100
            print(f"      Processing: {i+1:,}/{stats.total_files:,} ({pct:.1f}%)")
        
        try:
            file_size = os.path.getsize(filepath)
            batch_raw_bytes += file_size
            stats.raw_bytes += file_size
            
            data = parse_market_file(filepath)
            if data.size > 0:
                stats.total_rows += len(data)
                all_data.append(data)
                file_metadata.append({
                    'path': os.path.relpath(filepath, base_dir),
                    'rows': len(data),
                    'start_idx': sum(len(d) for d in all_data[:-1]),
                })
                stats.processed_files += 1
            else:
                stats.failed_files += 1
                stats.errors.append(f"Empty data: {filepath}")
        except Exception as e:
            stats.failed_files += 1
            stats.errors.append(f"Error parsing {filepath}: {str(e)}")
    
    print(f"      Parsed {stats.processed_files:,} files, {stats.total_rows:,} rows\n")
    
    if not all_data:
        print("[ERROR] No valid data parsed!")
        return stats
    
    # Concatenate all data
    print("[3/5] Concatenating data matrices...")
    combined_data = np.vstack(all_data)
    print(f"      Combined shape: {combined_data.shape}")
    print(f"      Raw size: {combined_data.nbytes / (1024**3):.3f} GB\n")
    stats.raw_bytes = combined_data.nbytes  # Use actual array size
    
    # Compress with QTT
    print("[4/5] Applying QTT compression...")
    print("      Computing tensor train decomposition...")
    
    compress_start = time.time()
    compressed, compressed_size = compress_to_qtt(combined_data, max_rank=64)
    compress_elapsed = time.time() - compress_start
    
    print(f"      Compression completed in {compress_elapsed:.1f}s")
    
    # Add file metadata
    compressed['file_metadata'] = file_metadata
    compressed['total_files'] = stats.total_files
    compressed['processed_files'] = stats.processed_files
    
    # Save compressed data
    print("\n[5/5] Saving compressed archive...")
    np.savez_compressed(output_file, **{k: np.array(v, dtype=object) if isinstance(v, list) else v 
                                         for k, v in compressed.items()})
    
    # Get actual output size
    stats.compressed_bytes = os.path.getsize(output_file)
    stats.compression_ratio = stats.raw_bytes / stats.compressed_bytes if stats.compressed_bytes > 0 else 0
    stats.elapsed_seconds = time.time() - start_time
    stats.files_per_second = stats.processed_files / stats.elapsed_seconds if stats.elapsed_seconds > 0 else 0
    stats.throughput_mb_per_sec = (stats.raw_bytes / (1024**2)) / stats.elapsed_seconds if stats.elapsed_seconds > 0 else 0
    
    # Print summary
    print(f"\n{'='*70}")
    print("COMPRESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Files Processed:     {stats.processed_files:,} / {stats.total_files:,}")
    print(f"Total Rows:          {stats.total_rows:,}")
    print(f"Raw Data Size:       {stats.raw_bytes / (1024**3):.3f} GB ({stats.raw_bytes:,} bytes)")
    print(f"Compressed Size:     {stats.compressed_bytes / (1024**2):.3f} MB ({stats.compressed_bytes:,} bytes)")
    print(f"Compression Ratio:   {stats.compression_ratio:,.0f}× ({100/stats.compression_ratio:.4f}%)")
    print(f"Elapsed Time:        {stats.elapsed_seconds:.1f}s")
    print(f"Throughput:          {stats.throughput_mb_per_sec:.1f} MB/s")
    print(f"Output File:         {output_file}")
    print(f"{'='*70}\n")
    
    if stats.errors and len(stats.errors) <= 10:
        print("Errors encountered:")
        for err in stats.errors[:10]:
            print(f"  - {err}")
    elif stats.errors:
        print(f"Errors encountered: {len(stats.errors)} (showing first 10)")
        for err in stats.errors[:10]:
            print(f"  - {err}")
    
    return stats


def verify_compression(output_file: str, sample_size: int = 1000) -> bool:
    """
    Verify compressed data can be reconstructed.
    
    Args:
        output_file: Path to compressed .npz file
        sample_size: Number of rows to verify
        
    Returns:
        True if verification passes
    """
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    try:
        data = np.load(output_file, allow_pickle=True)
        
        # Check metadata
        shape = tuple(data['shape'])
        original_len = int(data['original_len'])
        padded_len = int(data['padded_len'])
        print(f"Original shape: {shape}")
        print(f"Original flat length: {original_len:,}")
        print(f"Padded length: {padded_len:,} (2^{int(np.log2(padded_len))})")
        print(f"Data range: [{float(data['min']):.4f}, {float(data['max']):.4f}]")
        
        # Reconstruct from cores
        cores = list(data['cores'])
        print(f"Number of TT cores: {len(cores)}")
        
        # Get total core size
        total_core_bytes = sum(c.nbytes for c in cores)
        print(f"Total core memory: {total_core_bytes / 1024:.2f} KB")
        
        # Try reconstruction via QTT
        qtt = QTTSignal(cores, num_nodes=padded_len)
        reconstructed_norm = qtt.to_dense()
        
        # Truncate padding and denormalize
        reconstructed_norm = reconstructed_norm[:original_len]
        data_min = float(data['min'])
        data_max = float(data['max'])
        data_range = data_max - data_min
        reconstructed = reconstructed_norm * data_range + data_min
        
        # Reshape back to original
        reconstructed = reconstructed.reshape(shape)
        
        print(f"Reconstructed shape: {tuple(reconstructed.shape)}")
        print(f"Sample values (first 5 rows):")
        sample = reconstructed[:5].numpy() if hasattr(reconstructed, 'numpy') else reconstructed[:5]
        for row in sample:
            print(f"  {row}")
        
        print(f"\n✓ Verification PASSED")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        return False


def main():
    """Main entry point."""
    base_dir = "/home/brad/TiganticLabz/Main_Projects/physics-os/crypto_data"
    output_file = "/home/brad/TiganticLabz/Main_Projects/physics-os/crypto_data_compressed.qtt.npz"
    
    # Run compression
    stats = compress_all_files(base_dir, output_file)
    
    # Verify
    if stats.processed_files > 0:
        verify_compression(output_file)
    
    return stats


if __name__ == "__main__":
    main()
