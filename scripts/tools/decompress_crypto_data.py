#!/usr/bin/env python3
"""
Quantum Decompression of Market Data
=====================================
Reconstructs the original data from QTT compressed format.

Uses efficient einsum-based contraction from pure_qtt_ops.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import torch


def qtt_to_dense_fast(cores: List[torch.Tensor]) -> torch.Tensor:
    """
    Efficient QTT to dense conversion using einsum contractions.
    
    This is the FAST method from pure_qtt_ops.py - uses torch.einsum
    to contract all cores in O(n_qubits) operations.
    
    Args:
        cores: List of TT cores, each shape (r_left, 2, r_right)
        
    Returns:
        Dense vector of length 2^num_cores
    """
    # Contract all cores using einsum
    result = cores[0]  # (1, 2, r1)
    
    for i in range(1, len(cores)):
        c = cores[i]  # (r_{i-1}, 2, r_i)
        # result: (..., r_{i-1}) @ c: (r_{i-1}, 2, r_i) -> (..., 2, r_i)
        result = torch.einsum("...i,ijk->...jk", result, c)
    
    # Final shape: (1, 2, 2, ..., 2, 1) -> (2^n,)
    return result.squeeze(0).squeeze(-1).reshape(-1)


def decompress_qtt(input_file: str, output_file: str = None) -> np.ndarray:
    """
    Decompress a QTT-compressed archive back to original data.
    
    Args:
        input_file: Path to .qtt.npz compressed file
        output_file: Optional path to save reconstructed .npy file
        
    Returns:
        Reconstructed numpy array
    """
    print(f"\n{'='*70}")
    print("QUANTUM TENSOR TRAIN (QTT) DECOMPRESSION")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file or '(memory only)'}")
    print(f"{'='*70}\n")
    
    # Load compressed data
    print("[1/4] Loading compressed archive...")
    data = np.load(input_file, allow_pickle=True)
    
    shape = tuple(data['shape'])
    original_len = int(data['original_len'])
    padded_len = int(data['padded_len'])
    data_min = float(data['min'])
    data_max = float(data['max'])
    
    print(f"      Original shape: {shape}")
    print(f"      Elements: {original_len:,} (padded to {padded_len:,})")
    print(f"      Data range: [{data_min:.4f}, {data_max:.4f}]")
    
    # Load cores
    print("\n[2/4] Loading TT cores...")
    cores_np = list(data['cores'])
    cores = [torch.from_numpy(c).float() for c in cores_np]
    
    total_core_bytes = sum(c.numel() * 4 for c in cores)
    print(f"      {len(cores)} cores, {total_core_bytes / 1024:.2f} KB")
    for i, c in enumerate(cores[:5]):
        print(f"        Core {i}: {tuple(c.shape)}")
    if len(cores) > 5:
        print(f"        ... ({len(cores) - 5} more)")
    
    # Reconstruct using fast einsum method
    print("\n[3/4] Reconstructing data (fast einsum contraction)...")
    start_time = time.time()
    
    # This is the EFFICIENT method - O(d * r^2 * 2) operations per core
    # Total: O(d^2 * r^2) where d=29, r~64 = very fast
    reconstructed_padded = qtt_to_dense_fast(cores)
    
    elapsed = time.time() - start_time
    print(f"      Contraction completed in {elapsed:.2f}s")
    print(f"      Reconstructed {len(reconstructed_padded):,} elements")
    
    # Truncate padding and denormalize
    reconstructed_norm = reconstructed_padded[:original_len]
    data_range = data_max - data_min
    reconstructed = reconstructed_norm * data_range + data_min
    
    # Reshape to original
    reconstructed = reconstructed.reshape(shape)
    
    # Convert to numpy
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.numpy()
    
    # Save if requested
    if output_file:
        print(f"\n[4/4] Saving to {output_file}...")
        np.save(output_file, reconstructed)
        output_size = os.path.getsize(output_file)
        print(f"      Saved {output_size:,} bytes ({output_size / 1e9:.3f} GB)")
    else:
        print("\n[4/4] Skipping save (no output file specified)")
    
    # Summary
    print(f"\n{'='*70}")
    print("DECOMPRESSION COMPLETE")
    print(f"{'='*70}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstructed dtype: {reconstructed.dtype}")
    print(f"Memory size:         {reconstructed.nbytes / 1e9:.3f} GB")
    print(f"Elapsed time:        {elapsed:.2f}s")
    print(f"\nSample values (first 5 rows):")
    for i, row in enumerate(reconstructed[:5]):
        print(f"  Row {i}: {row}")
    print(f"{'='*70}\n")
    
    return reconstructed


def main():
    """Main entry point."""
    input_file = "/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/crypto_data_compressed.qtt.npz"
    output_file = "/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/crypto_data_reconstructed.npy"
    
    reconstructed = decompress_qtt(input_file, output_file)
    
    return reconstructed


if __name__ == "__main__":
    main()
