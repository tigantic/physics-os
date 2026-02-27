#!/usr/bin/env python3
"""
QTT Reconstruction: Decompress and query QTT-compressed data
=============================================================
Loads the 27-core QTT representation and reconstructs:
- Full tensor (if memory permits)
- Single frames
- Point queries (nanosecond access)

Usage:
    python qtt_reconstruct.py --input noaa_24h_qtt.npz --frame 16 --output frame_16.npy
    python qtt_reconstruct.py --input noaa_24h_qtt.npz --point 16,1024,1024
"""

import numpy as np
import torch
import time
import argparse
from pathlib import Path


def compute_bits(shape: tuple) -> tuple:
    """Compute bit widths from shape (must be powers of 2)."""
    import math
    return tuple(int(math.log2(s)) for s in shape)


def load_qtt(path: Path) -> dict:
    """Load QTT cores and metadata from .npz file."""
    data = np.load(path, allow_pickle=True)
    
    # Extract cores (arr_0, arr_1, ..., arr_N)
    cores = []
    i = 0
    while f'arr_{i}' in data:
        cores.append(data[f'arr_{i}'])
        i += 1
    
    shape = tuple(int(x) for x in data['shape'])
    
    # Compute bits from shape (assuming powers of 2)
    if 'bits' in data:
        bits = tuple(int(x) for x in data['bits'])
    else:
        bits = compute_bits(shape)
    
    return {
        'cores': cores,
        'mean': float(data['mean'][0]),
        'std': float(data['std'][0]),
        'shape': shape,
        'original_shape': tuple(int(x) for x in data['original_shape']),
        'ranks': data['ranks'],
        'bits': bits
    }


def morton_deinterleave_3d(idx: int, t_bits: int, y_bits: int, x_bits: int) -> tuple:
    """Extract t, y, x from Morton index."""
    t, y, x = 0, 0, 0
    bit_pos = 0
    max_bits = max(t_bits, y_bits, x_bits)
    
    for b in range(max_bits):
        if b < x_bits:
            x |= ((idx >> bit_pos) & 1) << b
            bit_pos += 1
        if b < y_bits:
            y |= ((idx >> bit_pos) & 1) << b
            bit_pos += 1
        if b < t_bits:
            t |= ((idx >> bit_pos) & 1) << b
            bit_pos += 1
    
    return t, y, x


def index_to_bits(idx: int, n_bits: int) -> list:
    """Convert integer index to list of bits (LSB first)."""
    return [(idx >> b) & 1 for b in range(n_bits)]


def query_point_qtt(cores: list, morton_idx: int, total_bits: int) -> float:
    """
    Query a single point from QTT representation.
    O(total_bits * max_rank^2) complexity - nanoseconds for small ranks.
    """
    bits = index_to_bits(morton_idx, total_bits)
    
    # Start with first core slice
    result = cores[0][0, bits[0], :].astype(np.float32)
    
    # Contract through remaining cores
    for k in range(1, len(cores)):
        core = cores[k][:, bits[k], :].astype(np.float32)
        result = result @ core
    
    return float(result[0])


def reconstruct_full(cores: list, mean: float, std: float, 
                     shape: tuple, bits: tuple, device: str = 'cuda') -> np.ndarray:
    """
    Reconstruct full tensor from QTT cores.
    Warning: This expands to full size (e.g., 0.5 GB for 32x2048x2048).
    """
    n_t, n_y, n_x = shape
    t_bits, y_bits, x_bits = bits
    total_bits = t_bits + y_bits + x_bits
    total_elements = 2 ** total_bits
    
    print(f"Reconstructing {total_elements:,} elements...")
    t0 = time.time()
    
    device = torch.device(device)
    
    # Convert cores to torch
    cores_gpu = [torch.from_numpy(c.astype(np.float32)).to(device) for c in cores]
    
    # Full contraction (memory intensive)
    # Start from leftmost core
    result = cores_gpu[0].reshape(2, -1)  # (2, r1)
    
    for k in range(1, len(cores_gpu)):
        core = cores_gpu[k]  # (r_{k-1}, 2, r_k)
        r_prev, _, r_next = core.shape
        
        # result: (2^k, r_k)
        # Contract: result @ core reshaped
        result = result @ core.reshape(r_prev, 2 * r_next)
        result = result.reshape(-1, r_next)
    
    result = result.flatten().cpu().numpy()
    
    # Denormalize
    result = result * std + mean
    
    # Inverse Morton to get (t, y, x) ordering
    print(f"  Contraction: {time.time()-t0:.2f}s")
    print("  Inverse Morton reordering...")
    
    t0 = time.time()
    output = np.zeros((n_t, n_y, n_x), dtype=np.float32)
    
    # Build inverse Morton map
    for ti in range(n_t):
        for yi in range(n_y):
            for xi in range(n_x):
                # Compute Morton index
                morton_idx = 0
                bit_pos = 0
                max_bits = max(t_bits, y_bits, x_bits)
                for b in range(max_bits):
                    if b < x_bits:
                        morton_idx |= ((xi >> b) & 1) << bit_pos
                        bit_pos += 1
                    if b < y_bits:
                        morton_idx |= ((yi >> b) & 1) << bit_pos
                        bit_pos += 1
                    if b < t_bits:
                        morton_idx |= ((ti >> b) & 1) << bit_pos
                        bit_pos += 1
                
                output[ti, yi, xi] = result[morton_idx]
    
    print(f"  Reorder: {time.time()-t0:.2f}s")
    
    return output


def reconstruct_frame(cores: list, mean: float, std: float,
                      shape: tuple, bits: tuple, frame_idx: int,
                      device: str = 'cuda') -> np.ndarray:
    """
    Reconstruct a single frame from QTT representation.
    More memory efficient than full reconstruction.
    """
    n_t, n_y, n_x = shape
    t_bits, y_bits, x_bits = bits
    
    if frame_idx >= n_t:
        raise ValueError(f"Frame {frame_idx} out of range (max {n_t-1})")
    
    print(f"Reconstructing frame {frame_idx}...")
    t0 = time.time()
    
    device_obj = torch.device(device)
    
    # For each (y, x) point in the frame, query the QTT
    frame = np.zeros((n_y, n_x), dtype=np.float32)
    
    # Vectorized approach: batch query all points
    total_bits = t_bits + y_bits + x_bits
    
    for yi in range(n_y):
        for xi in range(n_x):
            # Compute Morton index for (frame_idx, yi, xi)
            morton_idx = 0
            bit_pos = 0
            max_bits = max(t_bits, y_bits, x_bits)
            for b in range(max_bits):
                if b < x_bits:
                    morton_idx |= ((xi >> b) & 1) << bit_pos
                    bit_pos += 1
                if b < y_bits:
                    morton_idx |= ((yi >> b) & 1) << bit_pos
                    bit_pos += 1
                if b < t_bits:
                    morton_idx |= ((frame_idx >> b) & 1) << bit_pos
                    bit_pos += 1
            
            # Query point
            frame[yi, xi] = query_point_qtt(cores, morton_idx, total_bits)
        
        if (yi + 1) % 256 == 0:
            print(f"  Row {yi+1}/{n_y}")
    
    # Denormalize
    frame = frame * std + mean
    
    print(f"  Reconstruction: {time.time()-t0:.2f}s")
    
    return frame


def benchmark_point_query(cores: list, bits: tuple, n_queries: int = 10000) -> float:
    """Benchmark point query speed."""
    total_bits = sum(bits)
    max_idx = 2 ** total_bits
    
    # Random indices
    indices = np.random.randint(0, max_idx, size=n_queries)
    
    t0 = time.time()
    for idx in indices:
        _ = query_point_qtt(cores, int(idx), total_bits)
    
    elapsed = time.time() - t0
    ns_per_query = (elapsed / n_queries) * 1e9
    
    return ns_per_query


def main():
    parser = argparse.ArgumentParser(description='QTT Reconstruction')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input .npz file')
    parser.add_argument('--frame', '-f', type=int, default=None,
                        help='Reconstruct specific frame index')
    parser.add_argument('--point', '-p', type=str, default=None,
                        help='Query point as t,y,x')
    parser.add_argument('--full', action='store_true',
                        help='Reconstruct full tensor')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark point query speed')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .npy file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Load QTT
    print(f"Loading {args.input}...")
    qtt = load_qtt(Path(args.input))
    
    print(f"  Shape: {qtt['shape']}")
    print(f"  Original: {qtt['original_shape']}")
    print(f"  Cores: {len(qtt['cores'])}")
    print(f"  Max rank: {max(qtt['ranks'])}")
    print()
    
    if args.benchmark:
        print("Benchmarking point queries...")
        ns = benchmark_point_query(qtt['cores'], qtt['bits'])
        print(f"  {ns:.1f} ns per query")
        print(f"  {1e9/ns:.0f} queries/sec")
        return
    
    if args.point:
        t, y, x = map(int, args.point.split(','))
        t_bits, y_bits, x_bits = qtt['bits']
        total_bits = t_bits + y_bits + x_bits
        
        # Compute Morton index
        morton_idx = 0
        bit_pos = 0
        max_bits = max(t_bits, y_bits, x_bits)
        for b in range(max_bits):
            if b < x_bits:
                morton_idx |= ((x >> b) & 1) << bit_pos
                bit_pos += 1
            if b < y_bits:
                morton_idx |= ((y >> b) & 1) << bit_pos
                bit_pos += 1
            if b < t_bits:
                morton_idx |= ((t >> b) & 1) << bit_pos
                bit_pos += 1
        
        t0 = time.time()
        value = query_point_qtt(qtt['cores'], morton_idx, total_bits)
        value = value * qtt['std'] + qtt['mean']
        elapsed = (time.time() - t0) * 1e6
        
        print(f"Point ({t}, {y}, {x}) = {value:.6f}")
        print(f"Query time: {elapsed:.1f} µs")
        return
    
    if args.frame is not None:
        frame = reconstruct_frame(
            qtt['cores'], qtt['mean'], qtt['std'],
            qtt['shape'], qtt['bits'], args.frame, args.device
        )
        
        if args.output:
            np.save(args.output, frame)
            print(f"Saved to {args.output}")
        else:
            print(f"Frame stats: min={frame.min():.2f}, max={frame.max():.2f}, mean={frame.mean():.2f}")
        return
    
    if args.full:
        tensor = reconstruct_full(
            qtt['cores'], qtt['mean'], qtt['std'],
            qtt['shape'], qtt['bits'], args.device
        )
        
        if args.output:
            np.save(args.output, tensor)
            print(f"Saved to {args.output}")
        else:
            print(f"Tensor stats: min={tensor.min():.2f}, max={tensor.max():.2f}, mean={tensor.mean():.2f}")
        return
    
    print("Specify --frame, --point, --full, or --benchmark")


if __name__ == '__main__':
    main()
