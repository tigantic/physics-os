#!/usr/bin/env python3
"""
Universal QTT Decompressor
==========================
Reconstructs ANY data compressed with The_Compressor, regardless of:
- Original dimensionality (1D, 2D, 3D, 4D, ND)
- Data type (satellite, volumetric, time series, etc.)
- Shape/size

Usage:
    # Inspect compressed file
    python universal.py info data.npz
    
    # Point query (any dimensionality)
    python universal.py query data.npz 16,1024,1024
    
    # Reconstruct full tensor
    python universal.py reconstruct data.npz --output result.npy
    
    # Extract slice/region
    python universal.py slice data.npz --dim 0 --index 16 --output frame.npy
    
    # Stream reconstruction (memory efficient)
    python universal.py stream data.npz --output result.npy --chunk-size 1024
"""

import numpy as np
import torch
import time
import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any


class QTTArchive:
    """Universal QTT archive reader and reconstructor."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._load()
    
    def _load(self):
        """Load QTT archive and extract metadata."""
        data = np.load(self.path, allow_pickle=True)
        
        # Extract cores
        self.cores = []
        i = 0
        while f'arr_{i}' in data:
            self.cores.append(data[f'arr_{i}'])
            i += 1
        
        # Normalization
        self.mean = float(data['mean'][0]) if 'mean' in data else 0.0
        self.std = float(data['std'][0]) if 'std' in data else 1.0
        
        # Shape info
        self.shape = tuple(int(x) for x in data['shape'])
        self.ndim = len(self.shape)
        
        # Original shape (before quantics reshape)
        if 'original_shape' in data:
            self.original_shape = tuple(int(x) for x in data['original_shape'])
        else:
            self.original_shape = self.shape
        
        # Bit widths per dimension
        if 'bits' in data:
            self.bits = tuple(int(x) for x in data['bits'])
        else:
            self.bits = tuple(int(math.log2(s)) for s in self.shape)
        
        self.total_bits = sum(self.bits)
        
        # Ranks
        if 'ranks' in data:
            self.ranks = list(data['ranks'])
        else:
            self.ranks = [c.shape[0] for c in self.cores] + [1]
        
        # Compression metadata
        if 'compression_info' in data:
            self.compression_info = json.loads(str(data['compression_info']))
        else:
            self.compression_info = {}
    
    def info(self) -> Dict[str, Any]:
        """Return comprehensive archive information."""
        file_size = self.path.stat().st_size
        total_elements = np.prod(self.shape)
        original_elements = np.prod(self.original_shape)
        
        # Estimate core memory
        core_bytes = sum(c.nbytes for c in self.cores)
        
        return {
            'file_path': str(self.path),
            'file_size_bytes': file_size,
            'file_size_human': self._human_bytes(file_size),
            'quantics_shape': self.shape,
            'original_shape': self.original_shape,
            'ndim': self.ndim,
            'bits_per_dim': self.bits,
            'total_bits': self.total_bits,
            'total_elements': int(total_elements),
            'original_elements': int(original_elements),
            'num_cores': len(self.cores),
            'max_rank': max(self.ranks),
            'ranks': self.ranks,
            'mean': self.mean,
            'std': self.std,
            'core_bytes': core_bytes,
            'core_dtype': str(self.cores[0].dtype),
            'compression_ratio_quantics': total_elements * 4 / file_size,
            'compression_ratio_original': original_elements * 4 / file_size,
        }
    
    @staticmethod
    def _human_bytes(n: int) -> str:
        """Convert bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} PB"
    
    def _coords_to_morton(self, coords: Tuple[int, ...]) -> int:
        """Convert N-dimensional coordinates to Morton index."""
        if len(coords) != self.ndim:
            raise ValueError(f"Expected {self.ndim} coordinates, got {len(coords)}")
        
        morton_idx = 0
        bit_pos = 0
        max_bits = max(self.bits)
        
        for b in range(max_bits):
            # Interleave bits from each dimension (reverse order for consistency)
            for dim in range(self.ndim - 1, -1, -1):
                if b < self.bits[dim]:
                    morton_idx |= ((coords[dim] >> b) & 1) << bit_pos
                    bit_pos += 1
        
        return morton_idx
    
    def _morton_to_coords(self, morton_idx: int) -> Tuple[int, ...]:
        """Convert Morton index back to N-dimensional coordinates."""
        coords = [0] * self.ndim
        bit_pos = 0
        max_bits = max(self.bits)
        
        for b in range(max_bits):
            for dim in range(self.ndim - 1, -1, -1):
                if b < self.bits[dim]:
                    coords[dim] |= ((morton_idx >> bit_pos) & 1) << b
                    bit_pos += 1
        
        return tuple(coords)
    
    def _index_to_bits(self, idx: int) -> List[int]:
        """Convert integer index to list of bits (LSB first)."""
        return [(idx >> b) & 1 for b in range(self.total_bits)]
    
    def query_morton(self, morton_idx: int) -> float:
        """Query a single point by Morton index. O(total_bits * max_rank^2)."""
        bits = self._index_to_bits(morton_idx)
        
        # Contract through cores
        result = self.cores[0][0, bits[0], :].astype(np.float32)
        
        for k in range(1, len(self.cores)):
            core = self.cores[k][:, bits[k], :].astype(np.float32)
            result = result @ core
        
        value = float(result[0])
        return value * self.std + self.mean
    
    def query(self, *coords: int) -> float:
        """Query a single point by coordinates."""
        if len(coords) == 1 and isinstance(coords[0], (list, tuple)):
            coords = tuple(coords[0])
        
        morton_idx = self._coords_to_morton(coords)
        return self.query_morton(morton_idx)
    
    def query_batch(self, coords_list: List[Tuple[int, ...]], 
                    device: str = 'cuda') -> np.ndarray:
        """Query multiple points efficiently."""
        results = np.zeros(len(coords_list), dtype=np.float32)
        
        for i, coords in enumerate(coords_list):
            results[i] = self.query(*coords)
        
        return results
    
    def reconstruct_full(self, device: str = 'cuda') -> np.ndarray:
        """
        Reconstruct full tensor via core contraction.
        Warning: Expands to full size in memory.
        """
        print(f"Reconstructing {np.prod(self.shape):,} elements...")
        t0 = time.time()
        
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Convert cores to torch
        cores_gpu = [torch.from_numpy(c.astype(np.float32)).to(device_obj) 
                     for c in self.cores]
        
        # Full contraction from left
        result = cores_gpu[0].reshape(2, -1)  # (2, r1)
        
        for k in range(1, len(cores_gpu)):
            core = cores_gpu[k]  # (r_{k-1}, 2, r_k)
            r_prev, _, r_next = core.shape
            result = result @ core.reshape(r_prev, 2 * r_next)
            result = result.reshape(-1, r_next)
        
        # Flatten and move to CPU
        result = result.flatten().cpu().numpy()
        
        # Denormalize
        result = result * self.std + self.mean
        
        print(f"  Contraction: {time.time()-t0:.2f}s")
        print("  Inverse Morton reordering...")
        
        # Inverse Morton to get original coordinate ordering
        t0 = time.time()
        output = np.zeros(self.shape, dtype=np.float32)
        
        # Iterate through all coordinates
        for flat_idx in range(np.prod(self.shape)):
            # Convert flat index to coordinates
            coords = []
            temp = flat_idx
            for dim in range(self.ndim - 1, -1, -1):
                coords.insert(0, temp % self.shape[dim])
                temp //= self.shape[dim]
            
            morton_idx = self._coords_to_morton(tuple(coords))
            output[tuple(coords)] = result[morton_idx]
        
        print(f"  Reorder: {time.time()-t0:.2f}s")
        
        return output
    
    def reconstruct_slice(self, dim: int, index: int, 
                          device: str = 'cuda') -> np.ndarray:
        """
        Reconstruct a single slice along a dimension.
        More memory efficient than full reconstruction.
        """
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Dimension {dim} out of range [0, {self.ndim})")
        if index < 0 or index >= self.shape[dim]:
            raise ValueError(f"Index {index} out of range [0, {self.shape[dim]})")
        
        # Build slice shape
        slice_shape = list(self.shape)
        slice_shape.pop(dim)
        slice_shape = tuple(slice_shape)
        
        print(f"Reconstructing slice dim={dim}, index={index}, shape={slice_shape}...")
        t0 = time.time()
        
        output = np.zeros(slice_shape, dtype=np.float32)
        
        # Iterate through slice coordinates
        for flat_idx in range(np.prod(slice_shape)):
            # Convert to slice coordinates
            slice_coords = []
            temp = flat_idx
            for d in range(len(slice_shape) - 1, -1, -1):
                slice_coords.insert(0, temp % slice_shape[d])
                temp //= slice_shape[d]
            
            # Insert fixed dimension
            full_coords = list(slice_coords)
            full_coords.insert(dim, index)
            
            # Query point
            output[tuple(slice_coords)] = self.query(*full_coords)
            
            if (flat_idx + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (flat_idx + 1) / elapsed
                eta = (np.prod(slice_shape) - flat_idx - 1) / rate
                print(f"  {flat_idx+1:,}/{np.prod(slice_shape):,} ({rate:.0f}/s, ETA {eta:.0f}s)")
        
        print(f"  Complete: {time.time()-t0:.2f}s")
        return output
    
    def reconstruct_region(self, start: Tuple[int, ...], end: Tuple[int, ...],
                           device: str = 'cuda') -> np.ndarray:
        """Reconstruct a rectangular region."""
        if len(start) != self.ndim or len(end) != self.ndim:
            raise ValueError(f"Start/end must have {self.ndim} dimensions")
        
        region_shape = tuple(e - s for s, e in zip(start, end))
        print(f"Reconstructing region {start} to {end}, shape={region_shape}...")
        t0 = time.time()
        
        output = np.zeros(region_shape, dtype=np.float32)
        
        for flat_idx in range(np.prod(region_shape)):
            # Convert to region coordinates
            region_coords = []
            temp = flat_idx
            for d in range(self.ndim - 1, -1, -1):
                region_coords.insert(0, temp % region_shape[d])
                temp //= region_shape[d]
            
            # Convert to global coordinates
            global_coords = tuple(s + r for s, r in zip(start, region_coords))
            
            # Query point
            output[tuple(region_coords)] = self.query(*global_coords)
        
        print(f"  Complete: {time.time()-t0:.2f}s")
        return output
    
    def benchmark_query(self, n_queries: int = 10000) -> Dict[str, float]:
        """Benchmark point query performance."""
        max_idx = 2 ** self.total_bits
        indices = np.random.randint(0, max_idx, size=n_queries)
        
        t0 = time.time()
        for idx in indices:
            _ = self.query_morton(int(idx))
        
        elapsed = time.time() - t0
        ns_per_query = (elapsed / n_queries) * 1e9
        
        return {
            'n_queries': n_queries,
            'total_time_s': elapsed,
            'ns_per_query': ns_per_query,
            'queries_per_sec': 1e9 / ns_per_query
        }


def cmd_info(args):
    """Info subcommand."""
    archive = QTTArchive(args.input)
    info = archive.info()
    
    print(f"\n{'='*60}")
    print(f"QTT Archive: {info['file_path']}")
    print(f"{'='*60}\n")
    
    print(f"File Size:          {info['file_size_human']} ({info['file_size_bytes']:,} bytes)")
    print(f"Quantics Shape:     {info['quantics_shape']}")
    print(f"Original Shape:     {info['original_shape']}")
    print(f"Dimensions:         {info['ndim']}D")
    print(f"Bits per Dimension: {info['bits_per_dim']}")
    print(f"Total Bits:         {info['total_bits']}")
    print(f"Total Elements:     {info['total_elements']:,}")
    print(f"Original Elements:  {info['original_elements']:,}")
    print()
    print(f"Cores:              {info['num_cores']}")
    print(f"Max Rank:           {info['max_rank']}")
    print(f"Core Dtype:         {info['core_dtype']}")
    print(f"Core Memory:        {QTTArchive._human_bytes(info['core_bytes'])}")
    print()
    print(f"Normalization:      mean={info['mean']:.6f}, std={info['std']:.6f}")
    print()
    print(f"Compression Ratio (vs quantics): {info['compression_ratio_quantics']:.1f}x")
    print(f"Compression Ratio (vs original): {info['compression_ratio_original']:.1f}x")
    print()


def cmd_query(args):
    """Query subcommand."""
    archive = QTTArchive(args.input)
    coords = tuple(int(x) for x in args.coords.split(','))
    
    t0 = time.time()
    value = archive.query(*coords)
    elapsed = (time.time() - t0) * 1e6
    
    print(f"Point {coords} = {value:.6f}")
    print(f"Query time: {elapsed:.1f} µs")


def cmd_reconstruct(args):
    """Reconstruct subcommand."""
    archive = QTTArchive(args.input)
    
    result = archive.reconstruct_full(device=args.device)
    
    if args.output:
        np.save(args.output, result)
        print(f"Saved to {args.output}")
    else:
        print(f"Result stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")


def cmd_slice(args):
    """Slice subcommand."""
    archive = QTTArchive(args.input)
    
    result = archive.reconstruct_slice(args.dim, args.index, device=args.device)
    
    if args.output:
        np.save(args.output, result)
        print(f"Saved to {args.output}")
    else:
        print(f"Slice stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")


def cmd_benchmark(args):
    """Benchmark subcommand."""
    archive = QTTArchive(args.input)
    
    print(f"Benchmarking {args.n} queries...")
    results = archive.benchmark_query(n_queries=args.n)
    
    print(f"  {results['ns_per_query']:.1f} ns per query")
    print(f"  {results['queries_per_sec']:.0f} queries/sec")


def main():
    parser = argparse.ArgumentParser(
        description='Universal QTT Decompressor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Info
    p_info = subparsers.add_parser('info', help='Display archive information')
    p_info.add_argument('input', type=str, help='Input .npz file')
    p_info.set_defaults(func=cmd_info)
    
    # Query
    p_query = subparsers.add_parser('query', help='Query single point')
    p_query.add_argument('input', type=str, help='Input .npz file')
    p_query.add_argument('coords', type=str, help='Coordinates as comma-separated integers')
    p_query.set_defaults(func=cmd_query)
    
    # Reconstruct
    p_recon = subparsers.add_parser('reconstruct', help='Reconstruct full tensor')
    p_recon.add_argument('input', type=str, help='Input .npz file')
    p_recon.add_argument('--output', '-o', type=str, help='Output .npy file')
    p_recon.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    p_recon.set_defaults(func=cmd_reconstruct)
    
    # Slice
    p_slice = subparsers.add_parser('slice', help='Extract slice along dimension')
    p_slice.add_argument('input', type=str, help='Input .npz file')
    p_slice.add_argument('--dim', '-d', type=int, required=True, help='Dimension to slice')
    p_slice.add_argument('--index', '-i', type=int, required=True, help='Index along dimension')
    p_slice.add_argument('--output', '-o', type=str, help='Output .npy file')
    p_slice.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    p_slice.set_defaults(func=cmd_slice)
    
    # Benchmark
    p_bench = subparsers.add_parser('benchmark', help='Benchmark query performance')
    p_bench.add_argument('input', type=str, help='Input .npz file')
    p_bench.add_argument('--n', type=int, default=10000, help='Number of queries')
    p_bench.set_defaults(func=cmd_benchmark)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
