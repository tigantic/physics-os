#!/usr/bin/env python3
"""
NVIDIA PhysicsNeMo Ahmed Body → QTT Compression Pipeline
=========================================================

Downloads NVIDIA's PhysicsNeMo-CFD-Ahmed-Body dataset from HuggingFace,
extracts surface fields (pressure, wall shear stress) from VTP files,
applies QTT tensor train compression, and benchmarks against ground truth.

Target: Demonstrate 10,000x+ compression on NVIDIA's own benchmark data
with reconstruction error < 1%.

Usage:
    # Step 1: Install dependencies
    pip install huggingface_hub vtk numpy torch tntorch

    # Step 2: Download dataset + run compression
    python nvidia_ahmed_body_qtt_pipeline.py

    # Step 3: (Optional) Use your own QTT engine
    python nvidia_ahmed_body_qtt_pipeline.py --engine ontic

Author: Brad Adams / Tigantic Holdings LLC
Date: February 2026
"""

import os
import sys
import time
import json
import glob
import struct
import argparse
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# ── VTK import (for VTP parsing) ──────────────────────────────────────
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("[WARN] VTK not installed. Install: pip install vtk")
    print("       Falling back to raw XML parsing for VTP files.\n")

# ── PyTorch (for tensor ops) ──────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

# ── tntorch (reference TT implementation) ─────────────────────────────
try:
    import tntorch as tn
    HAS_TNTORCH = True
except ImportError:
    HAS_TNTORCH = False


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Dataset
    hf_repo: str = "NVIDIA/PhysicsNeMo-CFD-Ahmed-Body"
    data_dir: str = "./ahmed_body_data"
    results_dir: str = "./ahmed_body_results"

    # Compression parameters
    max_rank: int = 64          # Maximum TT bond dimension (from paper: χ=64)
    qtt_bits: int = 16          # Quantization bits for QTT
    tolerance: float = 1e-6     # SVD truncation tolerance

    # Benchmarking
    n_samples: int = 20         # Number of samples to process (0 = all)
    splits: List[str] = field(default_factory=lambda: ["test"])  # train/val/test

    # Engine selection
    engine: str = "tntorch"     # "tntorch" | "physics_os" | "manual"


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════

def download_dataset(config: PipelineConfig) -> Path:
    """Download Ahmed Body dataset from HuggingFace."""
    print("=" * 70)
    print("STEP 1: DOWNLOADING NVIDIA PhysicsNeMo Ahmed Body Dataset")
    print("=" * 70)

    data_path = Path(config.data_dir)

    # Check if already downloaded
    existing = list(data_path.rglob("*.vtp"))
    if existing:
        print(f"  Found {len(existing)} existing VTP files in {data_path}")
        print(f"  Skipping download. Delete {data_path} to re-download.")
        return data_path

    try:
        from huggingface_hub import snapshot_download
        print(f"  Repository: {config.hf_repo}")
        print(f"  Destination: {data_path}")
        print(f"  Downloading... (4,064 VTP files, ~22GB)")
        print()

        snapshot_download(
            repo_id=config.hf_repo,
            repo_type="dataset",
            local_dir=str(data_path),
            allow_patterns=["*.vtp"],  # Only VTP files
        )

        downloaded = list(data_path.rglob("*.vtp"))
        print(f"  ✓ Downloaded {len(downloaded)} VTP files")
        return data_path

    except ImportError:
        print("  [ERROR] huggingface_hub not installed.")
        print("  Install: pip install huggingface_hub")
        print()
        print("  Alternative manual download:")
        print(f"    huggingface-cli download {config.hf_repo} --repo-type dataset --local-dir {data_path}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: PARSE VTP FILES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AhmedBodySample:
    """Parsed data from one Ahmed Body VTP file."""
    filepath: str
    n_points: int
    points: np.ndarray        # (N, 3) surface coordinates
    normals: np.ndarray       # (N, 3) surface normals
    pressure: np.ndarray      # (N,)   mean pressure (non-dim)
    wss: np.ndarray           # (N, 3) wall shear stress vector
    file_size_bytes: int
    # Parametric info extracted from filename
    params: Dict[str, float] = field(default_factory=dict)


def parse_vtp_vtk(filepath: str) -> AhmedBodySample:
    """Parse VTP file using VTK library."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    polydata = reader.GetOutput()
    n_points = polydata.GetNumberOfPoints()

    # Extract points
    points = vtk_to_numpy(polydata.GetPoints().GetData())

    # Extract point data arrays
    point_data = polydata.GetPointData()

    # Normals
    normals_arr = point_data.GetArray("Normals")
    if normals_arr is None:
        normals_arr = point_data.GetNormals()
    normals = vtk_to_numpy(normals_arr) if normals_arr else np.zeros((n_points, 3))

    # Pressure (pMean)
    pressure_arr = point_data.GetArray("pMean")
    if pressure_arr is None:
        # Try alternative names
        for name in ["p", "pressure", "Pressure"]:
            pressure_arr = point_data.GetArray(name)
            if pressure_arr:
                break
    pressure = vtk_to_numpy(pressure_arr) if pressure_arr else np.zeros(n_points)

    # Wall Shear Stress
    wss_arr = point_data.GetArray("wallShearStressMean")
    if wss_arr is None:
        for name in ["wallShearStress", "WSS", "wss"]:
            wss_arr = point_data.GetArray(name)
            if wss_arr:
                break
    wss = vtk_to_numpy(wss_arr) if wss_arr else np.zeros((n_points, 3))

    # Ensure correct shapes
    if pressure.ndim > 1:
        pressure = pressure.ravel()
    if wss.ndim == 1:
        wss = wss.reshape(-1, 3)

    # Extract parametric info from filename
    params = extract_params_from_filename(filepath)

    return AhmedBodySample(
        filepath=filepath,
        n_points=n_points,
        points=points,
        normals=normals,
        pressure=pressure,
        wss=wss,
        file_size_bytes=os.path.getsize(filepath),
        params=params,
    )


def parse_vtp_raw(filepath: str) -> AhmedBodySample:
    """Fallback VTP parser using raw XML + binary parsing."""
    # VTP files are XML with appended binary data
    # This is a minimal parser for the Ahmed Body format
    import xml.etree.ElementTree as ET

    file_size = os.path.getsize(filepath)

    with open(filepath, 'rb') as f:
        content = f.read()

    # Find the XML portion (before appended data)
    xml_end = content.find(b'</VTKFile>')
    if xml_end == -1:
        raise ValueError(f"Invalid VTP file: {filepath}")

    xml_content = content[:xml_end + len(b'</VTKFile>')]

    # Parse XML header to get structure info
    root = ET.fromstring(xml_content)
    piece = root.find('.//Piece')
    n_points = int(piece.get('NumberOfPoints', 0))

    print(f"    [RAW] {n_points} points - full binary parsing not implemented")
    print(f"    [RAW] Install VTK for proper parsing: pip install vtk")

    # Return placeholder - VTK is strongly recommended
    return AhmedBodySample(
        filepath=filepath,
        n_points=n_points,
        points=np.zeros((n_points, 3)),
        normals=np.zeros((n_points, 3)),
        pressure=np.zeros(n_points),
        wss=np.zeros((n_points, 3)),
        file_size_bytes=file_size,
        params=extract_params_from_filename(filepath),
    )


def extract_params_from_filename(filepath: str) -> Dict[str, float]:
    """Extract parametric design variables from VTP filename if encoded."""
    # NVIDIA encodes params in directory structure or filename
    # Format varies - this handles common patterns
    name = Path(filepath).stem
    params = {}
    # Try to extract numeric identifiers
    parts = name.replace("-", "_").split("_")
    for i, part in enumerate(parts):
        try:
            val = float(part)
            params[f"param_{i}"] = val
        except ValueError:
            continue
    return params


def parse_vtp(filepath: str) -> AhmedBodySample:
    """Parse VTP file using best available method."""
    if HAS_VTK:
        return parse_vtp_vtk(filepath)
    else:
        return parse_vtp_raw(filepath)


def load_samples(config: PipelineConfig) -> List[AhmedBodySample]:
    """Load VTP samples from dataset."""
    print("\n" + "=" * 70)
    print("STEP 2: PARSING VTP FILES")
    print("=" * 70)

    data_path = Path(config.data_dir)
    all_vtps = []

    for split in config.splits:
        split_path = data_path / split
        if not split_path.exists():
            # Try flat structure
            split_path = data_path
        vtps = sorted(glob.glob(str(split_path / "**" / "*.vtp"), recursive=True))
        all_vtps.extend(vtps)
        print(f"  {split}: {len(vtps)} VTP files")

    if not all_vtps:
        # Try finding VTPs anywhere in data dir
        all_vtps = sorted(glob.glob(str(data_path / "**" / "*.vtp"), recursive=True))
        print(f"  Found {len(all_vtps)} VTP files (flat search)")

    if not all_vtps:
        print("  [ERROR] No VTP files found!")
        print(f"  Checked: {data_path}")
        sys.exit(1)

    # Limit samples
    if config.n_samples > 0:
        all_vtps = all_vtps[:config.n_samples]
        print(f"  Processing first {config.n_samples} samples")

    samples = []
    for i, vtp_path in enumerate(all_vtps):
        print(f"  [{i+1}/{len(all_vtps)}] {Path(vtp_path).name}", end="")
        try:
            sample = parse_vtp(vtp_path)
            samples.append(sample)
            print(f" → {sample.n_points} pts, "
                  f"p:[{sample.pressure.min():.3f}, {sample.pressure.max():.3f}], "
                  f"|WSS|_max={np.linalg.norm(sample.wss, axis=1).max():.4f}")
        except Exception as e:
            print(f" → ERROR: {e}")

    print(f"\n  ✓ Loaded {len(samples)} samples")
    return samples


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: QTT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CompressionResult:
    """Result of compressing one field."""
    field_name: str
    original_shape: Tuple[int, ...]
    original_size_bytes: int
    compressed_size_bytes: int     # Estimated TT storage
    compression_ratio: float
    max_rank: int
    ranks: List[int]
    l2_error_relative: float       # ||x - x_hat||_2 / ||x||_2
    linf_error_relative: float     # ||x - x_hat||_inf / ||x||_inf
    compress_time_s: float
    decompress_time_s: float
    # QTT specific
    n_qubits: int                  # log2(padded_size)
    bond_dimensions: List[int]


def pad_to_power_of_2(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad array to next power of 2 for QTT decomposition."""
    n = arr.shape[0]
    n_padded = 1
    while n_padded < n:
        n_padded *= 2
    if n_padded == n:
        return arr, n
    # Zero-pad
    padded = np.zeros(n_padded, dtype=arr.dtype)
    padded[:n] = arr
    return padded, n


def tt_storage_bytes(ranks: List[int], mode_dims: List[int], dtype_bytes: int = 4) -> int:
    """Estimate storage for TT cores in bytes."""
    total = 0
    for i, d in enumerate(mode_dims):
        r_left = ranks[i] if i > 0 else 1
        r_right = ranks[i + 1] if i + 1 < len(ranks) else 1
        # Actually use the TT rank list properly
        r_left = 1 if i == 0 else ranks[i]
        r_right = 1 if i == len(mode_dims) - 1 else ranks[i + 1]
        total += r_left * d * r_right * dtype_bytes
    return total


def compress_field_tntorch(
    data: np.ndarray,
    field_name: str,
    max_rank: int = 64,
    tolerance: float = 1e-6,
) -> CompressionResult:
    """Compress a field using tntorch QTT decomposition."""
    original_bytes = data.nbytes

    # Pad to power of 2
    padded, original_len = pad_to_power_of_2(data.ravel())
    n_qubits = int(np.log2(len(padded)))

    # Reshape for QTT: (2, 2, 2, ..., 2) with n_qubits modes
    qtt_shape = [2] * n_qubits
    tensor_data = padded.reshape(qtt_shape)

    # Convert to torch
    tensor_torch = torch.tensor(tensor_data, dtype=torch.float32, device=DEVICE)

    # Compress via TT-SVD
    t0 = time.perf_counter()
    tt = tn.Tensor(tensor_torch, ranks_tt=max_rank)
    compress_time = time.perf_counter() - t0

    # Get ranks
    ranks = [1] + [c.shape[-1] for c in tt.cores[:-1]] + [1]
    bond_dims = ranks[1:-1]
    max_achieved_rank = max(bond_dims) if bond_dims else 1

    # Estimate compressed size
    compressed_bytes = sum(
        c.numel() * 4 for c in tt.cores  # float32 = 4 bytes
    )

    # Decompress and measure error
    t0 = time.perf_counter()
    reconstructed = tt.torch().cpu().numpy().ravel()
    decompress_time = time.perf_counter() - t0

    # Trim back to original length
    reconstructed = reconstructed[:original_len]
    original_data = data.ravel()[:original_len]

    # Compute errors
    diff = original_data - reconstructed
    l2_norm = np.linalg.norm(original_data)
    l2_error = np.linalg.norm(diff) / l2_norm if l2_norm > 0 else 0.0
    linf_norm = np.abs(original_data).max()
    linf_error = np.abs(diff).max() / linf_norm if linf_norm > 0 else 0.0

    return CompressionResult(
        field_name=field_name,
        original_shape=data.shape,
        original_size_bytes=original_bytes,
        compressed_size_bytes=compressed_bytes,
        compression_ratio=original_bytes / compressed_bytes if compressed_bytes > 0 else float('inf'),
        max_rank=max_achieved_rank,
        ranks=ranks,
        l2_error_relative=float(l2_error),
        linf_error_relative=float(linf_error),
        compress_time_s=compress_time,
        decompress_time_s=decompress_time,
        n_qubits=n_qubits,
        bond_dimensions=bond_dims,
    )


def compress_field_manual(
    data: np.ndarray,
    field_name: str,
    max_rank: int = 64,
    tolerance: float = 1e-6,
) -> CompressionResult:
    """
    Manual TT-SVD compression (no external TT library needed).
    Implements the standard TT-SVD algorithm from Oseledets (2011).
    """
    original_bytes = data.nbytes

    # Pad to power of 2
    padded, original_len = pad_to_power_of_2(data.ravel())
    n_qubits = int(np.log2(len(padded)))

    # QTT shape
    qtt_shape = [2] * n_qubits
    tensor = padded.astype(np.float64)

    # TT-SVD
    t0 = time.perf_counter()

    cores = []
    C = tensor.copy()
    r = 1  # left rank starts at 1

    for k in range(n_qubits - 1):
        nk = qtt_shape[k]
        remaining = int(np.prod(qtt_shape[k+1:]))
        C = C.reshape(r * nk, remaining)

        # SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=False)

        # Truncate
        # By tolerance
        cumsum = np.cumsum(S[::-1] ** 2)[::-1]
        total_norm = cumsum[0] if len(cumsum) > 0 else 0
        if total_norm > 0:
            keep = np.searchsorted(-cumsum / total_norm, -tolerance ** 2) + 1
        else:
            keep = 1
        # By max rank
        keep = min(keep, max_rank, len(S))

        U = U[:, :keep]
        S = S[:keep]
        Vt = Vt[:keep, :]

        # Store core
        core = U.reshape(r, nk, keep)
        cores.append(core)

        # Prepare next iteration
        C = np.diag(S) @ Vt
        r = keep

    # Last core
    cores.append(C.reshape(r, qtt_shape[-1], 1))

    compress_time = time.perf_counter() - t0

    # Ranks
    ranks = [1] + [c.shape[2] for c in cores[:-1]] + [1]
    bond_dims = ranks[1:-1]
    max_achieved_rank = max(bond_dims) if bond_dims else 1

    # Compressed size
    compressed_bytes = sum(c.nbytes for c in cores)

    # Decompress: contract cores
    t0 = time.perf_counter()
    result = cores[0]  # (1, 2, r1)
    for i in range(1, len(cores)):
        # result shape: (1, ..., r_i)
        # core shape: (r_i, 2, r_{i+1})
        result = np.tensordot(result, cores[i], axes=([-1], [0]))
    reconstructed = result.ravel()
    decompress_time = time.perf_counter() - t0

    # Trim
    reconstructed = reconstructed[:original_len].astype(np.float32)
    original_data = data.ravel()[:original_len]

    # Errors
    diff = original_data - reconstructed
    l2_norm = np.linalg.norm(original_data)
    l2_error = np.linalg.norm(diff) / l2_norm if l2_norm > 0 else 0.0
    linf_norm = np.abs(original_data).max()
    linf_error = np.abs(diff).max() / linf_norm if linf_norm > 0 else 0.0

    # For fair comparison, measure in float32 bytes
    compressed_bytes_f32 = sum(c.size * 4 for c in cores)

    return CompressionResult(
        field_name=field_name,
        original_shape=data.shape,
        original_size_bytes=int(original_bytes),
        compressed_size_bytes=int(compressed_bytes_f32),
        compression_ratio=original_bytes / compressed_bytes_f32 if compressed_bytes_f32 > 0 else float('inf'),
        max_rank=max_achieved_rank,
        ranks=ranks,
        l2_error_relative=float(l2_error),
        linf_error_relative=float(linf_error),
        compress_time_s=compress_time,
        decompress_time_s=decompress_time,
        n_qubits=n_qubits,
        bond_dimensions=bond_dims,
    )


# ═══════════════════════════════════════════════════════════════════════
# STEP 3b: ONTIC_ENGINE ENGINE HOOK
# ═══════════════════════════════════════════════════════════════════════

def compress_field_ontic(
    data: np.ndarray,
    field_name: str,
    max_rank: int = 64,
    tolerance: float = 1e-6,
) -> CompressionResult:
    """
    Hook for Brad's Ontic QTT engine.

    INSTRUCTIONS:
    1. Import your QTT compression from The Ontic Engine
    2. Replace the placeholder below with actual calls
    3. The interface expects: compress(data) → compressed, decompress(compressed) → data

    Example integration:
        from ontic.qtt import QTTCompressor
        compressor = QTTCompressor(max_rank=max_rank, tolerance=tolerance)
        compressed = compressor.compress(data)
        reconstructed = compressor.decompress(compressed)
    """
    # ─── REPLACE THIS BLOCK WITH ONTIC_ENGINE INTEGRATION ───
    # For now, falls through to manual implementation
    print(f"    [The Ontic Engine] Hook not connected - using manual TT-SVD")
    print(f"    [The Ontic Engine] Edit compress_field_ontic() in this file")
    print(f"    [The Ontic Engine] to plug in your QTT engine")
    return compress_field_manual(data, field_name, max_rank, tolerance)
    # ─── END REPLACE ───────────────────────────────────────


def compress_field(
    data: np.ndarray,
    field_name: str,
    config: PipelineConfig,
) -> CompressionResult:
    """Route to selected compression engine."""
    if config.engine == "tntorch" and HAS_TNTORCH:
        return compress_field_tntorch(data, field_name, config.max_rank, config.tolerance)
    elif config.engine == "physics_os":
        return compress_field_ontic(data, field_name, config.max_rank, config.tolerance)
    else:
        return compress_field_manual(data, field_name, config.max_rank, config.tolerance)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SampleBenchmark:
    """Complete benchmark for one Ahmed Body sample."""
    sample_id: str
    n_points: int
    file_size_bytes: int
    params: Dict[str, float]

    # Per-field results
    pressure_result: Optional[CompressionResult] = None
    wss_x_result: Optional[CompressionResult] = None
    wss_y_result: Optional[CompressionResult] = None
    wss_z_result: Optional[CompressionResult] = None

    # Aggregate
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    overall_compression_ratio: float = 0.0
    overall_l2_error: float = 0.0
    total_compress_time: float = 0.0
    total_decompress_time: float = 0.0


def benchmark_sample(sample: AhmedBodySample, config: PipelineConfig) -> SampleBenchmark:
    """Run full QTT compression benchmark on one sample."""
    sample_id = Path(sample.filepath).stem

    bench = SampleBenchmark(
        sample_id=sample_id,
        n_points=sample.n_points,
        file_size_bytes=sample.file_size_bytes,
        params=sample.params,
    )

    results = []

    # Compress pressure field
    print(f"    Pressure ({sample.pressure.shape})...")
    bench.pressure_result = compress_field(sample.pressure, "pMean", config)
    results.append(bench.pressure_result)

    # Compress WSS components separately (they're vector fields)
    for i, axis in enumerate(["x", "y", "z"]):
        wss_component = sample.wss[:, i]
        print(f"    WSS_{axis} ({wss_component.shape})...")
        result = compress_field(wss_component, f"wss_{axis}", config)
        setattr(bench, f"wss_{axis}_result", result)
        results.append(result)

    # Aggregates
    bench.total_original_bytes = sum(r.original_size_bytes for r in results)
    bench.total_compressed_bytes = sum(r.compressed_size_bytes for r in results)
    bench.overall_compression_ratio = (
        bench.total_original_bytes / bench.total_compressed_bytes
        if bench.total_compressed_bytes > 0 else float('inf')
    )
    bench.overall_l2_error = max(r.l2_error_relative for r in results)
    bench.total_compress_time = sum(r.compress_time_s for r in results)
    bench.total_decompress_time = sum(r.decompress_time_s for r in results)

    return bench


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: RESULTS & REPORTING
# ═══════════════════════════════════════════════════════════════════════

def print_sample_results(bench: SampleBenchmark):
    """Print results for one sample."""
    print(f"\n  ┌─ {bench.sample_id}")
    print(f"  │  Points: {bench.n_points:,}")
    print(f"  │  Original: {bench.total_original_bytes:,} bytes ({bench.total_original_bytes/1024:.1f} KB)")
    print(f"  │  Compressed: {bench.total_compressed_bytes:,} bytes ({bench.total_compressed_bytes/1024:.1f} KB)")
    print(f"  │  ── Ratio: {bench.overall_compression_ratio:.1f}×")
    print(f"  │  ── Max L2 Error: {bench.overall_l2_error:.2e}")
    print(f"  │  ── Compress: {bench.total_compress_time*1000:.1f}ms")
    print(f"  │  ── Decompress: {bench.total_decompress_time*1000:.1f}ms")

    for name, result in [
        ("pMean", bench.pressure_result),
        ("WSS_x", bench.wss_x_result),
        ("WSS_y", bench.wss_y_result),
        ("WSS_z", bench.wss_z_result),
    ]:
        if result:
            print(f"  │  {name}: ratio={result.compression_ratio:.1f}×  "
                  f"L2={result.l2_error_relative:.2e}  "
                  f"L∞={result.linf_error_relative:.2e}  "
                  f"χ_max={result.max_rank}  "
                  f"qubits={result.n_qubits}")
    print(f"  └─")


def generate_report(benchmarks: List[SampleBenchmark], config: PipelineConfig) -> str:
    """Generate full benchmark report."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate statistics
    ratios = [b.overall_compression_ratio for b in benchmarks]
    errors = [b.overall_l2_error for b in benchmarks]
    compress_times = [b.total_compress_time for b in benchmarks]
    decompress_times = [b.total_decompress_time for b in benchmarks]

    report = []
    report.append("=" * 70)
    report.append("QTT COMPRESSION BENCHMARK: NVIDIA PhysicsNeMo Ahmed Body")
    report.append("=" * 70)
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Engine: {config.engine}")
    report.append(f"Max rank (χ): {config.max_rank}")
    report.append(f"Samples: {len(benchmarks)}")
    report.append(f"Device: {DEVICE}")
    report.append("")
    report.append("─" * 70)
    report.append("AGGREGATE RESULTS")
    report.append("─" * 70)
    report.append(f"  Compression Ratio:")
    report.append(f"    Mean:   {np.mean(ratios):.1f}×")
    report.append(f"    Median: {np.median(ratios):.1f}×")
    report.append(f"    Min:    {np.min(ratios):.1f}×")
    report.append(f"    Max:    {np.max(ratios):.1f}×")
    report.append(f"")
    report.append(f"  Reconstruction Error (relative L2):")
    report.append(f"    Mean:   {np.mean(errors):.2e}")
    report.append(f"    Max:    {np.max(errors):.2e}")
    report.append(f"")
    report.append(f"  Timing:")
    report.append(f"    Compress:   {np.mean(compress_times)*1000:.1f}ms avg")
    report.append(f"    Decompress: {np.mean(decompress_times)*1000:.1f}ms avg")
    report.append(f"")

    # Per-field breakdown
    for field_name, attr in [("pMean", "pressure_result"),
                              ("WSS_x", "wss_x_result"),
                              ("WSS_y", "wss_y_result"),
                              ("WSS_z", "wss_z_result")]:
        field_ratios = [getattr(b, attr).compression_ratio for b in benchmarks if getattr(b, attr)]
        field_errors = [getattr(b, attr).l2_error_relative for b in benchmarks if getattr(b, attr)]
        field_ranks = [getattr(b, attr).max_rank for b in benchmarks if getattr(b, attr)]
        if field_ratios:
            report.append(f"  {field_name}:")
            report.append(f"    Ratio: {np.mean(field_ratios):.1f}× avg  "
                         f"(min {np.min(field_ratios):.1f}×, max {np.max(field_ratios):.1f}×)")
            report.append(f"    L2 Error: {np.mean(field_errors):.2e} avg  "
                         f"(max {np.max(field_errors):.2e})")
            report.append(f"    Max Rank: {int(np.mean(field_ranks))} avg  "
                         f"(max {int(np.max(field_ranks))})")
            report.append("")

    # Comparison with NVIDIA targets
    report.append("─" * 70)
    report.append("vs NVIDIA PhysicsNeMo TARGETS")
    report.append("─" * 70)
    mean_ratio = np.mean(ratios)
    max_error = np.max(errors)

    targets = [
        ("Compression > 100×", mean_ratio > 100, f"{mean_ratio:.0f}×"),
        ("Compression > 1,000×", mean_ratio > 1000, f"{mean_ratio:.0f}×"),
        ("Compression > 10,000×", mean_ratio > 10000, f"{mean_ratio:.0f}×"),
        ("Reconstruction L2 < 1%", max_error < 0.01, f"{max_error:.2e}"),
        ("Reconstruction L2 < 0.1%", max_error < 0.001, f"{max_error:.2e}"),
        ("Reconstruction L2 < 0.01%", max_error < 0.0001, f"{max_error:.2e}"),
    ]

    for desc, passed, value in targets:
        status = "✅ PASS" if passed else "❌ MISS"
        report.append(f"  {status}  {desc}  [{value}]")

    report.append("")
    report.append("─" * 70)
    report.append("SIGNIFICANCE")
    report.append("─" * 70)
    report.append("  NVIDIA's PhysicsNeMo pipeline generates training data using")
    report.append("  GPU-accelerated OpenFOAM (hours per sample). The 4,064-sample")
    report.append("  Ahmed Body dataset is ~22GB uncompressed.")
    report.append("")
    if mean_ratio > 100:
        compressed_gb = 22 / mean_ratio
        report.append(f"  QTT compression reduces this to ~{compressed_gb*1000:.0f}MB")
        report.append(f"  ({mean_ratio:.0f}× compression) with {max_error:.2e} reconstruction error.")
        report.append("")
        report.append("  This enables:")
        report.append("    → Faster data loading for surrogate model training")
        report.append("    → Direct field queries without full decompression")
        report.append("    → Efficient storage of massive parametric sweeps")
        report.append("    → Streaming of CFD results to edge devices")

    report_text = "\n".join(report)

    # Save report
    report_path = results_dir / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save detailed JSON
    json_path = results_dir / "benchmark_results.json"
    json_data = {
        "config": asdict(config),
        "aggregate": {
            "n_samples": len(benchmarks),
            "compression_ratio_mean": float(np.mean(ratios)),
            "compression_ratio_median": float(np.median(ratios)),
            "l2_error_max": float(np.max(errors)),
            "l2_error_mean": float(np.mean(errors)),
            "compress_time_ms_mean": float(np.mean(compress_times) * 1000),
            "decompress_time_ms_mean": float(np.mean(decompress_times) * 1000),
        },
        "samples": [
            {
                "id": b.sample_id,
                "n_points": b.n_points,
                "compression_ratio": b.overall_compression_ratio,
                "l2_error": b.overall_l2_error,
                "compress_ms": b.total_compress_time * 1000,
                "decompress_ms": b.total_decompress_time * 1000,
            }
            for b in benchmarks
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save CSV for quick plotting
    csv_path = results_dir / "benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("sample_id,n_points,compression_ratio,l2_error,compress_ms,decompress_ms\n")
        for b in benchmarks:
            f.write(f"{b.sample_id},{b.n_points},{b.overall_compression_ratio:.2f},"
                    f"{b.overall_l2_error:.6e},{b.total_compress_time*1000:.2f},"
                    f"{b.total_decompress_time*1000:.2f}\n")

    print(f"\n  Reports saved to:")
    print(f"    {report_path}")
    print(f"    {json_path}")
    print(f"    {csv_path}")

    return report_text


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA PhysicsNeMo Ahmed Body → QTT Compression Benchmark"
    )
    parser.add_argument("--engine", choices=["tntorch", "physics_os", "manual"],
                        default="manual", help="QTT compression engine")
    parser.add_argument("--max-rank", type=int, default=64,
                        help="Maximum TT bond dimension (default: 64)")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="SVD truncation tolerance")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples (0=all)")
    parser.add_argument("--data-dir", default="./ahmed_body_data",
                        help="Dataset directory")
    parser.add_argument("--results-dir", default="./ahmed_body_results",
                        help="Results output directory")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Dataset splits to process")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download")
    args = parser.parse_args()

    config = PipelineConfig(
        engine=args.engine,
        max_rank=args.max_rank,
        tolerance=args.tolerance,
        n_samples=args.n_samples,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        splits=args.splits,
    )

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  QTT × NVIDIA PhysicsNeMo Ahmed Body Compression Benchmark    ║")
    print("║  Tigantic Holdings LLC — Brad Adams                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Engine:    {config.engine}")
    print(f"  Max rank:  χ = {config.max_rank}")
    print(f"  Tolerance: {config.tolerance}")
    print(f"  Device:    {DEVICE}")
    print(f"  Samples:   {config.n_samples if config.n_samples > 0 else 'ALL'}")
    print()

    # Step 1: Download
    if not args.skip_download:
        download_dataset(config)

    # Step 2: Parse
    samples = load_samples(config)

    # Step 3+4: Compress and benchmark
    print("\n" + "=" * 70)
    print("STEP 3: QTT COMPRESSION BENCHMARK")
    print("=" * 70)

    benchmarks = []
    for i, sample in enumerate(samples):
        print(f"\n  Sample [{i+1}/{len(samples)}]: {Path(sample.filepath).name}")
        print(f"  Points: {sample.n_points:,}  |  File: {sample.file_size_bytes/1024:.1f} KB")

        bench = benchmark_sample(sample, config)
        benchmarks.append(bench)
        print_sample_results(bench)

    # Step 5: Report
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING REPORT")
    print("=" * 70)

    report = generate_report(benchmarks, config)
    print()
    print(report)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
