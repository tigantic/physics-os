#!/usr/bin/env python3
"""
NVIDIA PhysicsNeMo Ahmed Body → QTT Compression Pipeline
=========================================================

Mesh-to-Rectilinear Bridge: converts unstructured VTP surface data
to a structured 3D Cartesian grid, then applies QTT tensor train
compression for massive storage reduction.

The key insight: QTT (Quantized Tensor Train) decomposition requires
structured data on a regular grid. OpenFOAM's unstructured polyMesh
has no implicit adjacency between indices, so QTT sees random noise.

The solution:
  1. Read unstructured VTP surface mesh
  2. Create a 3D UniformGrid bounding box
  3. Fill every grid point with nearest-surface-point field values
  4. QTT compress the smooth, structured 3D representation
  5. Measure compression ratio against original data sizes

This converts O(N) unordered surface points into an O(Nx × Ny × Nz) tensor
where QTT's binary tree decomposition exploits spatial coherence.

Dataset: NVIDIA/PhysicsNeMo-CFD-Ahmed-Body (HuggingFace, Apache 2.0)
  - 4,064 parametric RANS simulations of the Ahmed body
  - 7 design variables (Length, Width, Height, GC, SlantAngle, FilletRadius, Velocity)
  - Surface fields: p, k, nut, omega, yPlus, U(3), wallShearStress(3)
  - Re = 1.59M–3.51M

Usage:
    python nvidia_ahmed_body_qtt_pipeline.py
    python nvidia_ahmed_body_qtt_pipeline.py --grid-res 256 --chi 64 --n-samples 50
    python nvidia_ahmed_body_qtt_pipeline.py --skip-download --chi 32 64 128

Author: Brad Adams / Tigantic Holdings LLC
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import time
import json
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# ── VTK (for VTP parsing) ────────────────────────────────────────────
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("[FATAL] VTK required. Install: pip install vtk")
    sys.exit(1)

# ── PyTorch + HyperTensor QTT engine ────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("[FATAL] PyTorch required. Install: pip install torch")
    sys.exit(1)

try:
    from ontic.cfd.qtt import field_to_qtt, qtt_to_field
except ImportError:
    print("[FATAL] HyperTensor QTT engine not found.")
    print("        Ensure tensornet.cfd.qtt is importable.")
    sys.exit(1)

# ── SciPy (for KD-tree spatial queries) ──────────────────────────────
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator


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

    # Grid parameters
    grid_res: int = 128         # Grid cells along longest body axis (x)
    grid_pad: float = 0.05      # Padding around body bounds (meters)

    # QTT compression
    chi_values: List[int] = field(default_factory=lambda: [32, 64, 128])

    # Benchmarking
    n_samples: int = 0          # 0 = all available
    splits: List[str] = field(default_factory=lambda: ["test"])

    # Fields to compress (all scalar channels)
    field_names: List[str] = field(default_factory=lambda: [
        "p", "k", "nut", "omega", "yPlus",
        "U_x", "U_y", "U_z",
        "wallShearStress_x", "wallShearStress_y", "wallShearStress_z",
    ])


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD DATASET
# ═══════════════════════════════════════════════════════════════════════

def download_dataset(config: PipelineConfig) -> Path:
    """Download Ahmed Body dataset from HuggingFace."""
    data_path = Path(config.data_dir)
    existing = list(data_path.rglob("*.vtp"))
    if existing:
        print(f"  Found {len(existing)} existing VTP files — skipping download.")
        return data_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[FATAL] huggingface_hub required: pip install huggingface_hub")
        sys.exit(1)

    print(f"  Downloading from {config.hf_repo}...")
    snapshot_download(
        repo_id=config.hf_repo,
        repo_type="dataset",
        local_dir=str(data_path),
        allow_patterns=["*.vtp", "*.txt"],
    )
    downloaded = list(data_path.rglob("*.vtp"))
    print(f"  Downloaded {len(downloaded)} VTP files.")
    return data_path


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: PARSE VTP FILES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SurfaceSample:
    """Parsed data from one Ahmed Body VTP file."""
    case_id: str
    filepath: str
    n_points: int
    coords: np.ndarray          # (N, 3) float64
    fields: Dict[str, np.ndarray]  # field_name → (N,) float32/float64
    file_size_bytes: int


def parse_vtp(filepath: str) -> SurfaceSample:
    """Parse VTP file using VTK. Extracts all point data fields."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()

    n_points = polydata.GetNumberOfPoints()
    coords = vtk_to_numpy(polydata.GetPoints().GetData()).copy()
    point_data = polydata.GetPointData()

    fields: Dict[str, np.ndarray] = {}

    # Scalar fields
    for name in ["p", "k", "nut", "omega", "yPlus"]:
        arr = point_data.GetArray(name)
        if arr is not None:
            fields[name] = vtk_to_numpy(arr).ravel().copy()

    # Vector fields → split into components
    for vec_name in ["U", "wallShearStress"]:
        arr = point_data.GetArray(vec_name)
        if arr is not None:
            vec = vtk_to_numpy(arr).copy()
            if vec.ndim == 1:
                vec = vec.reshape(-1, 3)
            for c, suffix in enumerate(["x", "y", "z"]):
                fields[f"{vec_name}_{suffix}"] = vec[:, c].copy()

    case_id = Path(filepath).stem
    return SurfaceSample(
        case_id=case_id,
        filepath=filepath,
        n_points=n_points,
        coords=coords,
        fields=fields,
        file_size_bytes=os.path.getsize(filepath),
    )


def discover_vtp_files(config: PipelineConfig) -> List[str]:
    """Find VTP files in the dataset directory."""
    data_path = Path(config.data_dir)
    all_vtps: List[str] = []

    for split in config.splits:
        # Try nested structure: data_dir/dataset/split/*.vtp
        patterns = [
            str(data_path / "dataset" / split / "*.vtp"),
            str(data_path / split / "*.vtp"),
            str(data_path / "**" / "*.vtp"),
        ]
        for pattern in patterns:
            found = sorted(glob.glob(pattern, recursive=True))
            if found:
                all_vtps.extend(found)
                break

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for f in all_vtps:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    if config.n_samples > 0:
        unique = unique[:config.n_samples]

    return unique


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: MESH-TO-RECTILINEAR BRIDGE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StructuredGrid:
    """A structured 3D grid filled from surface data."""
    nx: int                     # Grid dimensions (power-of-2)
    ny: int
    nz: int
    dx: float                   # Cell size (meters)
    origin: Tuple[float, float, float]
    x_nodes: np.ndarray         # (nx+1,) node coordinates
    y_nodes: np.ndarray         # (ny+1,)
    z_nodes: np.ndarray         # (nz+1,)
    n_grid_points: int          # (nx+1)*(ny+1)*(nz+1)
    nn_indices: np.ndarray      # (n_grid_points,) index into surface nodes
    fields_3d: Dict[str, np.ndarray]  # field_name → flat array (n_grid_points,) float32


def voxelize_surface(
    sample: SurfaceSample,
    grid_res: int = 128,
    pad: float = 0.05,
) -> StructuredGrid:
    """
    Mesh-to-Rectilinear Bridge: map unstructured surface data onto a
    structured 3D Cartesian grid.

    Every grid point is assigned the value from its nearest surface node.
    This produces a smooth, continuous 3D field (a Voronoi tessellation
    of the surface data) that QTT can compress efficiently.

    Parameters
    ----------
    sample : SurfaceSample
        Parsed VTP data with coords and fields.
    grid_res : int
        Number of cells along the body's longest axis (x).
    pad : float
        Padding around body bounds in meters.

    Returns
    -------
    StructuredGrid
        The filled structured grid ready for QTT compression.
    """
    coords = sample.coords
    xmin, xmax = coords[:, 0].min() - pad, coords[:, 0].max() + pad
    ymin, ymax = coords[:, 1].min() - pad, coords[:, 1].max() + pad
    zmin, zmax = coords[:, 2].min() - pad, coords[:, 2].max() + pad

    # Cell size from longest axis
    dx = (xmax - xmin) / grid_res

    # Grid dimensions: round up to next power of 2
    def next_pow2(n: int) -> int:
        return max(2, int(2 ** np.ceil(np.log2(max(2, n)))))

    nx = next_pow2(grid_res)
    ny = next_pow2(max(2, int(np.ceil((ymax - ymin) / dx))))
    nz = next_pow2(max(2, int(np.ceil((zmax - zmin) / dx))))

    # Node coordinates
    x_nodes = np.linspace(xmin, xmin + nx * dx, nx + 1)
    y_nodes = np.linspace(ymin, ymin + ny * dx, ny + 1)
    z_nodes = np.linspace(zmin, zmin + nz * dx, nz + 1)

    # Build grid point coordinates
    XX, YY, ZZ = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
    grid_pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    n_grid_points = len(grid_pts)

    # KD-tree: nearest surface node for every grid point
    tree = cKDTree(coords)
    _, nn_indices = tree.query(grid_pts, k=1)

    # Fill grid with field values via nearest-neighbor lookup
    fields_3d: Dict[str, np.ndarray] = {}
    for fname, fdata in sample.fields.items():
        fields_3d[fname] = fdata[nn_indices].astype(np.float32)

    return StructuredGrid(
        nx=nx, ny=ny, nz=nz,
        dx=dx,
        origin=(xmin, ymin, zmin),
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        z_nodes=z_nodes,
        n_grid_points=n_grid_points,
        nn_indices=nn_indices,
        fields_3d=fields_3d,
    )


def reconstruct_to_surface(
    grid: StructuredGrid,
    reconstructed_flat: np.ndarray,
    surface_coords: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct surface-point values from a structured grid field
    via trilinear interpolation.

    Parameters
    ----------
    grid : StructuredGrid
        The structured grid specification.
    reconstructed_flat : np.ndarray
        Flat array of grid values (n_grid_points,).
    surface_coords : np.ndarray
        Original surface mesh coordinates (N_surface, 3).

    Returns
    -------
    np.ndarray
        Reconstructed field values at surface points (N_surface,).
    """
    rec_3d = reconstructed_flat.reshape(grid.nx + 1, grid.ny + 1, grid.nz + 1)
    interp = RegularGridInterpolator(
        (grid.x_nodes, grid.y_nodes, grid.z_nodes),
        rec_3d,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(surface_coords).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: QTT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FieldCompressionResult:
    """Compression metrics for one scalar field on the structured grid."""
    field_name: str
    chi_max: int
    raw_bytes: int              # Grid field size in bytes
    compressed_bytes: int       # QTT cores storage
    compression_ratio: float    # raw_bytes / compressed_bytes
    l2_error_grid: float        # ||f_grid - f_rec||₂ / ||f_grid||₂
    l2_error_surface: float     # Roundtrip error at original surface points
    compress_time_ms: float
    decompress_time_ms: float
    max_bond_dim: int           # Actual achieved bond dimension
    n_cores: int                # Number of TT cores


def compress_field_qtt(
    grid: StructuredGrid,
    field_name: str,
    chi_max: int,
    surface_coords: Optional[np.ndarray] = None,
    surface_field: Optional[np.ndarray] = None,
) -> FieldCompressionResult:
    """
    QTT-compress a single scalar field on the structured grid.

    Uses HyperTensor's field_to_qtt engine (TT-SVD with randomized SVD).

    Parameters
    ----------
    grid : StructuredGrid
        The structured grid with filled fields.
    field_name : str
        Which field to compress.
    chi_max : int
        Maximum bond dimension.
    surface_coords : np.ndarray, optional
        Original surface coordinates for roundtrip error measurement.
    surface_field : np.ndarray, optional
        Original surface field values for roundtrip error measurement.

    Returns
    -------
    FieldCompressionResult
        Compression metrics.
    """
    field_data = grid.fields_3d[field_name]
    raw_bytes = field_data.nbytes
    t_tensor = torch.from_numpy(field_data).float()

    # Compress
    t0 = time.perf_counter()
    qtt_result = field_to_qtt(t_tensor, chi_max=chi_max, tol=1e-14)
    compress_time = (time.perf_counter() - t0) * 1000  # ms

    # Compressed size
    compressed_bytes = sum(
        c.numel() * c.element_size() for c in qtt_result.mps.tensors
    )

    # Decompress
    t0 = time.perf_counter()
    reconstructed = qtt_to_field(qtt_result).numpy()[:len(field_data)]
    decompress_time = (time.perf_counter() - t0) * 1000  # ms

    # Grid-space error
    norm_grid = np.linalg.norm(field_data)
    l2_grid = (np.linalg.norm(field_data - reconstructed) / norm_grid
               if norm_grid > 0 else 0.0)

    # Surface roundtrip error (if original surface data provided)
    l2_surface = 0.0
    if surface_coords is not None and surface_field is not None:
        recovered = reconstruct_to_surface(grid, reconstructed, surface_coords)
        norm_surf = np.linalg.norm(surface_field)
        if norm_surf > 0:
            l2_surface = float(np.linalg.norm(surface_field - recovered) / norm_surf)

    # Bond dimensions
    bond_dims = [c.shape[-1] for c in qtt_result.mps.tensors[:-1]]
    max_bond = max(bond_dims) if bond_dims else 1

    return FieldCompressionResult(
        field_name=field_name,
        chi_max=chi_max,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=raw_bytes / compressed_bytes if compressed_bytes > 0 else float('inf'),
        l2_error_grid=float(l2_grid),
        l2_error_surface=float(l2_surface),
        compress_time_ms=compress_time,
        decompress_time_ms=decompress_time,
        max_bond_dim=max_bond,
        n_cores=len(qtt_result.mps.tensors),
    )


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: FULL SAMPLE BENCHMARK
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SampleBenchmark:
    """Complete benchmark for one Ahmed Body case across all χ values."""
    case_id: str
    n_surface_points: int
    file_size_bytes: int
    grid_dims: Tuple[int, int, int]
    n_grid_points: int
    grid_raw_bytes: int         # Total raw bytes for all fields on grid
    dx_mm: float
    voxelize_time_ms: float

    # Per-χ results: chi → list of FieldCompressionResult
    results_by_chi: Dict[int, List[FieldCompressionResult]] = field(
        default_factory=dict
    )

    def total_compressed_bytes(self, chi: int) -> int:
        return sum(r.compressed_bytes for r in self.results_by_chi.get(chi, []))

    def overall_ratio(self, chi: int) -> float:
        cb = self.total_compressed_bytes(chi)
        return self.grid_raw_bytes / cb if cb > 0 else float('inf')

    def max_l2_grid(self, chi: int) -> float:
        results = self.results_by_chi.get(chi, [])
        # Exclude all-zero fields (velocity at wall surface)
        physical = [r for r in results if r.l2_error_grid > 0 or r.raw_bytes > 640]
        return max((r.l2_error_grid for r in physical), default=0.0)

    def mean_l2_grid(self, chi: int) -> float:
        results = self.results_by_chi.get(chi, [])
        physical = [r for r in results if r.l2_error_grid > 0 or r.raw_bytes > 640]
        if not physical:
            return 0.0
        return float(np.mean([r.l2_error_grid for r in physical]))


def benchmark_sample(
    vtp_path: str,
    config: PipelineConfig,
) -> SampleBenchmark:
    """
    Full benchmark pipeline for one VTP sample:
    parse → voxelize → compress at each χ → measure errors.
    """
    # Parse
    sample = parse_vtp(vtp_path)

    # Voxelize
    t0 = time.perf_counter()
    grid = voxelize_surface(sample, grid_res=config.grid_res, pad=config.grid_pad)
    voxelize_time = (time.perf_counter() - t0) * 1000

    grid_raw_bytes = sum(f.nbytes for f in grid.fields_3d.values())

    bench = SampleBenchmark(
        case_id=sample.case_id,
        n_surface_points=sample.n_points,
        file_size_bytes=sample.file_size_bytes,
        grid_dims=(grid.nx, grid.ny, grid.nz),
        n_grid_points=grid.n_grid_points,
        grid_raw_bytes=grid_raw_bytes,
        dx_mm=grid.dx * 1000,
        voxelize_time_ms=voxelize_time,
    )

    # Compress each field at each χ
    for chi in config.chi_values:
        results: List[FieldCompressionResult] = []
        for fname in grid.fields_3d:
            surf_field = sample.fields.get(fname)
            result = compress_field_qtt(
                grid, fname, chi,
                surface_coords=sample.coords if surf_field is not None else None,
                surface_field=surf_field,
            )
            results.append(result)
        bench.results_by_chi[chi] = results

    return bench


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: REPORTING
# ═══════════════════════════════════════════════════════════════════════

def print_sample_summary(bench: SampleBenchmark) -> None:
    """Print compact summary for one sample."""
    chi_summaries = []
    for chi in sorted(bench.results_by_chi.keys()):
        ratio = bench.overall_ratio(chi)
        l2 = bench.mean_l2_grid(chi)
        cb = bench.total_compressed_bytes(chi)
        chi_summaries.append(f"χ{chi}={ratio:.0f}×")
    parts = " | ".join(chi_summaries)
    print(f"  {bench.case_id:>10}: {bench.n_surface_points:>6,}pts "
          f"grid={bench.grid_dims[0]}×{bench.grid_dims[1]}×{bench.grid_dims[2]} "
          f"| {parts}")


def generate_report(
    benchmarks: List[SampleBenchmark],
    config: PipelineConfig,
) -> str:
    """Generate comprehensive benchmark report."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    sep = "═" * 72

    lines.append(sep)
    lines.append("QTT COMPRESSION BENCHMARK: NVIDIA PhysicsNeMo Ahmed Body")
    lines.append(f"Mesh-to-Rectilinear Bridge — HyperTensor QTT Engine")
    lines.append(sep)
    lines.append(f"Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Samples:    {len(benchmarks)}")
    lines.append(f"Grid res:   {config.grid_res} (cells along x-axis)")
    lines.append(f"χ values:   {config.chi_values}")
    lines.append(f"Device:     {DEVICE}")
    lines.append(f"Padding:    {config.grid_pad*100:.0f} cm")
    lines.append("")

    # Grid info from first sample
    if benchmarks:
        b0 = benchmarks[0]
        lines.append(f"Grid:       {b0.grid_dims[0]}×{b0.grid_dims[1]}×{b0.grid_dims[2]}"
                      f" = {b0.n_grid_points:,} nodes")
        lines.append(f"dx:         {b0.dx_mm:.1f} mm")
        lines.append(f"Fields:     {len(b0.results_by_chi.get(config.chi_values[0], []))} scalar channels")
        lines.append("")

    lines.append("─" * 72)
    lines.append("AGGREGATE RESULTS PER χ (GRID-SPACE METRICS)")
    lines.append("─" * 72)

    for chi in config.chi_values:
        ratios = [b.overall_ratio(chi) for b in benchmarks]
        l2s = [b.mean_l2_grid(chi) for b in benchmarks]
        max_l2s = [b.max_l2_grid(chi) for b in benchmarks]
        compressed = [b.total_compressed_bytes(chi) for b in benchmarks]
        grid_raw = [b.grid_raw_bytes for b in benchmarks]

        lines.append(f"")
        lines.append(f"  χ = {chi}")
        lines.append(f"    Grid compression ratio:")
        lines.append(f"      Mean:   {np.mean(ratios):>8.0f}×")
        lines.append(f"      Median: {np.median(ratios):>8.0f}×")
        lines.append(f"      Min:    {np.min(ratios):>8.0f}×")
        lines.append(f"      Max:    {np.max(ratios):>8.0f}×")
        lines.append(f"    Grid L2 error (mean across fields):")
        lines.append(f"      Mean:   {np.mean(l2s):.2e}")
        lines.append(f"      Max:    {np.max(l2s):.2e}")
        lines.append(f"    Grid L2 error (worst single field):")
        lines.append(f"      Max:    {np.max(max_l2s):.2e}")
        lines.append(f"    Storage:")
        lines.append(f"      Grid raw:    {np.mean(grid_raw)/1e6:.1f} MB avg per sample")
        lines.append(f"      QTT:         {np.mean(compressed)/1e3:.1f} KB avg per sample")
        lines.append(f"      VTP file:    {np.mean([b.file_size_bytes for b in benchmarks])/1e6:.1f} MB avg")
        lines.append(f"      vs VTP:      {np.mean([b.file_size_bytes/b.total_compressed_bytes(chi) for b in benchmarks]):.0f}×")

    # Per-field breakdown at best χ
    lines.append("")
    lines.append("─" * 72)
    lines.append("PER-FIELD BREAKDOWN (BEST χ)")
    lines.append("─" * 72)

    best_chi = config.chi_values[0]  # Highest compression = lowest χ
    field_stats: Dict[str, List[Tuple[float, float]]] = {}
    for b in benchmarks:
        for r in b.results_by_chi.get(best_chi, []):
            field_stats.setdefault(r.field_name, []).append(
                (r.compression_ratio, r.l2_error_grid)
            )

    lines.append(f"  χ = {best_chi}")
    lines.append(f"  {'Field':>22} | {'Ratio':>8} | {'L2 Grid':>10} |")
    lines.append(f"  {'-'*22}-+-{'-'*8}-+-{'-'*10}-+")
    for fname, stats in sorted(field_stats.items()):
        mean_ratio = np.mean([s[0] for s in stats])
        mean_l2 = np.mean([s[1] for s in stats])
        lines.append(f"  {fname:>22} | {mean_ratio:>7.0f}× | {mean_l2:>10.2e} |")

    # NVIDIA comparison targets
    lines.append("")
    lines.append("─" * 72)
    lines.append("vs NVIDIA PhysicsNeMo TARGETS")
    lines.append("─" * 72)

    for chi in config.chi_values:
        mean_ratio = np.mean([b.overall_ratio(chi) for b in benchmarks])
        max_l2 = np.max([b.max_l2_grid(chi) for b in benchmarks])
        mean_vtp_ratio = np.mean([
            b.file_size_bytes / b.total_compressed_bytes(chi) for b in benchmarks
        ])

        lines.append(f"")
        lines.append(f"  χ = {chi}:")

        targets = [
            ("Grid compression > 10×", mean_ratio > 10, f"{mean_ratio:.0f}×"),
            ("Grid compression > 20×", mean_ratio > 20, f"{mean_ratio:.0f}×"),
            ("Grid compression > 50×", mean_ratio > 50, f"{mean_ratio:.0f}×"),
            ("vs VTP file > 10×", mean_vtp_ratio > 10, f"{mean_vtp_ratio:.0f}×"),
            ("Grid L2 error < 10%", max_l2 < 0.10, f"{max_l2:.2e}"),
            ("Grid L2 error < 5%", max_l2 < 0.05, f"{max_l2:.2e}"),
            ("Grid L2 error < 1%", max_l2 < 0.01, f"{max_l2:.2e}"),
        ]
        for desc, passed, value in targets:
            status = "PASS" if passed else "MISS"
            lines.append(f"    [{status}] {desc}  [{value}]")

    # Dataset projection
    lines.append("")
    lines.append("─" * 72)
    lines.append("FULL DATASET PROJECTION (4,064 samples)")
    lines.append("─" * 72)

    for chi in config.chi_values:
        mean_compressed = np.mean([b.total_compressed_bytes(chi) for b in benchmarks])
        total_projected = mean_compressed * 4064
        mean_vtp = np.mean([b.file_size_bytes for b in benchmarks])
        total_vtp = mean_vtp * 4064

        lines.append(f"  χ = {chi}:")
        lines.append(f"    Original VTP:  {total_vtp/1e9:.1f} GB")
        lines.append(f"    QTT compressed: {total_projected/1e6:.0f} MB")
        lines.append(f"    Reduction:      {total_vtp/total_projected:.0f}×")

    # Timing
    lines.append("")
    lines.append("─" * 72)
    lines.append("TIMING")
    lines.append("─" * 72)
    vox_times = [b.voxelize_time_ms for b in benchmarks]
    lines.append(f"  Voxelization: {np.mean(vox_times):.0f} ms avg per sample")

    for chi in config.chi_values:
        comp_times: List[float] = []
        decomp_times: List[float] = []
        for b in benchmarks:
            for r in b.results_by_chi.get(chi, []):
                comp_times.append(r.compress_time_ms)
                decomp_times.append(r.decompress_time_ms)
        if comp_times:
            lines.append(f"  QTT χ={chi}:")
            lines.append(f"    Compress:   {np.mean(comp_times):.1f} ms avg per field")
            lines.append(f"    Decompress: {np.mean(decomp_times):.1f} ms avg per field")

    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)

    # Save outputs
    report_path = results_dir / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # JSON with full per-sample detail
    json_data: Dict[str, Any] = {
        "config": {
            "grid_res": config.grid_res,
            "chi_values": config.chi_values,
            "n_samples": len(benchmarks),
            "device": DEVICE,
        },
        "samples": [],
    }

    for b in benchmarks:
        sample_dict: Dict[str, Any] = {
            "case_id": b.case_id,
            "n_surface_points": b.n_surface_points,
            "file_size_bytes": b.file_size_bytes,
            "grid_dims": list(b.grid_dims),
            "n_grid_points": b.n_grid_points,
            "grid_raw_bytes": b.grid_raw_bytes,
            "dx_mm": b.dx_mm,
            "voxelize_time_ms": b.voxelize_time_ms,
            "chi_results": {},
        }
        for chi in config.chi_values:
            chi_dict: Dict[str, Any] = {
                "total_compressed_bytes": b.total_compressed_bytes(chi),
                "overall_ratio": b.overall_ratio(chi),
                "mean_l2_grid": b.mean_l2_grid(chi),
                "max_l2_grid": b.max_l2_grid(chi),
                "fields": [
                    {
                        "field": r.field_name,
                        "compressed_bytes": r.compressed_bytes,
                        "ratio": r.compression_ratio,
                        "l2_grid": r.l2_error_grid,
                        "l2_surface": r.l2_error_surface,
                        "max_bond_dim": r.max_bond_dim,
                        "compress_ms": r.compress_time_ms,
                        "decompress_ms": r.decompress_time_ms,
                    }
                    for r in b.results_by_chi.get(chi, [])
                ],
            }
            sample_dict["chi_results"][str(chi)] = chi_dict
        json_data["samples"].append(sample_dict)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = results_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)

    # CSV summary
    csv_path = results_dir / "benchmark_results.csv"
    with open(csv_path, "w") as f:
        header = "case_id,n_surface_points,file_size_bytes,grid_dims,n_grid_points,dx_mm"
        for chi in config.chi_values:
            header += f",chi{chi}_compressed_bytes,chi{chi}_ratio,chi{chi}_mean_l2"
        f.write(header + "\n")
        for b in benchmarks:
            row = (f"{b.case_id},{b.n_surface_points},{b.file_size_bytes},"
                   f"{b.grid_dims[0]}x{b.grid_dims[1]}x{b.grid_dims[2]},"
                   f"{b.n_grid_points},{b.dx_mm:.1f}")
            for chi in config.chi_values:
                row += (f",{b.total_compressed_bytes(chi)}"
                        f",{b.overall_ratio(chi):.1f}"
                        f",{b.mean_l2_grid(chi):.4e}")
            f.write(row + "\n")

    print(f"  Reports saved to {results_dir}/")
    print(f"    benchmark_report.txt")
    print(f"    benchmark_results.json")
    print(f"    benchmark_results.csv")

    return report_text


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVIDIA PhysicsNeMo Ahmed Body → QTT Compression Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nvidia_ahmed_body_qtt_pipeline.py
  python nvidia_ahmed_body_qtt_pipeline.py --grid-res 256 --chi 64
  python nvidia_ahmed_body_qtt_pipeline.py --n-samples 10 --chi 16 32 64 128
        """,
    )
    parser.add_argument("--grid-res", type=int, default=128,
                        help="Grid cells along x-axis (default: 128)")
    parser.add_argument("--chi", type=int, nargs="+", default=[32, 64, 128],
                        help="Bond dimension values to test (default: 32 64 128)")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Max samples to process (0=all, default: 0)")
    parser.add_argument("--data-dir", default="./ahmed_body_data",
                        help="Dataset directory")
    parser.add_argument("--results-dir", default="./ahmed_body_results",
                        help="Results output directory")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Dataset splits (default: test)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download step")
    parser.add_argument("--pad", type=float, default=0.05,
                        help="Grid padding in meters (default: 0.05)")
    args = parser.parse_args()

    config = PipelineConfig(
        grid_res=args.grid_res,
        chi_values=sorted(args.chi),
        n_samples=args.n_samples,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        splits=args.splits,
        grid_pad=args.pad,
    )

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  QTT × NVIDIA PhysicsNeMo Ahmed Body Compression Benchmark    ║")
    print("║  Mesh-to-Rectilinear Bridge — HyperTensor QTT Engine          ║")
    print("║  Tigantic Holdings LLC — Brad Adams                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Grid:      {config.grid_res} cells along x → power-of-2 structured grid")
    print(f"  χ values:  {config.chi_values}")
    print(f"  Device:    {DEVICE}")
    print(f"  Samples:   {config.n_samples if config.n_samples > 0 else 'ALL'}")
    print()

    # Step 1: Download
    if not args.skip_download:
        print("─" * 72)
        print("STEP 1: DATASET")
        print("─" * 72)
        download_dataset(config)

    # Step 2: Discover VTP files
    print()
    print("─" * 72)
    print("STEP 2: DISCOVERING VTP FILES")
    print("─" * 72)
    vtp_files = discover_vtp_files(config)
    print(f"  Found {len(vtp_files)} VTP files to process.")

    if not vtp_files:
        print("[FATAL] No VTP files found. Check --data-dir path.")
        sys.exit(1)

    # Steps 3-5: Voxelize + compress + benchmark each sample
    print()
    print("─" * 72)
    print("STEP 3: MESH-TO-RECTILINEAR + QTT COMPRESSION")
    print("─" * 72)
    print()

    benchmarks: List[SampleBenchmark] = []
    t_total = time.perf_counter()

    for i, vtp_path in enumerate(vtp_files):
        case_name = Path(vtp_path).stem
        print(f"  [{i+1}/{len(vtp_files)}] {case_name}...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            bench = benchmark_sample(vtp_path, config)
            elapsed = time.perf_counter() - t0
            benchmarks.append(bench)
            # Compact one-line summary
            parts = []
            for chi in config.chi_values:
                r = bench.overall_ratio(chi)
                parts.append(f"χ{chi}={r:.0f}×")
            print(f"{bench.n_surface_points:>6,}pts → "
                  f"{bench.grid_dims[0]}×{bench.grid_dims[1]}×{bench.grid_dims[2]} "
                  f"| {' '.join(parts)} "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

    total_time = time.perf_counter() - t_total
    print(f"\n  Total: {len(benchmarks)} samples in {total_time:.1f}s "
          f"({total_time/len(benchmarks):.1f}s/sample)")

    # Step 6: Report
    print()
    print("─" * 72)
    print("STEP 4: GENERATING REPORT")
    print("─" * 72)

    report = generate_report(benchmarks, config)
    print()
    print(report)


if __name__ == "__main__":
    main()
