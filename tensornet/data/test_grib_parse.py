#!/usr/bin/env python3
"""
GRIB2 Parser & Tensor Extractor - Global Eye Phase 1A-2/3
==========================================================

Parses HRRR GRIB2 files and extracts U/V wind components as Float32 tensors.

Exit Gates:
  - 1A-2: Prints "[SUCCESS] Found wind components: u10, v10"
  - 1A-3: Prints "[READY] Tensors prepared"

Usage:
    python test_grib_parse.py
    python test_grib_parse.py --file path/to/custom.grib2
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Check for cfgrib/xarray
try:
    import xarray as xr
except ImportError:
    print("[ERROR] xarray not installed. Run: pip install xarray cfgrib")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent / "cache" / "hrrr"
DEFAULT_GRIB = CACHE_DIR / "hrrr.t12z.wrfsfcf01.grib2"


def find_latest_grib() -> Path:
    """Find the most recently modified GRIB2 file in cache."""
    grib_files = list(CACHE_DIR.glob("*.grib2"))
    if not grib_files:
        return None
    return max(grib_files, key=lambda p: p.stat().st_mtime)


def try_open_grib(filepath: Path, filter_config: dict) -> "xr.Dataset":
    """Attempt to open GRIB with specific filter configuration."""
    return xr.open_dataset(
        filepath, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_config}
    )


def inspect_and_extract(filepath: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Parse GRIB2 file and extract U/V wind tensors.

    Args:
        filepath: Path to .grib2 file

    Returns:
        Tuple of (u_tensor, v_tensor) as Float32, or None on failure
    """
    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return None

    print(f"[LOADING] Opening {filepath.name}...")
    print(f"[INFO] File size: {filepath.stat().st_size / (1024*1024):.1f} MB")

    # ───────────────────────────────────────────────────────────────────────────
    # Strategy: Try multiple filter configurations
    # HRRR wind data can be at different levels depending on the product
    # ───────────────────────────────────────────────────────────────────────────

    filter_strategies = [
        # Strategy 1: 10m height above ground (most common for surface winds)
        {
            "name": "10m height",
            "filter": {"typeOfLevel": "heightAboveGround", "level": 10},
        },
        # Strategy 2: Surface level
        {
            "name": "surface",
            "filter": {"typeOfLevel": "surface", "stepType": "instant"},
        },
        # Strategy 3: Any instant data
        {"name": "instant", "filter": {"stepType": "instant"}},
    ]

    ds = None
    for strategy in filter_strategies:
        try:
            print(f"[TRYING] Filter: {strategy['name']}")
            ds = try_open_grib(filepath, strategy["filter"])
            print(f"[SUCCESS] Opened with '{strategy['name']}' filter")
            break
        except Exception as e:
            print(f"[SKIP] {strategy['name']}: {str(e)[:60]}...")
            continue

    if ds is None:
        print("[FATAL] Could not parse GRIB with any filter strategy")
        return None

    # ───────────────────────────────────────────────────────────────────────────
    # Find U and V wind components
    # ───────────────────────────────────────────────────────────────────────────

    print(f"\n[DATASET] Variables found: {list(ds.data_vars)}")

    # Common variable names for 10m wind
    u_candidates = ["u10", "u", "U component of wind", "10u", "ugrd"]
    v_candidates = ["v10", "v", "V component of wind", "10v", "vgrd"]

    u_var = None
    v_var = None

    for var in ds.data_vars:
        var_lower = var.lower()
        if u_var is None and any(c.lower() in var_lower for c in u_candidates):
            u_var = var
        if v_var is None and any(c.lower() in var_lower for c in v_candidates):
            v_var = var

    # Fallback: look for anything with 'u' or 'v'
    if u_var is None or v_var is None:
        for var in ds.data_vars:
            if u_var is None and "u" in var.lower() and "v" not in var.lower():
                u_var = var
            elif v_var is None and "v" in var.lower() and "u" not in var.lower():
                v_var = var

    if u_var is None or v_var is None:
        print("[ERROR] Could not find U/V wind components")
        print(f"        Available: {list(ds.data_vars)}")
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # EXIT GATE 1A-2: Found wind components
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[SUCCESS] Found wind components: {u_var}, {v_var}")

    # ───────────────────────────────────────────────────────────────────────────
    # Extract and validate data
    # ───────────────────────────────────────────────────────────────────────────

    print("\n[EXTRACTING] Converting to numpy arrays...")

    u_data = ds[u_var].values
    v_data = ds[v_var].values

    print(f"[INFO] Raw shape: {u_data.shape}")
    print(f"[INFO] Dtype: {u_data.dtype}")
    print(f"[INFO] U range: [{np.nanmin(u_data):.2f}, {np.nanmax(u_data):.2f}] m/s")
    print(f"[INFO] V range: [{np.nanmin(v_data):.2f}, {np.nanmax(v_data):.2f}] m/s")

    # ───────────────────────────────────────────────────────────────────────────
    # Task 1A-3: Tensor Normalization
    # ───────────────────────────────────────────────────────────────────────────

    # Handle NaNs (common at domain boundaries)
    nan_count = np.isnan(u_data).sum()
    if nan_count > 0:
        print(
            f"[WARNING] Found {nan_count} NaN values ({100*nan_count/u_data.size:.1f}%)"
        )
        print("[FIX] Replacing NaNs with 0.0...")
        u_data = np.nan_to_num(u_data, nan=0.0)
        v_data = np.nan_to_num(v_data, nan=0.0)

    # Ensure Float32 for Rust bridge
    u_f32 = u_data.astype(np.float32)
    v_f32 = v_data.astype(np.float32)

    # Ensure 2D shape [H, W]
    if u_f32.ndim == 3:
        print(f"[RESHAPE] Squeezing from {u_f32.shape} to 2D")
        u_f32 = u_f32.squeeze()
        v_f32 = v_f32.squeeze()

    # Compute magnitude for verification
    magnitude = np.sqrt(u_f32**2 + v_f32**2)
    max_speed = float(np.max(magnitude))

    print(f"\n[STATS] Final tensor shape: {u_f32.shape}")
    print(f"[STATS] Memory per tensor: {u_f32.nbytes / (1024*1024):.2f} MB")
    print(f"[STATS] Max wind speed: {max_speed:.1f} m/s")

    # ═══════════════════════════════════════════════════════════════════════════
    # EXIT GATE 1A-3: Tensors prepared
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[READY] Tensors prepared for Bridge.")

    return u_f32, v_f32


def main():
    parser = argparse.ArgumentParser(
        description="Parse HRRR GRIB2 and extract wind tensors"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to GRIB2 file. Default: latest in cache",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GRIB2 Parser - Global Eye Phase 1A-2/3")
    print("=" * 60)

    # Find file to parse
    if args.file:
        filepath = Path(args.file)
    else:
        filepath = find_latest_grib()
        if filepath is None:
            print("[ERROR] No GRIB2 files in cache. Run hrrr_fetcher.py first.")
            sys.exit(1)

    result = inspect_and_extract(filepath)

    if result is None:
        print("\n[FAILED] Could not extract wind data")
        sys.exit(1)

    u_tensor, v_tensor = result

    # Save tensors for manual inspection (optional)
    output_dir = CACHE_DIR / "tensors"
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "u_wind.npy", u_tensor)
    np.save(output_dir / "v_wind.npy", v_tensor)
    print(f"\n[SAVED] Tensors saved to {output_dir}/")

    print("\n" + "=" * 60)
    print("  ✓ EXIT GATE 1A-2: Found wind components")
    print("  ✓ EXIT GATE 1A-3: Tensors prepared")
    print("=" * 60)


if __name__ == "__main__":
    main()
