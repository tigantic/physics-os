#!/usr/bin/env python3
"""
HRRR Data Fetcher - Global Eye Phase 1A-1
==========================================

Downloads High-Resolution Rapid Refresh (HRRR) GRIB2 files from NOAA's
public S3 bucket (noaa-hrrr-bdp-pds).

Exit Gate: .grib2 file appears in ./cache/hrrr/ folder.

Usage:
    python hrrr_fetcher.py
    python hrrr_fetcher.py --hour 12 --forecast 1
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# AWS S3 access (no credentials needed - public bucket)
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError:
    print("[ERROR] boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BUCKET_NAME = "noaa-hrrr-bdp-pds"
CACHE_DIR = Path(__file__).parent / "cache" / "hrrr"

# HRRR produces forecasts every hour, with forecast hours 0-18 (or 48 for 00z/12z)
# We want the "analysis" (f00) or short forecast (f01) for near-real-time data
DEFAULT_FORECAST_HOUR = 1  # f01 = 1 hour forecast


def get_latest_available_cycle() -> tuple[datetime, int]:
    """
    HRRR data becomes available ~50 minutes after the cycle time.
    Returns (cycle_date, cycle_hour) for the most recent available data.
    """
    now = datetime.now(timezone.utc)
    
    # HRRR cycles run every hour (00z, 01z, ..., 23z)
    # Data is typically available ~50 min after cycle start
    # To be safe, look for data from 2 hours ago
    target = now - timedelta(hours=2)
    
    return target.replace(minute=0, second=0, microsecond=0), target.hour


def build_s3_key(cycle_date: datetime, cycle_hour: int, forecast_hour: int) -> str:
    """
    Build the S3 object key for HRRR CONUS surface data.
    
    Format: hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2
    
    Args:
        cycle_date: The date of the model run
        cycle_hour: Hour of the model run (0-23)
        forecast_hour: Forecast lead time (0-18, or 0-48 for 00z/12z)
    """
    date_str = cycle_date.strftime("%Y%m%d")
    return f"hrrr.{date_str}/conus/hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"


def download_hrrr(
    cycle_date: datetime = None,
    cycle_hour: int = None,
    forecast_hour: int = DEFAULT_FORECAST_HOUR,
    force: bool = False
) -> Path:
    """
    Download HRRR GRIB2 file from NOAA S3.
    
    Args:
        cycle_date: Date of model run (default: latest available)
        cycle_hour: Hour of model run (default: latest available)
        forecast_hour: Forecast lead time
        force: Re-download even if cached
        
    Returns:
        Path to the downloaded file
    """
    # Determine which cycle to fetch
    if cycle_date is None or cycle_hour is None:
        auto_date, auto_hour = get_latest_available_cycle()
        cycle_date = cycle_date or auto_date
        cycle_hour = cycle_hour if cycle_hour is not None else auto_hour
    
    # Build paths
    s3_key = build_s3_key(cycle_date, cycle_hour, forecast_hour)
    local_filename = f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
    local_path = CACHE_DIR / local_filename
    
    print(f"[HRRR] Target: s3://{BUCKET_NAME}/{s3_key}")
    print(f"[HRRR] Local:  {local_path}")
    
    # Check cache
    if local_path.exists() and not force:
        print(f"[CACHE] Using cached file (use --force to re-download)")
        return local_path
    
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create unsigned S3 client (public bucket)
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Download
    print(f"[DOWNLOAD] Fetching from AWS S3...")
    try:
        s3.download_file(BUCKET_NAME, s3_key, str(local_path))
        file_size = local_path.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Downloaded {file_size:.1f} MB → {local_path.name}")
        return local_path
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        
        # Try previous hour if latest isn't available yet
        if cycle_hour > 0:
            print(f"[RETRY] Trying previous cycle (hour {cycle_hour - 1})...")
            return download_hrrr(
                cycle_date=cycle_date,
                cycle_hour=cycle_hour - 1,
                forecast_hour=forecast_hour,
                force=force
            )
        else:
            print(f"[FATAL] No available HRRR data found")
            sys.exit(1)


def list_available_cycles(limit: int = 5):
    """List recent available HRRR cycles on S3."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # List date folders
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="hrrr.",
        Delimiter="/"
    )
    
    prefixes = sorted(
        [p['Prefix'] for p in response.get('CommonPrefixes', [])],
        reverse=True
    )[:limit]
    
    print(f"\n[AVAILABLE] Recent HRRR cycles:")
    for prefix in prefixes:
        print(f"  • {prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HRRR GRIB2 data from NOAA S3"
    )
    parser.add_argument(
        "--hour", type=int, default=None,
        help="Model cycle hour (0-23). Default: latest available"
    )
    parser.add_argument(
        "--forecast", type=int, default=DEFAULT_FORECAST_HOUR,
        help=f"Forecast hour (0-18). Default: {DEFAULT_FORECAST_HOUR}"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available cycles and exit"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  HRRR Data Fetcher - Global Eye Phase 1A")
    print("=" * 60)
    
    if args.list:
        list_available_cycles()
        return
    
    filepath = download_hrrr(
        cycle_hour=args.hour,
        forecast_hour=args.forecast,
        force=args.force
    )
    
    print(f"\n[EXIT GATE] ✓ GRIB2 file ready: {filepath}")


if __name__ == "__main__":
    main()
