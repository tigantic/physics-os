#!/usr/bin/env python3
"""
Build script for Ontic Rust-Python extensions.

Usage:
    python build_extensions.py [--release]
    
This builds:
    - tci_core: Rust TCI via PyO3
    - ontic_gpu_py: CUDA pipeline via PyO3
"""

import subprocess
import sys
import os
from pathlib import Path

# Project root
ROOT = Path(__file__).parent


def run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run command and stream output."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def build_extension(crate_name: str, release: bool = True) -> bool:
    """Build a single Rust-Python extension."""
    crate_dir = ROOT / "crates" / crate_name
    
    if not crate_dir.exists():
        print(f"  ❌ Crate not found: {crate_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Building {crate_name}")
    print('='*60)
    
    # Use maturin to build the extension
    cmd = ["maturin", "develop"]
    if release:
        cmd.append("--release")
    
    result = run(cmd, cwd=crate_dir)
    
    if result == 0:
        print(f"  ✅ {crate_name} built successfully")
        return True
    else:
        print(f"  ❌ {crate_name} build failed")
        return False


def check_maturin() -> bool:
    """Check if maturin is installed."""
    result = subprocess.run(
        ["maturin", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ maturin not found. Install with: pip install maturin")
        return False
    print(f"✅ {result.stdout.strip()}")
    return True


def main():
    release = "--release" in sys.argv or "-r" in sys.argv
    
    print("="*60)
    print("Ontic Extension Builder")
    print("="*60)
    
    # Check prerequisites
    if not check_maturin():
        sys.exit(1)
    
    # Extensions to build
    extensions = [
        "tci_core",      # Rust TCI
        "ontic_gpu_py",  # CUDA pipeline
    ]
    
    results = {}
    for ext in extensions:
        results[ext] = build_extension(ext, release)
    
    # Summary
    print(f"\n{'='*60}")
    print("Build Summary")
    print('='*60)
    
    all_success = True
    for ext, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {ext}")
        all_success = all_success and success
    
    if all_success:
        print("\n✅ All extensions built successfully!")
        print("\nUsage in Python:")
        print("  from tci_core import TCISampler, RUST_AVAILABLE")
        print("  from ontic_gpu_py import CudaTTEvaluator, CUDA_AVAILABLE")
    else:
        print("\n⚠️ Some extensions failed to build.")
        print("Check that Rust, CUDA toolkit, and Python dev headers are installed.")
    
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
