#!/usr/bin/env python3
# Copyright 2025 Tigantic Labs. All Rights Reserved.
"""
HyperTensor Build Script

Builds distribution packages for various platforms and formats.

Usage:
    python build.py wheel       # Build Python wheel
    python build.py conda       # Build Conda package
    python build.py docker      # Build Docker images
    python build.py all         # Build all formats
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
SDK_DIR = ROOT_DIR / "sdk"
DIST_DIR = ROOT_DIR / "dist"


def run_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT_DIR, env={**os.environ, **(env or {})})
    return result.returncode


def clean() -> None:
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    
    dirs_to_clean = [
        ROOT_DIR / "build",
        ROOT_DIR / "dist",
        ROOT_DIR / "tensornet.egg-info",
        ROOT_DIR / ".pytest_cache",
    ]
    
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed: {d}")


def build_wheel() -> int:
    """Build Python wheel package."""
    print("\n" + "=" * 60)
    print("Building Python Wheel")
    print("=" * 60)
    
    # Ensure build tools are installed
    run_command([sys.executable, "-m", "pip", "install", "build", "wheel"])
    
    # Build wheel
    result = run_command([sys.executable, "-m", "build", "--wheel"])
    
    if result == 0:
        wheels = list((ROOT_DIR / "dist").glob("*.whl"))
        for wheel in wheels:
            print(f"  Built: {wheel}")
    
    return result


def build_sdist() -> int:
    """Build source distribution."""
    print("\n" + "=" * 60)
    print("Building Source Distribution")
    print("=" * 60)
    
    result = run_command([sys.executable, "-m", "build", "--sdist"])
    
    if result == 0:
        tarballs = list((ROOT_DIR / "dist").glob("*.tar.gz"))
        for tarball in tarballs:
            print(f"  Built: {tarball}")
    
    return result


def build_conda() -> int:
    """Build Conda package."""
    print("\n" + "=" * 60)
    print("Building Conda Package")
    print("=" * 60)
    
    conda_dir = SDK_DIR / "conda"
    
    # Check if conda-build is available
    result = run_command(["conda", "build", "--version"])
    if result != 0:
        print("Error: conda-build not installed. Install with: conda install conda-build")
        return 1
    
    # Build package
    result = run_command([
        "conda", "build",
        str(conda_dir),
        "--output-folder", str(DIST_DIR / "conda")
    ])
    
    return result


def build_docker(target: str = "base") -> int:
    """Build Docker image."""
    print("\n" + "=" * 60)
    print(f"Building Docker Image: {target}")
    print("=" * 60)
    
    dockerfile = SDK_DIR / "docker" / "Dockerfile"
    
    tag = "tigantic/hypertensor"
    if target != "base":
        tag = f"{tag}:{target}"
    else:
        tag = f"{tag}:latest"
    
    result = run_command([
        "docker", "build",
        "-f", str(dockerfile),
        "--target", target,
        "-t", tag,
        str(ROOT_DIR)
    ])
    
    if result == 0:
        print(f"  Built: {tag}")
    
    return result


def build_all_docker() -> int:
    """Build all Docker images."""
    targets = ["base", "cuda", "jupyter", "server"]
    
    for target in targets:
        result = build_docker(target)
        if result != 0:
            return result
    
    return 0


def run_tests() -> int:
    """Run test suite."""
    print("\n" + "=" * 60)
    print("Running Tests")
    print("=" * 60)
    
    return run_command([
        sys.executable, "-m", "pytest",
        "tests/", "-v", "--tb=short"
    ])


def publish_pypi(test: bool = True) -> int:
    """Publish to PyPI."""
    print("\n" + "=" * 60)
    print(f"Publishing to {'TestPyPI' if test else 'PyPI'}")
    print("=" * 60)
    
    # Ensure twine is installed
    run_command([sys.executable, "-m", "pip", "install", "twine"])
    
    repository = "testpypi" if test else "pypi"
    
    return run_command([
        sys.executable, "-m", "twine", "upload",
        "--repository", repository,
        str(DIST_DIR / "*.whl"),
        str(DIST_DIR / "*.tar.gz")
    ])


def main():
    parser = argparse.ArgumentParser(description="HyperTensor Build Script")
    parser.add_argument(
        "target",
        choices=["clean", "wheel", "sdist", "conda", "docker", "all", "test", "publish"],
        help="Build target"
    )
    parser.add_argument(
        "--docker-target",
        default="base",
        choices=["base", "cuda", "jupyter", "server", "all"],
        help="Docker image target"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip tests before publishing"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Publish to production PyPI (default: TestPyPI)"
    )
    
    args = parser.parse_args()
    
    # Ensure dist directory exists
    DIST_DIR.mkdir(exist_ok=True)
    
    if args.target == "clean":
        clean()
        return 0
    
    elif args.target == "wheel":
        return build_wheel()
    
    elif args.target == "sdist":
        return build_sdist()
    
    elif args.target == "conda":
        return build_conda()
    
    elif args.target == "docker":
        if args.docker_target == "all":
            return build_all_docker()
        return build_docker(args.docker_target)
    
    elif args.target == "all":
        results = []
        results.append(("wheel", build_wheel()))
        results.append(("sdist", build_sdist()))
        results.append(("docker", build_all_docker()))
        
        print("\n" + "=" * 60)
        print("Build Summary")
        print("=" * 60)
        for name, code in results:
            status = "✓" if code == 0 else "✗"
            print(f"  {status} {name}")
        
        return max(r[1] for r in results)
    
    elif args.target == "test":
        return run_tests()
    
    elif args.target == "publish":
        if not args.no_test:
            result = run_tests()
            if result != 0:
                print("Tests failed. Use --no-test to skip.")
                return result
        
        # Build first
        build_wheel()
        build_sdist()
        
        return publish_pypi(test=not args.production)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
