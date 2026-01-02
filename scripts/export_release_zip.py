#!/usr/bin/env python
"""
A) Release Export Script
========================

Creates a clean release artifact excluding development files.

Usage:
    python scripts/export_release_zip.py --out dist/hypertensor-release.zip

Pass Criteria:
    - Excludes: .git/, .venv/, .pytest_cache/, *.egg-info/, dist/, __pycache__/
    - Excludes: Large local results, IDE configs
    - Includes: source, tests, proofs, benchmarks, docs, configs
"""

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Set

# Patterns to exclude from release
EXCLUDE_PATTERNS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "dist",
    "build",
    ".eggs",
    "*.pyc",
    "*.pyo",
    ".coverage",
    "htmlcov",
    ".tox",
    ".nox",
    "node_modules",
    ".idea",
    ".vscode",
    "*.swp",
    "*.swo",
    ".DS_Store",
    "Thumbs.db",
    # Large result files
    "*.pt",
    "*.pth",
    "*.ckpt",
    # Secrets
    ".env",
    ".secrets*",
    "*.key",
    "*.pem",
    # Numba cache files
    "*.nbc",
    "*.nbi",
}

# Directories to always exclude
EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    ".tox",
    ".nox",
    "htmlcov",
    ".idea",
    ".vscode",
    # Rust build artifacts
    "target",
}

# File extensions for large files to exclude
LARGE_FILE_EXTENSIONS: Set[str] = {".pt", ".pth", ".ckpt", ".h5", ".hdf5"}

# Maximum file size to include (10 MB)
MAX_FILE_SIZE: int = 10 * 1024 * 1024


def should_exclude(path: Path, root: Path) -> bool:
    """Check if a path should be excluded from the release."""
    rel_path = path.relative_to(root)
    parts = rel_path.parts

    # Check directory exclusions
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True

    # Check file patterns
    name = path.name
    if name.startswith(".") and name not in {".gitignore", ".pre-commit-config.yaml"}:
        return True

    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True

    # Check file size for regular files
    if path.is_file():
        if path.suffix in LARGE_FILE_EXTENSIONS:
            return True
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                return True
        except OSError:
            return True

    return False


def collect_files(root: Path) -> List[Path]:
    """Collect all files to include in the release."""
    files = []

    for item in root.rglob("*"):
        if item.is_file() and not should_exclude(item, root):
            files.append(item)

    return sorted(files)


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_release_zip(root: Path, output: Path, verbose: bool = False) -> dict:
    """Create the release zip file."""
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(root)
    manifest = {
        "created": datetime.utcnow().isoformat() + "Z",
        "file_count": len(files),
        "files": {},
    }

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            rel_path = file_path.relative_to(root)
            arcname = str(rel_path).replace("\\", "/")

            zf.write(file_path, arcname)

            file_hash = compute_file_hash(file_path)
            manifest["files"][arcname] = {
                "size": file_path.stat().st_size,
                "sha256": file_hash,
            }

            if verbose:
                print(f"  Added: {arcname}")

    # Compute overall hash
    manifest["archive_sha256"] = compute_file_hash(output)
    manifest["archive_size"] = output.stat().st_size

    return manifest


def verify_exclusions(manifest: dict) -> List[str]:
    """Verify that excluded patterns are not in the manifest."""
    issues = []

    for filepath in manifest["files"].keys():
        parts = filepath.split("/")

        for part in parts:
            if part in EXCLUDE_DIRS:
                issues.append(f"Excluded directory found: {filepath}")
                break
            if part.endswith(".egg-info"):
                issues.append(f"Egg-info found: {filepath}")
                break

        if filepath.endswith((".pyc", ".pyo", ".pt", ".pth")):
            issues.append(f"Excluded extension found: {filepath}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Create clean release artifact")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("dist/hypertensor-release.zip"),
        help="Output zip file path",
    )
    parser.add_argument(
        "--root", "-r", type=Path, default=Path("."), help="Project root directory"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--manifest", "-m", type=Path, default=None, help="Output manifest JSON path"
    )

    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out

    print("=" * 60)
    print(" RELEASE EXPORT")
    print("=" * 60)
    print(f"Root: {root}")
    print(f"Output: {output}")
    print()

    # Create release
    print("Collecting files...")
    manifest = create_release_zip(root, output, verbose=args.verbose)

    print(f"\nFiles included: {manifest['file_count']}")
    print(f"Archive size: {manifest['archive_size'] / 1024 / 1024:.2f} MB")
    print(f"Archive SHA256: {manifest['archive_sha256'][:16]}...")

    # Verify exclusions
    issues = verify_exclusions(manifest)
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    # Save manifest
    manifest_path = args.manifest or output.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    print("\n" + "=" * 60)
    print(" ✓ RELEASE EXPORT PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
