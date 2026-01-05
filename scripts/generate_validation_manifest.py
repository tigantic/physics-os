#!/usr/bin/env python3
"""
Validation Manifest Generator
==============================

Generates a cryptographically verifiable validation manifest for V&V results.

Usage:
    python generate_validation_manifest.py --artifacts artifacts/ --commit SHA --output manifest.json

Constitution Compliance: Article IV.1 (Verification), Phase 3 Automation
Tags: [V&V] [PROVENANCE] [CRYPTOGRAPHY]
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def get_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_info() -> dict:
    """Get current Git information."""
    info = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": True,
        "remote": "unknown",
    }

    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        dirty_check = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["dirty"] = len(dirty_check) > 0

        info["remote"] = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        pass

    return info


def get_environment_info() -> dict:
    """Get environment information for reproducibility."""
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }

    # Get key package versions
    packages = {}
    for pkg in ["torch", "numpy", "scipy", "pytest"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    env["packages"] = packages

    # Check for GPU
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if env["cuda_available"]:
            env["cuda_version"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        env["cuda_available"] = False

    return env


def collect_artifact_hashes(artifacts_dir: Path) -> dict:
    """Collect SHA-256 hashes of all artifact files."""
    hashes = {}

    for path in artifacts_dir.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(artifacts_dir)
            hashes[str(rel_path)] = {
                "sha256": get_file_hash(path),
                "size": path.stat().st_size,
            }

    return hashes


def parse_test_results(artifacts_dir: Path) -> dict:
    """Parse test results from artifacts."""
    results = {
        "mms": {"passed": 0, "failed": 0, "status": "unknown"},
        "benchmark": {"passed": 0, "failed": 0, "status": "unknown"},
        "conservation": {"passed": 0, "failed": 0, "status": "unknown"},
    }

    import xml.etree.ElementTree as ET

    for subdir in artifacts_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.suffix == ".xml":
                    try:
                        tree = ET.parse(file)
                        root = tree.getroot()

                        for testsuite in root.iter("testsuite"):
                            passed = (
                                int(testsuite.get("tests", 0))
                                - int(testsuite.get("failures", 0))
                                - int(testsuite.get("errors", 0))
                            )
                            failed = int(testsuite.get("failures", 0)) + int(
                                testsuite.get("errors", 0)
                            )

                            if "mms" in file.name:
                                results["mms"]["passed"] += passed
                                results["mms"]["failed"] += failed
                            elif "benchmark" in file.name:
                                results["benchmark"]["passed"] += passed
                                results["benchmark"]["failed"] += failed
                            elif "conservation" in file.name:
                                results["conservation"]["passed"] += passed
                                results["conservation"]["failed"] += failed
                    except Exception:
                        pass

    # Determine status
    for key in results:
        if results[key]["passed"] > 0 or results[key]["failed"] > 0:
            results[key]["status"] = "PASS" if results[key]["failed"] == 0 else "FAIL"

    return results


def generate_manifest(
    artifacts_dir: Path,
    commit: str,
    output_path: Path,
) -> dict:
    """Generate the full validation manifest."""

    manifest = {
        "manifest_version": "1.0.0",
        "schema": "hypertensor-vv-manifest-v1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "code": get_git_info(),
        "environment": get_environment_info(),
        "artifacts": collect_artifact_hashes(artifacts_dir),
        "validation": parse_test_results(artifacts_dir),
        "metadata": {
            "framework": "HyperTensor V&V Framework",
            "framework_version": "1.4.0",
            "standard": "ASME V&V 10-2019",
            "classification": "PROPRIETARY - Tigantic Holdings LLC",
        },
        "signature": None,  # Filled in by sign_manifest.py
    }

    # Override commit if provided
    if commit and commit != "unknown":
        manifest["code"]["commit"] = commit

    # Compute manifest hash (excluding signature field)
    manifest_for_hash = {k: v for k, v in manifest.items() if k != "signature"}
    manifest_json = json.dumps(manifest_for_hash, sort_keys=True)
    manifest["manifest_hash"] = hashlib.sha256(manifest_json.encode()).hexdigest()

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate Validation Manifest")
    parser.add_argument(
        "--artifacts", type=Path, default=Path("artifacts"), help="Artifacts directory"
    )
    parser.add_argument("--commit", type=str, default="unknown", help="Commit SHA")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_manifest.json"),
        help="Output file",
    )
    args = parser.parse_args()

    print("Generating Validation Manifest")
    print("=" * 40)

    manifest = generate_manifest(args.artifacts, args.commit, args.output)

    # Print summary
    print(f"Timestamp: {manifest['timestamp']}")
    print(f"Commit: {manifest['code']['commit'][:12]}...")
    print(f"Artifacts: {len(manifest['artifacts'])} files")
    print(f"Manifest Hash: {manifest['manifest_hash'][:16]}...")

    print("\nValidation Results:")
    for cat, results in manifest["validation"].items():
        status_icon = (
            "✅"
            if results["status"] == "PASS"
            else "❌" if results["status"] == "FAIL" else "❓"
        )
        print(
            f"  {cat}: {status_icon} {results['passed']} passed, {results['failed']} failed"
        )

    # Write manifest
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to: {args.output}")


if __name__ == "__main__":
    main()
