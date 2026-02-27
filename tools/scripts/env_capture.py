#!/usr/bin/env python
"""
B) Environment Capture Script
==============================

Captures reproducible environment information.

Usage:
    python tools/scripts/env_capture.py --out artifacts/env.json

Pass Criteria:
    - File created with deterministic fields
    - Python, OS, CPU/GPU, torch versions captured
    - Installed packages hash computed
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
from typing import Any, Dict, List, Optional


def get_python_info() -> Dict[str, str]:
    """Get Python version information."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
    }


def get_os_info() -> Dict[str, str]:
    """Get operating system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    info = {
        "count": os.cpu_count(),
        "architecture": platform.machine(),
    }

    # Try to get more detailed CPU info on Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["model"] = line.split(":")[1].strip()
                        break
        except Exception:
            pass

    return info


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information via PyTorch."""
    info = {"available": False}

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = []

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
    except ImportError:
        info["error"] = "PyTorch not installed"
    except Exception as e:
        info["error"] = str(e)

    return info


def get_torch_info() -> Dict[str, Any]:
    """Get PyTorch version information."""
    info = {}

    try:
        import torch

        info["version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cudnn_enabled"] = torch.backends.cudnn.enabled
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        info["error"] = "PyTorch not installed"
    except Exception as e:
        info["error"] = str(e)

    return info


def get_installed_packages() -> List[Dict[str, str]]:
    """Get list of installed packages."""
    packages = []

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            packages = json.loads(result.stdout)
    except Exception:
        # Fallback to pkg_resources
        try:
            import pkg_resources

            packages = [
                {"name": d.project_name, "version": d.version}
                for d in pkg_resources.working_set
            ]
        except Exception:
            pass

    return sorted(packages, key=lambda x: x.get("name", "").lower())


def compute_packages_hash(packages: List[Dict[str, str]]) -> str:
    """Compute deterministic hash of installed packages."""
    # Create deterministic string representation
    pkg_str = "\n".join(
        f"{p.get('name', '')}=={p.get('version', '')}" for p in packages
    )
    return hashlib.sha256(pkg_str.encode()).hexdigest()


def get_git_info() -> Dict[str, Optional[str]]:
    """Get git repository information."""
    info = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "tag": None,
    }

    try:
        # Commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Dirty status
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0

        # Tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info["tag"] = result.stdout.strip()
    except Exception:
        pass

    return info


def capture_environment() -> Dict[str, Any]:
    """Capture complete environment information."""
    packages = get_installed_packages()

    return {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "python": get_python_info(),
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
        "torch": get_torch_info(),
        "git": get_git_info(),
        "packages": packages,
        "packages_hash": compute_packages_hash(packages),
        "packages_count": len(packages),
    }


def validate_capture(env: Dict[str, Any]) -> List[str]:
    """Validate that required fields are present."""
    issues = []

    required_fields = [
        "captured_at",
        "python.version",
        "os.system",
        "packages_hash",
    ]

    for field in required_fields:
        parts = field.split(".")
        obj = env
        for part in parts:
            if not isinstance(obj, dict) or part not in obj:
                issues.append(f"Missing required field: {field}")
                break
            obj = obj[part]

    if not env.get("packages"):
        issues.append("No packages captured")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Capture reproducible environment information"
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("artifacts/env.json"),
        help="Output JSON file path",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()
    output = args.out

    print("=" * 60)
    print(" ENVIRONMENT CAPTURE")
    print("=" * 60)

    # Capture environment
    print("Capturing environment...")
    env = capture_environment()

    # Validate
    issues = validate_capture(env)
    if issues:
        print("\n⚠️  VALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    # Output summary
    print(f"\nPython: {env['python']['version']}")
    print(f"OS: {env['os']['system']} {env['os']['release']}")
    print(f"Torch: {env['torch'].get('version', 'N/A')}")
    print(f"GPU: {'Available' if env['gpu']['available'] else 'Not available'}")
    print(f"Packages: {env['packages_count']}")
    print(f"Packages hash: {env['packages_hash'][:16]}...")

    if env["git"]["commit"]:
        print(f"Git: {env['git']['commit'][:8]} ({env['git']['branch']})")
        if env["git"]["dirty"]:
            print("  ⚠️  Working directory is dirty")

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(env, f, indent=2)

    print(f"\nOutput: {output}")

    print("\n" + "=" * 60)
    print(" ✓ ENVIRONMENT CAPTURE PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
