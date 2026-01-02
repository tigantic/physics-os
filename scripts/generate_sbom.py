#!/usr/bin/env python
"""
P) SBOM/CBOM Generation Script
==============================

Generates Software Bill of Materials (SBOM) and Code Bill of Materials (CBOM).

Usage:
    python scripts/generate_sbom.py --format cyclonedx

Pass Criteria:
    - Artifacts produced in CycloneDX or SPDX format
    - Paths redacted (no absolute paths)
    - Counts stable
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


def get_installed_packages() -> List[Dict[str, str]]:
    """Get list of installed packages."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def get_package_metadata(name: str) -> Dict[str, Any]:
    """Get metadata for a package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            metadata = {}
            for line in result.stdout.split("\n"):
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip().lower().replace("-", "_")] = value.strip()
            return metadata
    except Exception:
        pass
    return {}


def generate_cyclonedx_sbom(
    packages: List[Dict[str, str]], project_root: Path
) -> Dict[str, Any]:
    """Generate SBOM in CycloneDX format."""
    # Get project metadata
    pyproject_path = project_root / "pyproject.toml"
    project_name = "hypertensor"
    project_version = "2.13.0"

    if pyproject_path.exists():
        try:
            import tomli

            with open(pyproject_path, "rb") as f:
                pyproject = tomli.load(f)
            project_info = pyproject.get("project", {})
            project_name = project_info.get("name", project_name)
            project_version = project_info.get("version", project_version)
        except Exception:
            pass

    # Build components
    components = []
    for pkg in packages:
        name = pkg.get("name", "")
        version = pkg.get("version", "")

        metadata = get_package_metadata(name)

        component = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name}@{version}",
        }

        if metadata.get("license"):
            component["licenses"] = [{"license": {"name": metadata["license"]}}]

        if metadata.get("home_page"):
            component["externalReferences"] = [
                {"type": "website", "url": metadata["home_page"]}
            ]

        components.append(component)

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [
                {
                    "vendor": "HyperTensor",
                    "name": "generate_sbom.py",
                    "version": "1.0.0",
                }
            ],
            "component": {
                "type": "application",
                "name": project_name,
                "version": project_version,
            },
        },
        "components": components,
    }

    return sbom


def generate_cbom(project_root: Path) -> Dict[str, Any]:
    """Generate Code Bill of Materials."""
    tensornet_dir = project_root / "tensornet"

    modules = []
    total_lines = 0

    for py_file in tensornet_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        rel_path = py_file.relative_to(project_root)

        # Count lines
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            lines = len([l for l in content.split("\n") if l.strip()])

            # Compute hash
            file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        except Exception:
            lines = 0
            file_hash = "error"

        total_lines += lines

        modules.append(
            {
                "path": str(rel_path).replace("\\", "/"),  # Normalize path
                "lines": lines,
                "hash": file_hash,
            }
        )

    cbom = {
        "format": "HyperTensor-CBOM",
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_modules": len(modules),
            "total_lines": total_lines,
        },
        "modules": sorted(modules, key=lambda m: m["path"]),
    }

    return cbom


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SBOM and CBOM")
    parser.add_argument(
        "--format",
        "-f",
        choices=["cyclonedx", "spdx"],
        default="cyclonedx",
        help="SBOM format",
    )
    parser.add_argument(
        "--out", "-o", type=Path, default=Path("artifacts"), help="Output directory"
    )

    args = parser.parse_args()
    project_root = Path(__file__).parent.parent
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" SBOM/CBOM GENERATION")
    print("=" * 60)
    print()

    # Get packages
    print("Collecting packages...")
    packages = get_installed_packages()
    print(f"  Found {len(packages)} packages")

    # Generate SBOM
    print(f"\nGenerating SBOM ({args.format})...")
    if args.format == "cyclonedx":
        sbom = generate_cyclonedx_sbom(packages, project_root)
        sbom_path = output_dir / "sbom.cyclonedx.json"
    else:
        # SPDX format (simplified)
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": "hypertensor-sbom",
            "packages": [
                {
                    "name": p["name"],
                    "versionInfo": p["version"],
                    "SPDXID": f"SPDXRef-Package-{p['name']}",
                }
                for p in packages
            ],
        }
        sbom_path = output_dir / "sbom.spdx.json"

    with open(sbom_path, "w") as f:
        json.dump(sbom, f, indent=2)
    print(f"  SBOM: {sbom_path}")

    # Generate CBOM
    print("\nGenerating CBOM...")
    cbom = generate_cbom(project_root)
    cbom_path = output_dir / "cbom.json"

    with open(cbom_path, "w") as f:
        json.dump(cbom, f, indent=2)
    print(f"  CBOM: {cbom_path}")
    print(f"  Modules: {cbom['summary']['total_modules']}")
    print(f"  Lines: {cbom['summary']['total_lines']:,}")

    print()
    print("=" * 60)
    print(" ✓ SBOM/CBOM GENERATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
