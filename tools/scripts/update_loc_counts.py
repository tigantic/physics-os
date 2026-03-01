#!/usr/bin/env python3
"""
Ontic LOC Counter & Documentation Updater

Automatically scans the repository for line counts and updates:
- PLATFORM_SPECIFICATION.md
- TOOLBOX.md

Run: python tools/scripts/update_loc_counts.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class DirStats(TypedDict):
    files: int
    loc: int
    percentage: float


@dataclass
class LOCStats:
    """Repository line-of-code statistics."""
    
    python_files: int = 0
    python_loc: int = 0
    rust_files: int = 0
    rust_loc: int = 0
    lean_files: int = 0
    lean_loc: int = 0
    total_files: int = 0
    total_loc: int = 0
    
    python_dirs: dict[str, DirStats] = field(default_factory=dict)
    rust_crates: dict[str, DirStats] = field(default_factory=dict)
    lean_files_detail: dict[str, int] = field(default_factory=dict)
    ontic_modules: dict[str, DirStats] = field(default_factory=dict)
    
    gauntlet_count: int = 0
    demo_count: int = 0
    test_count: int = 0
    doc_count: int = 0
    attestation_count: int = 0


def count_lines(filepath: Path) -> int:
    """Count non-empty, non-comment lines in a file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        count = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Handle Python/Rust multiline strings/comments
            if '"""' in stripped or "'''" in stripped:
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_multiline_comment = not in_multiline_comment
                count += 1
                continue
            
            if in_multiline_comment:
                count += 1
                continue
            
            # Skip single-line comments
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("--"):
                continue
            
            count += 1
        
        return count
    except Exception:
        return 0


def scan_directory(
    root: Path,
    extension: str,
    exclude_dirs: set[str] | None = None
) -> tuple[int, int, dict[str, DirStats]]:
    """Scan directory for files with given extension."""
    exclude_dirs = exclude_dirs or {
        ".git", ".venv", "node_modules", "__pycache__", 
        ".pytest_cache", "target", "dist", ".cache",
        "venv", "env", ".mypy_cache", ".ruff_cache"
    }
    
    total_files = 0
    total_loc = 0
    dir_stats: dict[str, DirStats] = defaultdict(lambda: {"files": 0, "loc": 0, "percentage": 0.0})
    
    for filepath in root.rglob(f"*{extension}"):
        # Skip excluded directories
        if any(excluded in filepath.parts for excluded in exclude_dirs):
            continue
        
        loc = count_lines(filepath)
        total_files += 1
        total_loc += loc
        
        # Get relative directory
        rel_path = filepath.relative_to(root)
        if len(rel_path.parts) > 1:
            top_dir = rel_path.parts[0]
        else:
            top_dir = "root"
        
        dir_stats[top_dir]["files"] += 1
        dir_stats[top_dir]["loc"] += loc
    
    # Calculate percentages
    for dir_name in dir_stats:
        if total_loc > 0:
            dir_stats[dir_name]["percentage"] = round(
                dir_stats[dir_name]["loc"] / total_loc * 100, 1
            )
    
    return total_files, total_loc, dict(dir_stats)


def scan_ontic_modules(ontic_path: Path) -> dict[str, DirStats]:
    """Scan ontic submodules."""
    modules: dict[str, DirStats] = {}
    
    if not ontic_path.exists():
        return modules
    
    for subdir in sorted(ontic_path.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("_"):
            files = list(subdir.rglob("*.py"))
            loc = sum(count_lines(f) for f in files)
            modules[subdir.name] = {
                "files": len(files),
                "loc": loc,
                "percentage": 0.0
            }
    
    # Calculate percentages
    total_loc = sum(m["loc"] for m in modules.values())
    for name in modules:
        if total_loc > 0:
            modules[name]["percentage"] = round(modules[name]["loc"] / total_loc * 100, 1)
    
    return modules


def scan_rust_crates(root: Path) -> dict[str, DirStats]:
    """Scan Rust crates."""
    crates: dict[str, DirStats] = {}
    
    crate_locations = [
        ("fluidelite-zk", root / "fluidelite-zk"),
        ("glass_cockpit", root / "apps" / "glass_cockpit"),
        ("global_eye", root / "apps" / "global_eye"),
        ("ontic_bridge", root / "crates" / "ontic_bridge"),
        ("ontic_core", root / "crates" / "ontic_core"),
        ("tci_core_rust", root / "tci_core_rust"),
    ]
    
    for name, path in crate_locations:
        if path.exists():
            files = list(path.rglob("*.rs"))
            loc = sum(count_lines(f) for f in files)
            crates[name] = {
                "files": len(files),
                "loc": loc,
                "percentage": 0.0
            }
    
    return crates


def scan_lean_files(root: Path) -> dict[str, int]:
    """Scan Lean 4 files."""
    lean_files: dict[str, int] = {}
    
    for filepath in root.rglob("*.lean"):
        if ".lake" not in str(filepath):
            rel_path = filepath.relative_to(root)
            lean_files[str(rel_path)] = count_lines(filepath)
    
    return lean_files


def count_artifacts(root: Path) -> tuple[int, int, int, int, int]:
    """Count gauntlets, demos, tests, docs, attestations."""
    gauntlets = len(list(root.glob("*_gauntlet.py")))
    demos = len(list((root / "demos").rglob("*.py"))) if (root / "demos").exists() else 0
    tests = len(list(root.rglob("test_*.py"))) + len(list(root.rglob("*_test.py")))
    docs = len(list(root.glob("*.md")))
    attestations = len(list(root.glob("*_ATTESTATION.json")))
    
    return gauntlets, demos, tests, docs, attestations


def collect_stats(root: Path) -> LOCStats:
    """Collect all repository statistics."""
    stats = LOCStats()
    
    # Python
    stats.python_files, stats.python_loc, stats.python_dirs = scan_directory(root, ".py")
    
    # Rust
    stats.rust_files, stats.rust_loc, _ = scan_directory(root, ".rs")
    stats.rust_crates = scan_rust_crates(root)
    
    # Lean
    stats.lean_files_detail = scan_lean_files(root)
    stats.lean_files = len(stats.lean_files_detail)
    stats.lean_loc = sum(stats.lean_files_detail.values())
    
    # Total
    stats.total_files = stats.python_files + stats.rust_files + stats.lean_files
    stats.total_loc = stats.python_loc + stats.rust_loc + stats.lean_loc
    
    # ontic modules
    ontic_path = root / "ontic"
    if ontic_path.exists():
        stats.ontic_modules = scan_ontic_modules(ontic_path)
    
    # Artifacts
    (
        stats.gauntlet_count,
        stats.demo_count,
        stats.test_count,
        stats.doc_count,
        stats.attestation_count,
    ) = count_artifacts(root)
    
    return stats


def format_number(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"


def update_platform_spec(filepath: Path, stats: LOCStats, dry_run: bool = False) -> bool:
    """Update PLATFORM_SPECIFICATION.md with current stats."""
    if not filepath.exists():
        print(f"  ⚠️  {filepath} not found")
        return False
    
    content = filepath.read_text()
    original = content
    
    # Update repository metrics table
    patterns = [
        (r"\*\*Total Lines of Code\*\*\s*\|\s*\*\*[\d,]+\*\*", 
         f"**Total Lines of Code** | **{format_number(stats.total_loc)}**"),
        (r"\*\*Python LOC\*\*\s*\|\s*[\d,]+",
         f"**Python LOC** | {format_number(stats.python_loc)}"),
        (r"\*\*Rust LOC\*\*\s*\|\s*[\d,]+",
         f"**Rust LOC** | {format_number(stats.rust_loc)}"),
        (r"\*\*Lean 4 LOC\*\*\s*\|\s*[\d,]+",
         f"**Lean 4 LOC** | {format_number(stats.lean_loc)}"),
        (r"\*\*Total Files\*\*\s*\|\s*[\d,]+",
         f"**Total Files** | {format_number(stats.total_files)}"),
        (r"\*\*Test Files\*\*\s*\|\s*[\d,]+\+?",
         f"**Test Files** | {stats.test_count}+"),
        (r"\*\*Documentation Files\*\*\s*\|\s*[\d,]+\+?",
         f"**Documentation Files** | {stats.doc_count}+"),
        (r"\*\*Attestation JSONs\*\*\s*\|\s*[\d,]+\+?",
         f"**Attestation JSONs** | {stats.attestation_count}+"),
        (r"\*\*Gauntlets\*\*\s*\|\s*\d+",
         f"**Gauntlets** | {stats.gauntlet_count}"),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Update last updated date (preserve existing version number)
    today = datetime.now().strftime("%B %d, %Y")
    version_match = re.search(r"\*\*Version ([\d.]+)\*\*", content)
    version_str = version_match.group(1) if version_match else "34.0"
    content = re.sub(
        r"\*Last Updated:.*?\*",
        f"*Last Updated: {today} — Version {version_str}*",
        content
    )
    
    if content != original:
        if dry_run:
            print(f"  📝 Would update {filepath}")
        else:
            filepath.write_text(content)
            print(f"  ✅ Updated {filepath}")
        return True
    else:
        print(f"  ℹ️  No changes needed for {filepath}")
        return False


def update_toolbox(filepath: Path, stats: LOCStats, dry_run: bool = False) -> bool:
    """Update TOOLBOX.md with current stats."""
    if not filepath.exists():
        print(f"  ⚠️  {filepath} not found")
        return False
    
    content = filepath.read_text()
    original = content
    
    # Update overview table
    patterns = [
        (r"\*\*Total Files\*\*\s*\|\s*[\d,]+",
         f"**Total Files** | {format_number(stats.total_files)}"),
        (r"\*\*Total LOC\*\*\s*\|\s*\*\*[\d,]+\*\*",
         f"**Total LOC** | **{format_number(stats.total_loc)}**"),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Update date
    today = datetime.now().strftime("%B %d, %Y")
    content = re.sub(
        r"\*Generated by The Ontic Engine.*?\*",
        f"*Generated by Ontic Phase 26 • {today}*",
        content
    )
    
    if content != original:
        if dry_run:
            print(f"  📝 Would update {filepath}")
        else:
            filepath.write_text(content)
            print(f"  ✅ Updated {filepath}")
        return True
    else:
        print(f"  ℹ️  No changes needed for {filepath}")
        return False


def generate_catalog_json(root: Path, stats: LOCStats) -> dict:
    """Generate machine-readable component catalog."""
    return {
        "generated_at": datetime.now().isoformat(),
        "version": "34.0",
        "summary": {
            "total_loc": stats.total_loc,
            "total_files": stats.total_files,
            "python_loc": stats.python_loc,
            "rust_loc": stats.rust_loc,
            "lean_loc": stats.lean_loc,
        },
        "platforms": [
            {
                "name": "The Ontic Engine",
                "location": "ontic/",
                "language": "Python",
                "files": stats.python_dirs.get("ontic", {}).get("files", 0),
                "loc": stats.python_dirs.get("ontic", {}).get("loc", 0),
            },
            {
                "name": "FluidElite",
                "location": "fluidelite/, fluidelite-zk/",
                "language": "Python + Rust",
                "files": (
                    stats.python_dirs.get("fluidelite", {}).get("files", 0) +
                    stats.rust_crates.get("fluidelite-zk", {}).get("files", 0)
                ),
            },
            {
                "name": "Sovereign Compute",
                "location": "ontic/sovereign/, gevulot/",
                "language": "Python + Rust",
            },
        ],
        "modules": {
            "ontic": stats.ontic_modules,
            "rust_crates": stats.rust_crates,
        },
        "applications": {
            "gauntlets": stats.gauntlet_count,
            "demos": stats.demo_count,
        },
        "artifacts": {
            "tests": stats.test_count,
            "docs": stats.doc_count,
            "attestations": stats.attestation_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Ontic LOC counts")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    parser.add_argument("--json", action="store_true", help="Output stats as JSON")
    parser.add_argument("--catalog", action="store_true", help="Generate component-catalog.json")
    args = parser.parse_args()
    
    # Find repo root
    script_dir = Path(__file__).parent
    root = script_dir.parent
    
    if not (root / "ontic").exists():
        print("❌ Error: Must run from The Ontic Engine repo root")
        return 1
    
    print("🔍 Scanning repository...")
    stats = collect_stats(root)
    
    if args.json:
        print(json.dumps({
            "python_loc": stats.python_loc,
            "rust_loc": stats.rust_loc,
            "lean_loc": stats.lean_loc,
            "total_loc": stats.total_loc,
            "total_files": stats.total_files,
        }, indent=2))
        return 0
    
    print(f"\n📊 Repository Statistics:")
    print(f"   Python:  {format_number(stats.python_files):>6} files | {format_number(stats.python_loc):>9} LOC")
    print(f"   Rust:    {format_number(stats.rust_files):>6} files | {format_number(stats.rust_loc):>9} LOC")
    print(f"   Lean:    {format_number(stats.lean_files):>6} files | {format_number(stats.lean_loc):>9} LOC")
    print(f"   ─────────────────────────────────")
    print(f"   Total:   {format_number(stats.total_files):>6} files | {format_number(stats.total_loc):>9} LOC")
    
    print(f"\n📦 Artifacts:")
    print(f"   Gauntlets:    {stats.gauntlet_count}")
    print(f"   Demos:        {stats.demo_count}")
    print(f"   Tests:        {stats.test_count}")
    print(f"   Docs:         {stats.doc_count}")
    print(f"   Attestations: {stats.attestation_count}")
    
    if args.catalog:
        catalog = generate_catalog_json(root, stats)
        catalog_path = root / "component-catalog.json"
        if args.dry_run:
            print(f"\n📝 Would write {catalog_path}")
        else:
            catalog_path.write_text(json.dumps(catalog, indent=2))
            print(f"\n✅ Generated {catalog_path}")
    
    print(f"\n📄 Updating documentation...")
    update_platform_spec(root / "PLATFORM_SPECIFICATION.md", stats, args.dry_run)
    update_toolbox(root / "TOOLBOX.md", stats, args.dry_run)
    
    if args.dry_run:
        print("\n⚠️  Dry run - no files were modified")
    else:
        print("\n✅ Documentation updated successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
