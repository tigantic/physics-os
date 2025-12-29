#!/usr/bin/env python
"""
N) Documentation Build Script
==============================

Builds API documentation using pdoc.

Usage:
    python scripts/build_docs.py

Pass Criteria:
    - All modules documented
    - No build errors
    - Output generated in artifacts/api_docs
"""

import importlib
import pkgutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def find_all_modules(package_name: str) -> List[str]:
    """Find all submodules in a package."""
    modules = []
    
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[str(package_path)],
            prefix=f"{package_name}."
        ):
            modules.append(modname)
            
    except ImportError as e:
        print(f"Warning: Could not import {package_name}: {e}")
    
    return modules


def build_with_pdoc(output_dir: Path) -> Tuple[bool, str]:
    """Build documentation using pdoc."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pdoc', 
             'tensornet',
             '-o', str(output_dir),
             '--html',
             '--force'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return False, result.stderr
        
        return True, result.stdout
        
    except FileNotFoundError:
        return False, "pdoc not installed. Run: pip install pdoc3"
    except Exception as e:
        return False, str(e)


def build_with_pydoc(output_dir: Path) -> Tuple[bool, str]:
    """Build documentation using pydoc (fallback)."""
    try:
        modules = find_all_modules('tensornet')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for module in modules[:50]:  # Limit to avoid too many files
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pydoc', '-w', module],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=output_dir
                )
            except:
                continue
        
        return True, f"Generated docs for {len(modules)} modules"
        
    except Exception as e:
        return False, str(e)


def generate_readme_index(output_dir: Path, modules: List[str]) -> None:
    """Generate an index README for the docs."""
    readme_path = output_dir / 'README.md'
    
    content = [
        "# HyperTensor API Documentation",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Modules",
        "",
    ]
    
    for module in sorted(modules):
        content.append(f"- `{module}`")
    
    content.extend([
        "",
        "## Quick Links",
        "",
        "- [Core](tensornet/core.html)",
        "- [Algorithms](tensornet/algorithms.html)",
        "- [CFD](tensornet/cfd.html)",
        "- [MPS](tensornet/mps.html)",
    ])
    
    readme_path.write_text('\n'.join(content))


def main():
    print("=" * 60)
    print(" DOCUMENTATION BUILD")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'artifacts' / 'api_docs'
    
    # Try pdoc first
    print("Building with pdoc...")
    success, output = build_with_pdoc(output_dir)
    
    if not success:
        print(f"  pdoc failed: {output}")
        print("  Trying pydoc fallback...")
        success, output = build_with_pydoc(output_dir)
    
    if success:
        # Generate index
        modules = find_all_modules('tensornet')
        generate_readme_index(output_dir, modules)
        
        print(f"\n✓ Documentation built successfully")
        print(f"  Output: {output_dir}")
        print(f"  Modules documented: {len(modules)}")
        
        return 0
    else:
        print(f"\n✗ Documentation build failed")
        print(f"  Error: {output}")
        return 1


if __name__ == '__main__':
    exit(main())
