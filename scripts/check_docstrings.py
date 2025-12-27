#!/usr/bin/env python3
"""
Check docstring coverage for public modules.

Usage:
    python scripts/check_docstrings.py
    python scripts/check_docstrings.py --threshold 80

Returns:
    Exit code 0 if coverage >= threshold, 1 otherwise
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Tuple


def get_public_items(module_path: Path) -> Tuple[int, int, List[str]]:
    """
    Count public items and their docstring coverage.
    
    Returns:
        (total_public, documented, missing_docs)
    """
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return 0, 0, []
    
    total_public = 0
    documented = 0
    missing = []
    
    for node in ast.walk(tree):
        name = None
        
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.AsyncFunctionDef):
            name = node.name
        elif isinstance(node, ast.ClassDef):
            name = node.name
        
        if name is None:
            continue
            
        # Skip private items
        if name.startswith("_"):
            continue
            
        total_public += 1
        
        # Check for docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            documented += 1
        else:
            missing.append(f"{module_path}:{name}")
    
    return total_public, documented, missing


def check_coverage(package_path: Path, threshold: float, verbose: bool) -> bool:
    """Check docstring coverage for all modules in package."""
    total_public = 0
    total_documented = 0
    all_missing = []
    
    # Find all Python files
    for py_file in package_path.rglob("*.py"):
        # Skip test files and __pycache__
        if "__pycache__" in str(py_file):
            continue
        if "test_" in py_file.name:
            continue
            
        public, documented, missing = get_public_items(py_file)
        total_public += public
        total_documented += documented
        all_missing.extend(missing)
    
    if total_public == 0:
        print("No public items found")
        return True
    
    coverage = (total_documented / total_public) * 100
    
    print(f"\n{'='*60}")
    print(f"Docstring Coverage Report")
    print(f"{'='*60}")
    print(f"Total public items: {total_public}")
    print(f"Documented:         {total_documented}")
    print(f"Missing docs:       {len(all_missing)}")
    print(f"Coverage:           {coverage:.1f}%")
    print(f"Threshold:          {threshold}%")
    print(f"{'='*60}")
    
    if coverage >= threshold:
        print(f"✅ PASS: Docstring coverage {coverage:.1f}% >= {threshold}%")
        return True
    else:
        print(f"❌ FAIL: Docstring coverage {coverage:.1f}% < {threshold}%")
        
        if verbose and all_missing:
            print("\nMissing docstrings:")
            for item in sorted(all_missing)[:20]:  # Show first 20
                print(f"  - {item}")
            if len(all_missing) > 20:
                print(f"  ... and {len(all_missing) - 20} more")
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Check docstring coverage")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Minimum coverage percentage (default: 70)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show missing docstrings",
    )
    parser.add_argument(
        "package",
        nargs="?",
        default="tensornet",
        help="Package to check (default: tensornet)",
    )
    
    args = parser.parse_args()
    
    package_path = Path(args.package)
    if not package_path.exists():
        print(f"Error: Package path '{package_path}' not found")
        sys.exit(1)
    
    success = check_coverage(package_path, args.threshold, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
