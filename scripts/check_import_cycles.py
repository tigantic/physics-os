#!/usr/bin/env python3
"""
Check for import cycles in the tensornet package.

This script attempts to import all public modules to detect circular import
errors at runtime. It does NOT detect static cycles that Python handles
gracefully - only cycles that cause actual import failures.

Usage:
    python scripts/check_import_cycles.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports() -> tuple[list[str], list[str]]:
    """
    Check if all tensornet modules can be imported.
    
    Returns:
        Tuple of (successful_imports, failed_imports)
    """
    modules_to_check = [
        # Core modules
        "tensornet",
        "tensornet.core",
        "tensornet.core.mps",
        "tensornet.core.mpo",
        "tensornet.core.decompositions",
        "tensornet.core.gpu",
        # CFD modules
        "tensornet.cfd",
        "tensornet.cfd.qtt",
        "tensornet.cfd.euler_1d",
        "tensornet.cfd.euler_2d",
        "tensornet.cfd.euler_3d",
        "tensornet.cfd.godunov",
        "tensornet.cfd.weno",
        # ML modules
        "tensornet.ml_surrogates",
        "tensornet.ml_surrogates.base",
        "tensornet.ml_surrogates.surrogate_base",
        "tensornet.ml_surrogates.physics_informed",
        "tensornet.ml_surrogates.deep_onet",
        "tensornet.ml_surrogates.fourier_operator",
        # Distributed
        "tensornet.distributed",
        "tensornet.distributed.domain_decomp",
        # Autonomy
        "tensornet.autonomy",
        # Guidance
        "tensornet.guidance",
    ]
    
    successful = []
    failed = []
    
    for module in modules_to_check:
        try:
            __import__(module)
            successful.append(module)
        except ImportError as e:
            failed.append(f"{module}: {e}")
        except Exception as e:
            failed.append(f"{module}: {type(e).__name__}: {e}")
    
    return successful, failed


def main() -> int:
    """Run import cycle check."""
    print("Checking tensornet imports for circular import errors...")
    print("-" * 60)
    
    successful, failed = check_imports()
    
    print(f"✅ Successfully imported: {len(successful)} modules")
    
    if failed:
        print(f"\n❌ Failed imports: {len(failed)}")
        for failure in failed:
            print(f"  - {failure}")
        return 1
    
    print("\n✅ No circular import errors detected!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
