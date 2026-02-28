#!/usr/bin/env python3
"""
Check for import cycles in the ontic package.

This script attempts to import all public modules to detect circular import
errors at runtime. It does NOT detect static cycles that Python handles
gracefully - only cycles that cause actual import failures.

Usage:
    python tools/scripts/check_import_cycles.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports() -> tuple[list[str], list[str]]:
    """
    Check if all ontic modules can be imported.

    Returns:
        Tuple of (successful_imports, failed_imports)
    """
    modules_to_check = [
        # Core modules
        "ontic",
        "ontic.core",
        "ontic.core.mps",
        "ontic.core.mpo",
        "ontic.core.decompositions",
        "ontic.core.gpu",
        # CFD modules
        "ontic.cfd",
        "ontic.cfd.qtt",
        "ontic.cfd.euler_1d",
        "ontic.cfd.euler_2d",
        "ontic.cfd.euler_3d",
        "ontic.cfd.godunov",
        "ontic.cfd.weno",
        # ML modules
        "ontic.ml_surrogates",
        "ontic.ml_surrogates.base",
        "ontic.ml_surrogates.surrogate_base",
        "ontic.ml_surrogates.physics_informed",
        "ontic.ml_surrogates.deep_onet",
        "ontic.ml_surrogates.fourier_operator",
        # Distributed
        "ontic.distributed",
        "ontic.distributed.domain_decomp",
        # Autonomy
        "ontic.autonomy",
        # Guidance
        "ontic.guidance",
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
    print("Checking ontic imports for circular import errors...")
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
