"""
Numerical Behavior Lock
=======================

Centralized configuration for deterministic numerical behavior.

This module ensures:
1. All random seeds are controlled
2. Tolerances are documented
3. Reproducibility is guaranteed across runs

Usage:
    from ontic.core.determinism import lock_seeds, TOLERANCES

    lock_seeds()  # Call before any computation

Constitution Compliance: Article II (Reproducibility)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

# =============================================================================
# Master Seed Configuration
# =============================================================================

MASTER_SEED = 42

# Module-specific seeds (derived from master for reproducibility)
SEEDS = {
    "numpy": MASTER_SEED,
    "torch": MASTER_SEED + 1,
    "python": MASTER_SEED + 2,
    "tt_truncation": MASTER_SEED + 100,
    "weno_smoothness": MASTER_SEED + 200,
    "tdvp_integrator": MASTER_SEED + 300,
    "dmrg_random_init": MASTER_SEED + 400,
    "test_random": MASTER_SEED + 1000,
}


def lock_seeds(seed: int | None = None) -> None:
    """
    Lock all random number generators to ensure deterministic behavior.

    Args:
        seed: Optional override for master seed. If None, uses MASTER_SEED.
    """
    if seed is None:
        seed = MASTER_SEED

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Environment variable for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seed(component: str) -> int:
    """
    Get the deterministic seed for a specific component.

    Args:
        component: Name of the component (e.g., 'tt_truncation')

    Returns:
        Seed value
    """
    return SEEDS.get(component, MASTER_SEED)


# =============================================================================
# Tolerance Configuration
# =============================================================================


@dataclass
class ToleranceSpec:
    """Specification for a numerical tolerance."""

    name: str
    value: float
    description: str
    test_type: str  # 'absolute', 'relative', 'ulp'


TOLERANCES: dict[str, ToleranceSpec] = {
    # CFD tolerances
    "cfd_conservation": ToleranceSpec(
        name="CFD Conservation",
        value=1e-6,
        description="Max allowed mass/energy conservation error",
        test_type="relative",
    ),
    "cfd_shock_structure": ToleranceSpec(
        name="Shock Structure",
        value=0.2,
        description="Allowed deviation in density ratio across shock",
        test_type="relative",
    ),
    "cfd_positivity": ToleranceSpec(
        name="Positivity",
        value=1e-10,
        description="Minimum allowed density/pressure",
        test_type="absolute",
    ),
    # WENO tolerances
    "weno_smoothness": ToleranceSpec(
        name="WENO Smoothness",
        value=1e-40,
        description="Epsilon in smoothness indicator denominator",
        test_type="absolute",
    ),
    "weno_tt_agreement": ToleranceSpec(
        name="WENO-TT Agreement",
        value=0.05,
        description="Max L2 error between TT and dense WENO",
        test_type="relative",
    ),
    # TT/MPS tolerances
    "svd_cutoff": ToleranceSpec(
        name="SVD Cutoff",
        value=1e-10,
        description="Singular value truncation threshold",
        test_type="absolute",
    ),
    "mps_norm": ToleranceSpec(
        name="MPS Normalization",
        value=1e-12,
        description="Allowed deviation from unit norm",
        test_type="absolute",
    ),
    "dmrg_convergence": ToleranceSpec(
        name="DMRG Convergence",
        value=1e-8,
        description="Energy convergence threshold",
        test_type="absolute",
    ),
    # TDVP tolerances
    "tdvp_lanczos": ToleranceSpec(
        name="Lanczos Tolerance",
        value=1e-12,
        description="Convergence for Lanczos exponential",
        test_type="absolute",
    ),
    "tdvp_energy_conservation": ToleranceSpec(
        name="TDVP Energy",
        value=1e-8,
        description="Energy conservation during time evolution",
        test_type="relative",
    ),
    # Integration test tolerances
    "test_float_compare": ToleranceSpec(
        name="Float Comparison",
        value=1e-7,
        description="General floating point comparison tolerance",
        test_type="absolute",
    ),
    "test_physics": ToleranceSpec(
        name="Physics Tests",
        value=0.01,
        description="Physics validation tolerance (1%)",
        test_type="relative",
    ),
}


def get_tolerance(name: str) -> float:
    """
    Get the tolerance value for a specific check.

    Args:
        name: Tolerance name (e.g., 'cfd_conservation')

    Returns:
        Tolerance value

    Raises:
        KeyError: If tolerance not defined
    """
    if name not in TOLERANCES:
        raise KeyError(
            f"Unknown tolerance: {name}. Available: {list(TOLERANCES.keys())}"
        )
    return TOLERANCES[name].value


def document_tolerances() -> str:
    """
    Generate markdown documentation of all tolerances.

    Returns:
        Markdown-formatted tolerance table
    """
    lines = [
        "# Numerical Tolerances",
        "",
        "| Name | Value | Type | Description |",
        "|------|-------|------|-------------|",
    ]

    for key, spec in sorted(TOLERANCES.items()):
        lines.append(
            f"| {spec.name} | {spec.value:.0e} | {spec.test_type} | {spec.description} |"
        )

    return "\n".join(lines)


# =============================================================================
# Determinism Verification
# =============================================================================


def verify_determinism(func, n_runs: int = 3, **kwargs) -> bool:
    """
    Verify that a function produces deterministic results.

    Args:
        func: Function to test (should return hashable result)
        n_runs: Number of runs to compare
        **kwargs: Arguments to pass to func

    Returns:
        True if all runs produce identical results
    """
    results = []

    for _ in range(n_runs):
        lock_seeds()
        result = func(**kwargs)

        # Convert to hashable if numpy/torch
        if hasattr(result, "numpy"):
            result = result.numpy()
        if isinstance(result, np.ndarray):
            result = result.tobytes()

        results.append(result)

    return len(set(results)) == 1


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "MASTER_SEED",
    "SEEDS",
    "TOLERANCES",
    "lock_seeds",
    "get_seed",
    "get_tolerance",
    "document_tolerances",
    "verify_determinism",
]
