"""
Shared Utilities for Proof Scripts
===================================

This module provides common utilities, constants, and data structures
used across all proof scripts in the HyperTensor project.

Usage:
    from proofs.common import ProofResult, TOLERANCES, run_proofs, save_results
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import torch

# =============================================================================
# CONSTITUTIONAL TOLERANCES (Article I, Section 1.2)
# =============================================================================


class Tolerances:
    """Constitutional tolerance thresholds for proof verification."""

    # Absolute tolerances
    MACHINE_PRECISION: float = 1e-14
    NUMERICAL_STABILITY: float = 1e-10
    ALGORITHM_CONVERGENCE: float = 1e-8
    PHYSICAL_ACCURACY: float = 1e-6

    # CFD-specific
    CONSERVATION_ERROR: float = 1e-12
    SHOCK_LOCATION_ERROR: float = 0.02  # 2% of domain
    WENO_ORDER_TOLERANCE: float = 0.3  # Allow ±0.3 order

    @classmethod
    def get(cls, name: str) -> float:
        """Get tolerance by name."""
        return getattr(cls, name.upper(), cls.NUMERICAL_STABILITY)


# Singleton instance for convenient access
TOLERANCES = Tolerances()


# =============================================================================
# PROOF RESULT DATA STRUCTURE
# =============================================================================


@dataclass
class Measurement:
    """Single measurement in a proof."""

    name: str
    value: float
    unit: str = ""
    threshold: Optional[float] = None
    passed: Optional[bool] = None


@dataclass
class ProofResult:
    """Container for proof result.

    Attributes:
        id: Unique proof identifier (e.g., "2.1")
        name: Human-readable proof name
        category: Category (e.g., "MPS", "CFD", "Algorithms")
        claim: Mathematical claim being proved
        status: "PASS", "FAIL", or "SKIP"
        measurements: List of measurements taken
        tolerance: Tolerance used for verification
        max_error: Maximum observed error
        duration_sec: Execution time in seconds
        timestamp: ISO timestamp of execution
    """

    id: str
    name: str
    category: str
    claim: str
    status: str
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    tolerance: float = Tolerances.NUMERICAL_STABILITY
    max_error: float = 0.0
    duration_sec: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def passed(self) -> bool:
        """Check if proof passed."""
        return self.status == "PASS"

    def add_measurement(
        self, name: str, value: float, unit: str = "", threshold: Optional[float] = None
    ) -> None:
        """Add a measurement to this proof result."""
        passed = None
        if threshold is not None:
            passed = abs(value) <= threshold
        self.measurements.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "threshold": threshold,
                "passed": passed,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# PROOF EXECUTION UTILITIES
# =============================================================================

ProofFunction = Callable[[], ProofResult]


def run_proofs(proofs: List[ProofFunction], verbose: bool = True) -> List[ProofResult]:
    """Execute a list of proof functions and collect results.

    Args:
        proofs: List of proof functions that return ProofResult
        verbose: Print progress and results

    Returns:
        List of ProofResult objects
    """
    import time

    results: List[ProofResult] = []

    for proof_fn in proofs:
        start_time = time.time()

        try:
            result = proof_fn()
            result.duration_sec = time.time() - start_time

            if verbose:
                status_icon = "✓" if result.passed else "✗"
                print(
                    f"  [{status_icon}] {result.id}: {result.name} "
                    f"(max_error={result.max_error:.2e})"
                )

        except Exception as e:
            result = ProofResult(
                id=getattr(proof_fn, "__name__", "unknown"),
                name=str(e),
                category="ERROR",
                claim="",
                status="FAIL",
                max_error=float("inf"),
            )
            result.duration_sec = time.time() - start_time

            if verbose:
                print(f"  [✗] {result.id}: EXCEPTION - {e}")

        results.append(result)

    return results


def save_results(
    results: List[ProofResult],
    output_path: Path,
    script_name: str = "proof",
) -> Dict[str, Any]:
    """Save proof results to JSON file.

    Args:
        results: List of ProofResult objects
        output_path: Path to output JSON file
        script_name: Name of the proof script

    Returns:
        The report dictionary that was saved
    """
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    report = {
        "script": script_name,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results) if results else 0,
        },
        "proofs": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def format_result_table(results: List[ProofResult]) -> str:
    """Format proof results as a text table.

    Args:
        results: List of ProofResult objects

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'ID':<8} {'Name':<35} {'Status':<8} {'Max Error':<12}")
    lines.append("-" * 70)

    for r in results:
        status_icon = "✓ PASS" if r.passed else "✗ FAIL"
        error_str = f"{r.max_error:.2e}" if r.max_error < float("inf") else "N/A"
        lines.append(f"{r.id:<8} {r.name[:35]:<35} {status_icon:<8} {error_str:<12}")

    lines.append("=" * 70)

    passed = sum(1 for r in results if r.passed)
    lines.append(
        f"Total: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}"
    )

    return "\n".join(lines)


def print_summary(results: List[ProofResult]) -> None:
    """Print a summary of proof results."""
    print(format_result_table(results))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_deterministic(seed: int = 42) -> None:
    """Set seeds for reproducible proof execution."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_scientific(value: float, precision: int = 2) -> str:
    """Format a number in scientific notation."""
    return f"{value:.{precision}e}"


class ProofContext:
    """Context manager for proof execution with timing and error handling.

    Usage:
        with ProofContext("2.1", "MPS Round Trip") as ctx:
            # Run proof...
            ctx.result.max_error = error
            ctx.result.status = "PASS" if error < tol else "FAIL"
    """

    def __init__(
        self,
        proof_id: str,
        name: str,
        category: str = "General",
        claim: str = "",
        tolerance: float = Tolerances.NUMERICAL_STABILITY,
    ):
        self.result = ProofResult(
            id=proof_id,
            name=name,
            category=category,
            claim=claim,
            status="FAIL",
            tolerance=tolerance,
        )
        self._start_time = None

    def __enter__(self) -> "ProofContext":
        import time

        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.result.duration_sec = time.time() - self._start_time

        if exc_type is not None:
            self.result.status = "FAIL"
            self.result.add_measurement("exception", 0, str(exc_val))
            return False  # Re-raise exception

        return False


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Ensure the parent directory is in path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
