"""
Regression Testing Module for Project HyperTensor.

Provides regression testing framework including:
- Golden value comparison and management
- Array and tensor comparison utilities
- State comparison for CFD simulations
- Automated regression test suite execution

These tools ensure code changes don't introduce regressions.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
import json
import hashlib
import time
import pickle


class ComparisonType(Enum):
    """Type of comparison for regression testing."""
    EXACT = auto()  # Bit-exact comparison
    RELATIVE = auto()  # Relative tolerance
    ABSOLUTE = auto()  # Absolute tolerance
    HYBRID = auto()  # Both relative and absolute


@dataclass
class RegressionResult:
    """
    Result of a regression test.
    
    Attributes:
        test_name: Name of the regression test
        passed: Whether the test passed
        comparison_type: Type of comparison used
        max_difference: Maximum difference found
        mean_difference: Mean difference
        tolerance_used: Tolerance applied
        n_mismatches: Number of elements exceeding tolerance
        message: Result message
        details: Additional test details
    """
    test_name: str
    passed: bool
    comparison_type: ComparisonType
    max_difference: float
    mean_difference: float
    tolerance_used: float
    n_mismatches: int = 0
    n_elements: int = 0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mismatch_rate(self) -> float:
        """Fraction of elements that mismatch."""
        if self.n_elements == 0:
            return 0.0
        return self.n_mismatches / self.n_elements
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'comparison_type': self.comparison_type.name,
            'max_difference': float(self.max_difference),
            'mean_difference': float(self.mean_difference),
            'tolerance_used': float(self.tolerance_used),
            'n_mismatches': self.n_mismatches,
            'n_elements': self.n_elements,
            'mismatch_rate': self.mismatch_rate,
            'message': self.message,
            'details': self.details,
        }


@dataclass
class GoldenValue:
    """
    Reference value for regression testing.
    
    Stores a value along with metadata for version tracking
    and reproducibility.
    
    Attributes:
        name: Unique identifier for this golden value
        value: The reference value (tensor, array, or scalar)
        value_hash: Hash of the value for quick comparison
        created_at: Timestamp when created
        version: Version string
        metadata: Additional context
    """
    name: str
    value: Any
    value_hash: str = ""
    created_at: float = field(default_factory=time.time)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute hash if not provided."""
        if not self.value_hash:
            self.value_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of the value."""
        if isinstance(self.value, torch.Tensor):
            data = self.value.cpu().numpy().tobytes()
        elif isinstance(self.value, np.ndarray):
            data = self.value.tobytes()
        else:
            # Use JSON for non-array types (safer than pickle)
            data = json.dumps(self.value, sort_keys=True, default=str).encode()
        
        return hashlib.sha256(data).hexdigest()[:16]
    
    def verify_hash(self) -> bool:
        """Verify the stored hash matches current value."""
        return self.value_hash == self._compute_hash()
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary (excludes large value)."""
        return {
            'name': self.name,
            'value_hash': self.value_hash,
            'created_at': self.created_at,
            'version': self.version,
            'metadata': self.metadata,
            'shape': list(self.value.shape) if hasattr(self.value, 'shape') else None,
            'dtype': str(self.value.dtype) if hasattr(self.value, 'dtype') else type(self.value).__name__,
        }


class GoldenValueStore:
    """
    Persistent storage for golden values.
    
    Manages a collection of golden values with versioning
    and efficient lookup.
    
    Example:
        store = GoldenValueStore("./golden_values")
        store.save("dmrg_energy", energy_value)
        golden = store.load("dmrg_energy")
    """
    
    def __init__(self, directory: Union[str, Path]):
        """
        Initialize golden value store.
        
        Args:
            directory: Directory for storing golden values
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index_file = self.directory / "index.json"
        self._index: Dict[str, Dict] = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load index from disk."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save index to disk."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def save(
        self,
        name: str,
        value: Any,
        version: str = "1.0",
        **metadata,
    ) -> GoldenValue:
        """
        Save a golden value.
        
        Args:
            name: Unique identifier
            value: Value to save
            version: Version string
            **metadata: Additional metadata
            
        Returns:
            The created GoldenValue
        """
        golden = GoldenValue(
            name=name,
            value=value,
            version=version,
            metadata=metadata,
        )
        
        # Save value to disk using safe serialization
        # Use .npz for arrays, .json for other types
        if isinstance(golden.value, (np.ndarray, torch.Tensor)):
            value_path = self.directory / f"{name}.npz"
            arr = golden.value.cpu().numpy() if isinstance(golden.value, torch.Tensor) else golden.value
            np.savez_compressed(value_path, data=arr)
        else:
            value_path = self.directory / f"{name}.json"
            with open(value_path, 'w') as f:
                json.dump({'value': golden.value, 'type': type(golden.value).__name__}, f)
        
        # Update index
        self._index[name] = golden.to_dict()
        self._save_index()
        
        return golden
    
    def load(self, name: str) -> Optional[GoldenValue]:
        """
        Load a golden value.
        
        Args:
            name: Identifier of the golden value
            
        Returns:
            The GoldenValue or None if not found
        """
        # Try .npz first (array data)
        npz_path = self.directory / f"{name}.npz"
        json_path = self.directory / f"{name}.json"
        legacy_path = self.directory / f"{name}.pkl"
        
        if npz_path.exists():
            loaded = np.load(npz_path, allow_pickle=False)
            value = loaded['data']
        elif json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            value = data['value']
        elif legacy_path.exists():
            # Reject legacy pickle files - require migration
            raise ValueError(
                f"Legacy pickle file found: {legacy_path}. "
                "For security reasons, pickle files are no longer supported. "
                "Please re-save golden values using the new format."
            )
        else:
            return None
        
        # Reconstruct GoldenValue from index metadata
        meta = self._index.get(name, {})
        return GoldenValue(
            name=name,
            value=value,
            value_hash=meta.get('value_hash', ''),
            created_at=meta.get('created_at', time.time()),
            version=meta.get('version', '1.0'),
            metadata=meta.get('metadata', {}),
        )
    
    def exists(self, name: str) -> bool:
        """Check if a golden value exists."""
        return name in self._index
    
    def list_all(self) -> List[str]:
        """List all golden value names."""
        return list(self._index.keys())
    
    def delete(self, name: str):
        """Delete a golden value."""
        value_path = self.directory / f"{name}.pkl"
        if value_path.exists():
            value_path.unlink()
        if name in self._index:
            del self._index[name]
            self._save_index()
    
    def get_info(self, name: str) -> Optional[Dict]:
        """Get metadata about a golden value without loading it."""
        return self._index.get(name)


def update_golden_values(
    store: GoldenValueStore,
    values: Dict[str, Any],
    version: str = "1.0",
    force: bool = False,
) -> Dict[str, bool]:
    """
    Update multiple golden values.
    
    Args:
        store: Golden value store
        values: Dictionary of name -> value
        version: Version string for all values
        force: Overwrite existing values
        
    Returns:
        Dictionary of name -> whether updated
    """
    results = {}
    
    for name, value in values.items():
        if not force and store.exists(name):
            results[name] = False
        else:
            store.save(name, value, version=version)
            results[name] = True
    
    return results


class ArrayComparator:
    """
    Compare arrays with configurable tolerance.
    
    Supports numpy arrays and provides detailed mismatch information.
    """
    
    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        comparison_type: ComparisonType = ComparisonType.HYBRID,
    ):
        """
        Initialize array comparator.
        
        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance
            comparison_type: Type of comparison
        """
        self.rtol = rtol
        self.atol = atol
        self.comparison_type = comparison_type
    
    def compare(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        name: str = "array",
    ) -> RegressionResult:
        """
        Compare two arrays.
        
        Args:
            actual: Actual/computed array
            expected: Expected/reference array
            name: Name for the comparison
            
        Returns:
            RegressionResult with comparison details
        """
        if actual.shape != expected.shape:
            return RegressionResult(
                test_name=name,
                passed=False,
                comparison_type=self.comparison_type,
                max_difference=float('inf'),
                mean_difference=float('inf'),
                tolerance_used=self.rtol,
                message=f"Shape mismatch: {actual.shape} vs {expected.shape}",
            )
        
        diff = np.abs(actual - expected)
        
        if self.comparison_type == ComparisonType.EXACT:
            passed = np.array_equal(actual, expected)
            tolerance = 0.0
        elif self.comparison_type == ComparisonType.RELATIVE:
            tolerance = self.rtol
            rel_diff = diff / (np.abs(expected) + 1e-15)
            passed = np.all(rel_diff <= self.rtol)
        elif self.comparison_type == ComparisonType.ABSOLUTE:
            tolerance = self.atol
            passed = np.all(diff <= self.atol)
        else:  # HYBRID
            tolerance = max(self.rtol, self.atol)
            passed = np.allclose(actual, expected, rtol=self.rtol, atol=self.atol)
        
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        
        if self.comparison_type == ComparisonType.HYBRID:
            threshold = self.atol + self.rtol * np.abs(expected)
            n_mismatch = int(np.sum(diff > threshold))
        else:
            n_mismatch = int(np.sum(diff > tolerance))
        
        return RegressionResult(
            test_name=name,
            passed=passed,
            comparison_type=self.comparison_type,
            max_difference=max_diff,
            mean_difference=mean_diff,
            tolerance_used=tolerance,
            n_mismatches=n_mismatch,
            n_elements=int(actual.size),
            message="Pass" if passed else f"Max diff: {max_diff:.6e}",
        )


class TensorComparator(ArrayComparator):
    """
    Compare PyTorch tensors with configurable tolerance.
    
    Extends ArrayComparator with tensor-specific handling.
    """
    
    def compare(
        self,
        actual: Union[torch.Tensor, np.ndarray],
        expected: Union[torch.Tensor, np.ndarray],
        name: str = "tensor",
    ) -> RegressionResult:
        """
        Compare two tensors.
        
        Args:
            actual: Actual/computed tensor
            expected: Expected/reference tensor
            name: Name for the comparison
            
        Returns:
            RegressionResult with comparison details
        """
        # Convert to numpy
        if isinstance(actual, torch.Tensor):
            actual_np = actual.detach().cpu().numpy()
        else:
            actual_np = actual
        
        if isinstance(expected, torch.Tensor):
            expected_np = expected.detach().cpu().numpy()
        else:
            expected_np = expected
        
        return super().compare(actual_np, expected_np, name)


class StateComparator:
    """
    Compare CFD states with field-specific tolerances.
    
    Handles multi-field states where different variables
    may have different accuracy requirements.
    """
    
    def __init__(
        self,
        field_tolerances: Optional[Dict[str, float]] = None,
        default_rtol: float = 1e-5,
        default_atol: float = 1e-8,
    ):
        """
        Initialize state comparator.
        
        Args:
            field_tolerances: Per-field relative tolerances
            default_rtol: Default relative tolerance
            default_atol: Default absolute tolerance
        """
        self.field_tolerances = field_tolerances or {}
        self.default_rtol = default_rtol
        self.default_atol = default_atol
    
    def compare(
        self,
        actual: Dict[str, torch.Tensor],
        expected: Dict[str, torch.Tensor],
        name: str = "state",
    ) -> List[RegressionResult]:
        """
        Compare two states (dictionaries of fields).
        
        Args:
            actual: Actual state dictionary
            expected: Expected state dictionary
            name: Base name for comparisons
            
        Returns:
            List of RegressionResult for each field
        """
        results = []
        
        all_fields = set(actual.keys()) | set(expected.keys())
        
        for field in sorted(all_fields):
            if field not in actual:
                results.append(RegressionResult(
                    test_name=f"{name}.{field}",
                    passed=False,
                    comparison_type=ComparisonType.HYBRID,
                    max_difference=float('inf'),
                    mean_difference=float('inf'),
                    tolerance_used=0.0,
                    message=f"Field '{field}' missing from actual",
                ))
                continue
            
            if field not in expected:
                results.append(RegressionResult(
                    test_name=f"{name}.{field}",
                    passed=False,
                    comparison_type=ComparisonType.HYBRID,
                    max_difference=float('inf'),
                    mean_difference=float('inf'),
                    tolerance_used=0.0,
                    message=f"Field '{field}' missing from expected",
                ))
                continue
            
            rtol = self.field_tolerances.get(field, self.default_rtol)
            comparator = TensorComparator(
                rtol=rtol,
                atol=self.default_atol,
            )
            
            result = comparator.compare(
                actual[field],
                expected[field],
                name=f"{name}.{field}",
            )
            results.append(result)
        
        return results


@dataclass
class RegressionTest:
    """
    Definition of a regression test.
    
    Attributes:
        name: Test name
        generator: Function that generates the value to compare
        golden_name: Name of the golden value to compare against
        comparator: Comparator to use
        description: Test description
    """
    name: str
    generator: Callable[[], Any]
    golden_name: str
    comparator: Union[ArrayComparator, TensorComparator, StateComparator] = field(
        default_factory=TensorComparator
    )
    description: str = ""
    
    def run(self, store: GoldenValueStore) -> RegressionResult:
        """
        Run the regression test.
        
        Args:
            store: Golden value store
            
        Returns:
            RegressionResult
        """
        # Load golden value
        golden = store.load(self.golden_name)
        if golden is None:
            return RegressionResult(
                test_name=self.name,
                passed=False,
                comparison_type=ComparisonType.HYBRID,
                max_difference=float('inf'),
                mean_difference=float('inf'),
                tolerance_used=0.0,
                message=f"Golden value '{self.golden_name}' not found",
            )
        
        # Generate current value
        try:
            current = self.generator()
        except Exception as e:
            return RegressionResult(
                test_name=self.name,
                passed=False,
                comparison_type=ComparisonType.HYBRID,
                max_difference=float('inf'),
                mean_difference=float('inf'),
                tolerance_used=0.0,
                message=f"Generator failed: {str(e)}",
            )
        
        # Compare
        if isinstance(self.comparator, StateComparator):
            results = self.comparator.compare(current, golden.value, self.name)
            # Combine results
            all_passed = all(r.passed for r in results)
            max_diff = max(r.max_difference for r in results)
            return RegressionResult(
                test_name=self.name,
                passed=all_passed,
                comparison_type=ComparisonType.HYBRID,
                max_difference=max_diff,
                mean_difference=np.mean([r.mean_difference for r in results]),
                tolerance_used=self.comparator.default_rtol,
                message="Pass" if all_passed else "Field mismatch",
                details={'field_results': [r.to_dict() for r in results]},
            )
        else:
            return self.comparator.compare(current, golden.value, self.name)


@dataclass
class RegressionSuite:
    """
    Collection of regression tests.
    
    Manages multiple regression tests and provides
    summary reporting.
    """
    name: str
    store: GoldenValueStore
    tests: List[RegressionTest] = field(default_factory=list)
    results: List[RegressionResult] = field(default_factory=list)
    
    def add_test(self, test: RegressionTest):
        """Add a test to the suite."""
        self.tests.append(test)
    
    def run_all(self, verbose: bool = True) -> List[RegressionResult]:
        """
        Run all regression tests.
        
        Args:
            verbose: Print progress
            
        Returns:
            List of RegressionResult
        """
        self.results = []
        
        for test in self.tests:
            if verbose:
                print(f"Running: {test.name}...", end=" ")
            
            result = test.run(self.store)
            self.results.append(result)
            
            if verbose:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} (max diff: {result.max_difference:.2e})")
        
        return self.results
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return all(r.passed for r in self.results)
    
    @property
    def pass_count(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.passed)
    
    @property
    def fail_count(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if not r.passed)
    
    def report(self, format: str = "text") -> str:
        """
        Generate regression test report.
        
        Args:
            format: Output format ("text", "markdown")
            
        Returns:
            Formatted report
        """
        if format == "markdown":
            return self._report_markdown()
        return self._report_text()
    
    def _report_text(self) -> str:
        """Generate text report."""
        lines = [
            f"Regression Suite: {self.name}",
            "=" * 60,
            f"Passed: {self.pass_count}/{len(self.results)}",
            "",
        ]
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"[{status}] {result.test_name}")
            lines.append(f"       Max diff: {result.max_difference:.6e}")
            if result.message:
                lines.append(f"       {result.message}")
        
        return "\n".join(lines)
    
    def _report_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Regression Suite: {self.name}",
            "",
            f"**Pass Rate**: {self.pass_count}/{len(self.results)}",
            "",
            "| Test | Status | Max Diff | Message |",
            "|------|--------|----------|---------|",
        ]
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(
                f"| {result.test_name} | {status} | "
                f"{result.max_difference:.2e} | {result.message} |"
            )
        
        return "\n".join(lines)


def run_regression_tests(
    tests: List[RegressionTest],
    store: GoldenValueStore,
    verbose: bool = True,
) -> List[RegressionResult]:
    """
    Run a list of regression tests.
    
    Args:
        tests: List of regression tests
        store: Golden value store
        verbose: Print progress
        
    Returns:
        List of results
    """
    results = []
    
    for test in tests:
        if verbose:
            print(f"Running: {test.name}...", end=" ")
        
        result = test.run(store)
        results.append(result)
        
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status}")
    
    return results


def run_full_regression(
    suite: RegressionSuite,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Tuple[bool, List[RegressionResult]]:
    """
    Run full regression suite and optionally save report.
    
    Args:
        suite: Regression suite to run
        output_path: Optional path for report
        verbose: Print progress
        
    Returns:
        Tuple of (all_passed, results)
    """
    results = suite.run_all(verbose=verbose)
    
    if output_path:
        report = suite.report("markdown")
        with open(output_path, 'w') as f:
            f.write(report)
    
    return suite.all_passed, results
