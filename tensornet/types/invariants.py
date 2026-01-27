"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                      I N V A R I A N T S   M O D U L E                                  ║
║                                                                                          ║
║     The TypeChecker for geometric objects.                                              ║
║     Verifies that mathematical invariants hold.                                         ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

This module provides:

    TypeChecker       - Batch verification of constraints
    verify_invariant  - Single invariant check with rich diagnostics
    InvariantViolation - Exception for constraint failures

The key insight: constraints are not just documentation.
They are RUNTIME GUARANTEES that the type system enforces.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Optional, Tuple, Dict, Any,
    List, Union, Type, Callable, Sequence
)
from enum import Enum, auto
import torch
from torch import Tensor

from tensornet.types.constraints import Constraint


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

F = TypeVar("F")  # Field type


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT VIOLATION
# ═══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    """Severity level of invariant violation."""
    WARNING = auto()    # Soft violation, computation may continue
    ERROR = auto()      # Hard violation, computation should stop
    CRITICAL = auto()   # Critical failure, data may be corrupted


@dataclass
class InvariantViolation(Exception):
    """
    Exception raised when a geometric invariant is violated.
    
    This is not a bug - it's the type system catching physics errors.
    
    Attributes:
        constraint: The constraint that was violated
        field_name: Name of the field/object that failed
        expected: What the constraint expected
        actual: What was actually measured
        severity: How bad is this?
        context: Additional diagnostic information
    """
    
    constraint: Constraint
    field_name: str = ""
    expected: Any = None
    actual: Any = None
    severity: Severity = Severity.ERROR
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        msg = [
            f"\n{'='*70}",
            f"INVARIANT VIOLATION [{self.severity.name}]",
            f"{'='*70}",
            f"Constraint: {self.constraint}",
        ]
        
        if self.field_name:
            msg.append(f"Field: {self.field_name}")
        
        if self.expected is not None:
            msg.append(f"Expected: {self.expected}")
        
        if self.actual is not None:
            if isinstance(self.actual, Tensor):
                msg.append(f"Actual: tensor with max|val|={self.actual.abs().max().item():.2e}")
            else:
                msg.append(f"Actual: {self.actual}")
        
        if self.context:
            msg.append("Context:")
            for k, v in self.context.items():
                msg.append(f"  {k}: {v}")
        
        msg.append(f"{'='*70}")
        
        return "\n".join(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """
    Result of invariant verification.
    
    Provides detailed diagnostics even for passing checks.
    """
    
    passed: bool
    constraint: Constraint
    field_name: str = ""
    residual: Optional[float] = None  # How close to passing/failing
    message: str = ""
    suggestions: List[str] = field(default_factory=list)
    computation_time_ms: float = 0.0
    
    def __bool__(self) -> bool:
        return self.passed
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f"{status}: {self.constraint}"
        
        if self.field_name:
            msg += f" on {self.field_name}"
        
        if self.residual is not None:
            msg += f" (residual={self.residual:.2e})"
        
        return msg


@dataclass
class BatchVerificationResult:
    """
    Result of verifying multiple constraints.
    """
    
    results: List[VerificationResult]
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def failures(self) -> List[VerificationResult]:
        return [r for r in self.results if not r.passed]
    
    def __bool__(self) -> bool:
        return self.all_passed
    
    def __str__(self) -> str:
        lines = [
            f"Verification: {self.passed_count}/{len(self.results)} passed"
        ]
        
        for r in self.results:
            lines.append(f"  {r}")
        
        return "\n".join(lines)
    
    def raise_if_failed(self) -> None:
        """Raise InvariantViolation if any checks failed."""
        if not self.all_passed:
            failure = self.failures[0]
            raise InvariantViolation(
                constraint=failure.constraint,
                field_name=failure.field_name,
                expected=f"residual < tolerance",
                actual=failure.residual,
                context={"all_failures": self.failures}
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypeChecker:
    """
    Type checker for geometric objects.
    
    Verifies that fields satisfy their declared constraints.
    
    Usage:
        checker = TypeChecker(tolerance=1e-6)
        result = checker.verify(flow, [Divergence(0)])
        if not result:
            print(result)
            result.raise_if_failed()
    
    The TypeChecker can be configured with:
        - tolerance: Numerical tolerance for constraint checks
        - strict_mode: Whether to raise immediately on first failure
        - collect_diagnostics: Whether to compute detailed diagnostics
    """
    
    tolerance: float = 1e-6
    strict_mode: bool = False
    collect_diagnostics: bool = True
    
    def verify(
        self,
        field: Any,
        constraints: Optional[Sequence[Constraint]] = None,
        field_name: str = ""
    ) -> BatchVerificationResult:
        """
        Verify that a field satisfies constraints.
        
        Args:
            field: The geometric object to verify
            constraints: Constraints to check (default: field's declared constraints)
            field_name: Name for error messages
            
        Returns:
            BatchVerificationResult with all check results
        """
        import time
        
        # Get constraints from field if not specified
        if constraints is None:
            if hasattr(field, "constraints"):
                constraints = field.constraints
            else:
                constraints = []
        
        results = []
        
        for constraint in constraints:
            start = time.perf_counter()
            
            try:
                passed = constraint.verify(field, tolerance=self.tolerance)
                residual = self._compute_residual(constraint, field)
                
                result = VerificationResult(
                    passed=passed,
                    constraint=constraint,
                    field_name=field_name,
                    residual=residual,
                    computation_time_ms=(time.perf_counter() - start) * 1000
                )
                
            except Exception as e:
                result = VerificationResult(
                    passed=False,
                    constraint=constraint,
                    field_name=field_name,
                    message=f"Exception during verification: {e}",
                    computation_time_ms=(time.perf_counter() - start) * 1000
                )
            
            results.append(result)
            
            if self.strict_mode and not result.passed:
                batch = BatchVerificationResult(results=results)
                batch.raise_if_failed()
        
        return BatchVerificationResult(results=results)
    
    def verify_single(
        self,
        field: Any,
        constraint: Constraint,
        field_name: str = ""
    ) -> VerificationResult:
        """Verify a single constraint."""
        return self.verify(field, [constraint], field_name).results[0]
    
    def _compute_residual(self, constraint: Constraint, field: Any) -> Optional[float]:
        """
        Compute how close the field is to satisfying the constraint.
        
        Returns a non-negative number where 0 = perfect satisfaction.
        """
        from tensornet.types.constraints import Divergence, Curl, Normalized
        
        if not hasattr(field, "data"):
            return None
        
        data = field.data
        
        # Compute constraint-specific residuals
        if isinstance(constraint, Divergence):
            if hasattr(field, "divergence"):
                div = field.divergence()
                return float(div.abs().max().item())
        
        elif isinstance(constraint, Curl):
            if hasattr(field, "curl"):
                curl_field = field.curl()
                return float(curl_field.data.abs().max().item())
        
        elif isinstance(constraint, Normalized):
            norm = torch.norm(data, dim=-1)
            return float((norm - 1.0).abs().max().item())
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_invariant(
    field: Any,
    constraint: Constraint,
    *,
    tolerance: float = 1e-6,
    raise_on_failure: bool = True,
    field_name: str = ""
) -> VerificationResult:
    """
    Verify that a field satisfies a constraint.
    
    This is the primary entry point for constraint checking.
    
    Args:
        field: The geometric object
        constraint: The constraint to verify
        tolerance: Numerical tolerance
        raise_on_failure: Whether to raise InvariantViolation on failure
        field_name: Name for error messages
        
    Returns:
        VerificationResult
        
    Raises:
        InvariantViolation: If constraint fails and raise_on_failure=True
        
    Example:
        >>> flow = VectorField[R3, Divergence(0)](data=velocity)
        >>> verify_invariant(flow, Divergence(0))
        VerificationResult(passed=True, ...)
    """
    checker = TypeChecker(tolerance=tolerance, strict_mode=raise_on_failure)
    return checker.verify_single(field, constraint, field_name)


def verify_all(
    field: Any,
    *,
    tolerance: float = 1e-6,
    raise_on_failure: bool = True,
    field_name: str = ""
) -> BatchVerificationResult:
    """
    Verify all declared constraints on a field.
    
    Args:
        field: The geometric object
        tolerance: Numerical tolerance
        raise_on_failure: Whether to raise on first failure
        field_name: Name for error messages
        
    Returns:
        BatchVerificationResult
    """
    checker = TypeChecker(
        tolerance=tolerance,
        strict_mode=raise_on_failure
    )
    return checker.verify(field, field_name=field_name)


def assert_invariant(
    field: Any,
    constraint: Constraint,
    message: str = ""
) -> None:
    """
    Assert that a constraint holds. Raises if not.
    
    Use this for critical invariants that must hold.
    
    Example:
        >>> assert_invariant(flow, Divergence(0), "flow must be incompressible")
    """
    result = verify_invariant(
        field, constraint,
        raise_on_failure=False
    )
    
    if not result.passed:
        full_message = message or f"Invariant assertion failed: {constraint}"
        raise InvariantViolation(
            constraint=constraint,
            expected="constraint to pass",
            actual=f"residual = {result.residual}",
            severity=Severity.CRITICAL,
            context={"message": full_message}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def implies(
    antecedent: Constraint,
    consequent: Constraint
) -> Callable[[Any], bool]:
    """
    Check that one constraint implies another.
    
    This is for verifying constraint relationships.
    
    Example:
        If Divergence(0) → Laplacian(0) for some field class
    """
    def check(field: Any) -> bool:
        if antecedent.verify(field):
            return consequent.verify(field)
        return True  # Vacuously true if antecedent fails
    
    return check


def consistent(
    constraints: Sequence[Constraint]
) -> Callable[[Any], bool]:
    """
    Check that a set of constraints is consistent (can all be satisfied).
    """
    def check(field: Any) -> bool:
        return all(c.verify(field) for c in constraints)
    
    return check


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose(field: Any, verbose: bool = True) -> Dict[str, Any]:
    """
    Compute comprehensive diagnostics for a field.
    
    Returns:
        Dictionary with:
            - shape: Data shape
            - dtype: Data type
            - constraints: Declared constraints
            - constraint_status: Status of each constraint
            - statistics: Min, max, mean, std of data
            - norms: Various norms of the data
    """
    info: Dict[str, Any] = {}
    
    if hasattr(field, "data"):
        data = field.data
        info["shape"] = tuple(data.shape)
        info["dtype"] = str(data.dtype)
        info["device"] = str(data.device)
        info["statistics"] = {
            "min": float(data.min().item()),
            "max": float(data.max().item()),
            "mean": float(data.mean().item()),
            "std": float(data.std().item()),
        }
        info["norms"] = {
            "L1": float(data.abs().sum().item()),
            "L2": float(data.norm().item()),
            "Linf": float(data.abs().max().item()),
        }
    
    if hasattr(field, "constraints"):
        info["constraints"] = [str(c) for c in field.constraints]
        info["constraint_status"] = {}
        
        checker = TypeChecker()
        for c in field.constraints:
            result = checker.verify_single(field, c)
            info["constraint_status"][str(c)] = {
                "passed": result.passed,
                "residual": result.residual
            }
    
    if hasattr(field, "space"):
        info["space"] = str(field.space)
    
    if verbose:
        print(f"\n{'='*50}")
        print("FIELD DIAGNOSTICS")
        print(f"{'='*50}")
        for k, v in info.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for kk, vv in v.items():
                    print(f"  {kk}: {vv}")
            else:
                print(f"{k}: {v}")
        print(f"{'='*50}\n")
    
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

def is_divergence_free(field: Any, tolerance: float = 1e-6) -> bool:
    """Check if field is divergence-free."""
    from tensornet.types.constraints import Divergence
    return Divergence(0).verify(field, tolerance=tolerance)


def is_curl_free(field: Any, tolerance: float = 1e-6) -> bool:
    """Check if field is curl-free (irrotational)."""
    from tensornet.types.constraints import Curl
    return Curl(0).verify(field, tolerance=tolerance)


def is_normalized(field: Any, tolerance: float = 1e-6) -> bool:
    """Check if field is normalized."""
    from tensornet.types.constraints import Normalized
    return Normalized().verify(field, tolerance=tolerance)


def is_positive(field: Any) -> bool:
    """Check if field values are positive."""
    from tensornet.types.constraints import Positive
    return Positive().verify(field)


def is_symmetric(field: Any, tolerance: float = 1e-6) -> bool:
    """Check if tensor field is symmetric."""
    from tensornet.types.constraints import Symmetric
    return Symmetric().verify(field, tolerance=tolerance)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core classes
    "Severity",
    "InvariantViolation",
    "VerificationResult",
    "BatchVerificationResult",
    "TypeChecker",
    # Verification functions
    "verify_invariant",
    "verify_all",
    "assert_invariant",
    # Constraint composition
    "implies",
    "consistent",
    # Diagnostics
    "diagnose",
    # Type guards
    "is_divergence_free",
    "is_curl_free",
    "is_normalized",
    "is_positive",
    "is_symmetric",
]
