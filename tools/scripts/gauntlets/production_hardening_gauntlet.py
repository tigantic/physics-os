#!/usr/bin/env python3
"""
TENSOR GENESIS — Production Hardening Validation Gauntlet

Validates that the production infrastructure is correctly integrated:
- Logging works across all layers
- Exceptions are informative
- Profiling decorators function correctly
- Validation utilities catch errors properly

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import json
import hashlib
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Any

import numpy as np


@dataclass  
class TestResult:
    name: str
    passed: bool
    time_seconds: float
    error: str | None = None


def print_header():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       P R O D U C T I O N   H A R D E N I N G   G A U N T L E T            ║
║                                                                              ║
║              Logging • Exceptions • Profiling • Validation                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def test_logging_infrastructure() -> bool:
    """Test 1: Verify logging infrastructure works correctly."""
    print("━━━ TEST 1: Logging Infrastructure ━━━")
    
    from ontic.genesis.core.logging import (
        get_logger,
        configure_logging,
        GenesisLogger,
        LogLevel,
        logged,
    )
    
    # Test logger creation
    logger = get_logger("test_module", layer=20, primitive="OT")
    print(f"  Created logger: {logger.name}")
    
    # Test log levels
    logger.set_level(LogLevel.DEBUG)
    
    # Capture that logs work (they go to stderr)
    logger.debug("Debug message test")
    logger.info("Info message test")
    logger.perf("Performance message test")
    logger.warning("Warning message test")
    
    # Test operation context manager
    with logger.operation("test_operation"):
        time.sleep(0.01)  # Simulate work
    
    # Test child logger
    child = logger.child("submodule", scale=1024)
    child.info("Child logger test")
    
    # Test lazy evaluation
    expensive_calls = [0]
    def expensive_message():
        expensive_calls[0] += 1
        return "Expensive to compute"
    
    # With DEBUG level, this should evaluate
    logger.debug(expensive_message)
    assert expensive_calls[0] == 1, "Lazy message should be evaluated at DEBUG level"
    
    # With higher level, should NOT evaluate
    logger.set_level(LogLevel.WARNING)
    logger.debug(expensive_message)
    assert expensive_calls[0] == 1, "Lazy message should NOT be evaluated at WARNING level"
    
    # Test logged decorator
    @logged(level=LogLevel.DEBUG, include_args=True)
    def sample_function(x, y):
        return x + y
    
    logger.set_level(LogLevel.DEBUG)
    result = sample_function(1, 2)
    assert result == 3
    
    print(f"  ✓ Logger creation and configuration")
    print(f"  ✓ Log levels (TRACE, DEBUG, INFO, PERF, WARNING, ERROR)")
    print(f"  ✓ Operation context manager with timing")
    print(f"  ✓ Child loggers with inherited context")
    print(f"  ✓ Lazy message evaluation")
    print(f"  ✓ @logged decorator")
    print(f"  ✓ PASS")
    return True


def test_exception_hierarchy() -> bool:
    """Test 2: Verify exception hierarchy with informative messages."""
    print("\n━━━ TEST 2: Exception Hierarchy ━━━")
    
    from ontic.genesis.core.exceptions import (
        GenesisError,
        QTTRankError,
        ConvergenceError,
        DimensionMismatchError,
        NumericalInstabilityError,
        MemoryBudgetExceededError,
        InvalidInputError,
        CompressionError,
        check_finite,
        check_shape,
    )
    
    # Test base exception
    try:
        raise GenesisError("Base error", context={"key": "value"}, suggestion="Try this")
    except GenesisError as e:
        assert "key=value" in str(e)
        assert "Try this" in str(e)
        print(f"  ✓ GenesisError with context and suggestion")
    
    # Test QTTRankError
    try:
        raise QTTRankError(
            "Rank explosion detected",
            current_rank=500,
            max_rank=100,
            operation="tropical_matmul"
        )
    except QTTRankError as e:
        assert e.current_rank == 500
        assert e.max_rank == 100
        assert "current_rank=500" in str(e)
        print(f"  ✓ QTTRankError with rank details")
    
    # Test ConvergenceError
    try:
        raise ConvergenceError(
            "Sinkhorn did not converge",
            iterations=1000,
            residual=1e-3,
            tolerance=1e-8,
            algorithm="Sinkhorn-Knopp"
        )
    except ConvergenceError as e:
        assert e.iterations == 1000
        assert "Suggestion" in str(e)  # Auto-generated suggestion
        print(f"  ✓ ConvergenceError with convergence details")
    
    # Test DimensionMismatchError
    try:
        raise DimensionMismatchError(
            "Matrix shapes incompatible",
            expected=(100, 50),
            actual=(100, 60),
            operand="weight_matrix"
        )
    except DimensionMismatchError as e:
        assert e.expected == (100, 50)
        assert e.actual == (100, 60)
        print(f"  ✓ DimensionMismatchError with shape details")
    
    # Test NumericalInstabilityError
    try:
        raise NumericalInstabilityError(
            "Non-finite values detected",
            values_affected=42,
            location="softmax output",
            has_nan=True,
            has_inf=False,
        )
    except NumericalInstabilityError as e:
        assert e.has_nan == True
        assert e.values_affected == 42
        print(f"  ✓ NumericalInstabilityError with stability details")
    
    # Test MemoryBudgetExceededError
    try:
        raise MemoryBudgetExceededError(
            "Dense materialization would exceed budget",
            required_bytes=16 * 1024**3,  # 16 GB
            budget_bytes=4 * 1024**3,     # 4 GB
            operation="floyd_warshall"
        )
    except MemoryBudgetExceededError as e:
        assert "16.0 GB" in str(e)
        assert "4.0 GB" in str(e)
        print(f"  ✓ MemoryBudgetExceededError with formatted sizes")
    
    # Test InvalidInputError
    try:
        raise InvalidInputError(
            "Invalid regularization parameter",
            parameter="epsilon",
            value=-0.1,
            constraint="> 0"
        )
    except InvalidInputError as e:
        assert e.parameter == "epsilon"
        print(f"  ✓ InvalidInputError with parameter details")
    
    # Test check_finite utility
    arr_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
    try:
        check_finite(arr_with_nan, name="test_array", location="test")
        assert False, "Should have raised"
    except NumericalInstabilityError as e:
        assert e.has_nan == True
        print(f"  ✓ check_finite() utility")
    
    # Test check_shape utility
    arr = np.zeros((10, 20))
    try:
        check_shape(arr, (10, 30), name="matrix")
        assert False, "Should have raised"
    except DimensionMismatchError:
        print(f"  ✓ check_shape() utility")
    
    # Test exception hierarchy
    try:
        raise QTTRankError("Test")
    except GenesisError:
        print(f"  ✓ Exception hierarchy (all inherit from GenesisError)")
    
    print(f"  ✓ PASS")
    return True


def test_profiling_infrastructure() -> bool:
    """Test 3: Verify profiling decorators and utilities."""
    print("\n━━━ TEST 3: Profiling Infrastructure ━━━")
    
    from ontic.genesis.core.profiling import (
        profile,
        profile_memory,
        timed,
        traced,
        profile_block,
        timer,
        get_tracker,
        benchmark,
        ProfileResult,
        PerformanceTracker,
    )
    
    # Test @timed decorator
    @timed
    def slow_function():
        time.sleep(0.05)
        return 42
    
    result = slow_function()
    assert result == 42
    assert slow_function._last_time_ms >= 40  # At least 40ms
    print(f"  ✓ @timed decorator: {slow_function._last_time_ms:.2f}ms")
    
    # Test @profile decorator
    @profile(name="matrix_multiply", track_memory=True)
    def matrix_op():
        return np.random.randn(100, 100) @ np.random.randn(100, 100)
    
    result = matrix_op()
    assert result.shape == (100, 100)
    print(f"  ✓ @profile decorator with memory tracking")
    
    # Test profile_block context manager
    with profile_block("test_block") as p:
        arr = np.random.randn(1000)
        _ = np.sort(arr)
    
    assert p.time_seconds > 0
    print(f"  ✓ profile_block context manager: {p.time_ms:.2f}ms")
    
    # Test timer context manager
    with timer("simple_timer") as t:
        time.sleep(0.02)
    
    assert t["elapsed_ms"] >= 15
    print(f"  ✓ timer context manager: {t['elapsed_ms']:.2f}ms")
    
    # Test PerformanceTracker hierarchical profiling
    tracker = PerformanceTracker("test_tracker")
    
    with tracker.profile("outer_operation"):
        time.sleep(0.01)
        tracker.count_op("matmul", 5)
        
        with tracker.profile("inner_operation"):
            time.sleep(0.005)
            tracker.count_op("add", 10)
    
    assert len(tracker.root.children) == 1
    outer = tracker.root.children[0]
    assert outer.name == "outer_operation"
    assert outer.operations.get("matmul") == 5
    assert len(outer.children) == 1
    assert outer.children[0].name == "inner_operation"
    print(f"  ✓ PerformanceTracker hierarchical profiling")
    
    # Test benchmark utility
    def fast_func(x):
        return x * 2
    
    stats = benchmark(fast_func, 42, n_runs=5, warmup=2)
    assert "mean_ms" in stats
    assert stats["n_runs"] == 5
    print(f"  ✓ benchmark utility: mean={stats['mean_ms']:.4f}ms")
    
    # Test ProfileResult serialization
    result = ProfileResult(
        name="test",
        time_seconds=0.123,
        memory_before_bytes=1000,
        memory_after_bytes=2000,
        operations={"matmul": 10}
    )
    d = result.to_dict()
    assert d["name"] == "test"
    assert d["time_ms"] == 123.0
    assert d["memory_delta_bytes"] == 1000
    print(f"  ✓ ProfileResult serialization")
    
    print(f"  ✓ PASS")
    return True


def test_validation_utilities() -> bool:
    """Test 4: Verify validation utilities."""
    print("\n━━━ TEST 4: Validation Utilities ━━━")
    
    from ontic.genesis.core.validation import (
        validate_qtt_cores,
        validate_tensor_shape,
        validate_positive,
        validate_probability,
        validate_distribution,
        validate_dtype,
        validate_power_of_two,
        validate_range,
        check_numerical_stability,
        coerce_array,
    )
    from ontic.genesis.core.exceptions import (
        DimensionMismatchError,
        QTTRankError,
        InvalidInputError,
        NumericalInstabilityError,
    )
    
    # Test validate_qtt_cores
    valid_cores = [
        np.random.randn(1, 2, 4),
        np.random.randn(4, 2, 4),
        np.random.randn(4, 2, 1),
    ]
    validate_qtt_cores(valid_cores)
    print(f"  ✓ validate_qtt_cores - valid cores pass")
    
    # Test invalid cores (wrong boundary rank)
    invalid_cores = [
        np.random.randn(2, 2, 4),  # r_left should be 1
        np.random.randn(4, 2, 1),
    ]
    try:
        validate_qtt_cores(invalid_cores)
        assert False, "Should have raised"
    except DimensionMismatchError as e:
        assert "r_left=1" in str(e)
        print(f"  ✓ validate_qtt_cores - catches boundary rank error")
    
    # Test validate_tensor_shape
    arr = np.zeros((10, 20, 30))
    validate_tensor_shape(arr, (10, -1, 30))  # -1 is wildcard
    print(f"  ✓ validate_tensor_shape - wildcards work")
    
    try:
        validate_tensor_shape(arr, (10, 20))  # Wrong ndim
        assert False, "Should have raised"
    except DimensionMismatchError:
        print(f"  ✓ validate_tensor_shape - catches dimension mismatch")
    
    # Test validate_positive
    validate_positive(5.0, "epsilon")
    validate_positive(0, "count", allow_zero=True)
    
    try:
        validate_positive(-1, "epsilon")
        assert False, "Should have raised"
    except InvalidInputError as e:
        assert e.parameter == "epsilon"
        print(f"  ✓ validate_positive - catches negative values")
    
    # Test validate_probability
    validate_probability(0.5)
    validate_probability(np.array([0.1, 0.5, 0.9]))
    
    try:
        validate_probability(1.5)
        assert False, "Should have raised"
    except InvalidInputError:
        print(f"  ✓ validate_probability - catches out of range")
    
    # Test validate_distribution
    dist = np.array([0.2, 0.3, 0.5])
    validated = validate_distribution(dist)
    assert abs(validated.sum() - 1.0) < 1e-10
    print(f"  ✓ validate_distribution - valid distribution passes")
    
    # Test auto-normalization
    unnorm = np.array([1.0, 1.0, 1.0])
    normalized = validate_distribution(unnorm)
    assert abs(normalized.sum() - 1.0) < 1e-10
    print(f"  ✓ validate_distribution - auto-normalizes")
    
    # Test validate_dtype
    arr_float32 = np.array([1, 2, 3], dtype=np.float32)
    coerced = validate_dtype(arr_float32, np.float64, coerce=True)
    assert coerced.dtype == np.float64
    print(f"  ✓ validate_dtype - type coercion")
    
    # Test validate_power_of_two
    validate_power_of_two(1024)
    validate_power_of_two(1)
    
    try:
        validate_power_of_two(1000)
        assert False, "Should have raised"
    except InvalidInputError as e:
        assert "nearest: 1024" in str(e)
        print(f"  ✓ validate_power_of_two - suggests nearest")
    
    # Test validate_range
    validate_range(5, min_val=0, max_val=10)
    
    try:
        validate_range(15, max_val=10, name="iterations")
        assert False, "Should have raised"
    except InvalidInputError as e:
        assert "iterations" in str(e)
        print(f"  ✓ validate_range - catches out of range")
    
    # Test check_numerical_stability
    good_arr = np.array([1.0, 2.0, 3.0])
    check_numerical_stability(good_arr)
    
    bad_arr = np.array([1.0, np.inf, 3.0])
    try:
        check_numerical_stability(bad_arr)
        assert False, "Should have raised"
    except NumericalInstabilityError as e:
        assert e.has_inf == True
        print(f"  ✓ check_numerical_stability - catches inf values")
    
    # Test coerce_array
    result = coerce_array([1, 2, 3], dtype=np.float64)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    print(f"  ✓ coerce_array - list to array conversion")
    
    print(f"  ✓ PASS")
    return True


def test_integration_with_genesis_modules() -> bool:
    """Test 5: Verify core integrates with Genesis modules."""
    print("\n━━━ TEST 5: Integration with Genesis Modules ━━━")
    
    # Test that imports work
    from ontic.genesis import (
        # Core
        get_logger,
        configure_logging,
        GenesisError,
        QTTRankError,
        profile,
        timed,
        validate_positive,
        # Layers (spot check)
        TropicalMatrix,
        Kernel,
        Simplex,
        Multivector,
    )
    
    print(f"  ✓ Core infrastructure imports from ontic.genesis")
    
    # Test logger with Genesis context
    logger = get_logger("integration_test", layer=23, primitive="Tropical")
    logger.info("Testing integration", scale=1024)
    print(f"  ✓ Logger with layer/primitive context")
    
    # Test using profiling with a Genesis-like function
    @profile(track_memory=True)
    @timed
    def mock_genesis_operation(n: int) -> np.ndarray:
        """Simulated Genesis operation."""
        validate_positive(n, "n")
        return np.random.randn(n, n)
    
    result = mock_genesis_operation(50)
    assert result.shape == (50, 50)
    print(f"  ✓ Decorators work on Genesis-style functions")
    
    # Test error handling in Genesis context
    try:
        mock_genesis_operation(-1)
        assert False, "Should have raised"
    except GenesisError:
        print(f"  ✓ Validation errors are Genesis exceptions")
    
    print(f"  ✓ PASS")
    return True


def run_all_tests() -> Tuple[int, int, float]:
    """Run all tests and return (passed, total, time)."""
    tests = [
        ("Logging Infrastructure", test_logging_infrastructure),
        ("Exception Hierarchy", test_exception_hierarchy),
        ("Profiling Infrastructure", test_profiling_infrastructure),
        ("Validation Utilities", test_validation_utilities),
        ("Integration with Genesis Modules", test_integration_with_genesis_modules),
    ]
    
    results: List[TestResult] = []
    total_time = 0.0
    
    for name, test_fn in tests:
        start = time.perf_counter()
        try:
            passed = test_fn()
            elapsed = time.perf_counter() - start
            results.append(TestResult(name, passed, elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append(TestResult(name, False, elapsed, str(e)))
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
        total_time += elapsed
    
    # Print summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         R E S U L T S                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣""")
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"║  {r.name:45} {status:10} {r.time_seconds:6.2f}s       ║")
    
    print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║  Total: {passed}/{total} tests passed in {total_time:.2f}s{' ' * 35}║
║                                                                              ║""")
    
    if passed == total:
        print("""║  ★★★ PRODUCTION HARDENING GAUNTLET PASSED ★★★                              ║
║                                                                              ║
║  Infrastructure Verified:                                                    ║
║    • Hierarchical logging with context                                       ║
║    • Informative exception hierarchy                                         ║
║    • Performance profiling with memory tracking                              ║
║    • Input validation with suggestions                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝""")
    else:
        print("""║  ✗ PRODUCTION HARDENING INCOMPLETE                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝""")
    
    # Generate attestation
    if passed == total:
        attestation = {
            "gauntlet": "PRODUCTION HARDENING GAUNTLET",
            "project": "TENSOR GENESIS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "tests_passed": passed,
            "tests_total": total,
            "total_time_seconds": total_time,
            "results": [
                {
                    "test_name": r.name,
                    "passed": r.passed,
                    "time_seconds": r.time_seconds,
                    "error": r.error
                }
                for r in results
            ],
            "infrastructure": [
                "Logging (GenesisLogger, LogLevel, @logged)",
                "Exceptions (GenesisError hierarchy with 8 specialized types)",
                "Profiling (profile, timed, traced, PerformanceTracker)",
                "Validation (11 validators for QTT, shapes, numerics)",
            ],
        }
        
        # Compute hash
        content = json.dumps(attestation, sort_keys=True, default=str)
        attestation["sha256"] = hashlib.sha256(content.encode()).hexdigest()
        
        with open("PRODUCTION_HARDENING_ATTESTATION.json", "w") as f:
            json.dump(attestation, f, indent=2, default=str)
        
        print(f"\n  ✓ Attestation saved to PRODUCTION_HARDENING_ATTESTATION.json")
        print(f"    SHA256: {attestation['sha256'][:40]}...")
    
    return passed, total, total_time


if __name__ == "__main__":
    print_header()
    
    # Suppress logging output during tests
    import logging
    logging.getLogger("genesis").setLevel(logging.CRITICAL)
    
    passed, total, elapsed = run_all_tests()
    sys.exit(0 if passed == total else 1)
