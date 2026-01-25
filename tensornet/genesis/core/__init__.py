"""
TENSOR GENESIS Core Infrastructure

Production-hardened utilities for all Genesis primitives:
- Logging with configurable levels
- Custom exceptions with informative messages  
- Performance profiling decorators
- Type validation utilities

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from tensornet.genesis.core.logging import (
    get_logger,
    configure_logging,
    GenesisLogger,
    LogLevel,
)

from tensornet.genesis.core.exceptions import (
    GenesisError,
    QTTRankError,
    ConvergenceError,
    DimensionMismatchError,
    NumericalInstabilityError,
    MemoryBudgetExceededError,
    InvalidInputError,
    CompressionError,
)

from tensornet.genesis.core.profiling import (
    profile,
    profile_memory,
    timed,
    traced,
    ProfileResult,
    PerformanceTracker,
)

from tensornet.genesis.core.validation import (
    validate_qtt_cores,
    validate_tensor_shape,
    validate_positive,
    validate_probability,
    validate_dtype,
    check_numerical_stability,
)

__all__ = [
    # Logging
    "get_logger",
    "configure_logging",
    "GenesisLogger",
    "LogLevel",
    # Exceptions
    "GenesisError",
    "QTTRankError",
    "ConvergenceError",
    "DimensionMismatchError",
    "NumericalInstabilityError",
    "MemoryBudgetExceededError",
    "InvalidInputError",
    "CompressionError",
    # Profiling
    "profile",
    "profile_memory",
    "timed",
    "traced",
    "ProfileResult",
    "PerformanceTracker",
    # Validation
    "validate_qtt_cores",
    "validate_tensor_shape",
    "validate_positive",
    "validate_probability",
    "validate_dtype",
    "check_numerical_stability",
]
