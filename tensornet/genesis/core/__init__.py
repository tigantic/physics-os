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

from tensornet.genesis.core.rsvd import (
    rsvd_gpu,
    tt_decompose_rsvd,
    svd_fallback,
)

from tensornet.genesis.core.triton_ops import (
    rsvd_native,
    qtt_dot_native,
    qtt_norm_native,
    qtt_add_native,
    qtt_sub_native,
    qtt_hadamard_native,
    qtt_round_native,
    triton_matmul,
    triton_gram,
    adaptive_rank,
    qtt_evaluate_at_index,
    qtt_evaluate_at_indices,
    HAS_TRITON,
    HAS_CUDA,
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
    # rSVD (GPU-native)
    "rsvd_gpu",
    "tt_decompose_rsvd",
    "svd_fallback",
    # Triton QTT ops (native, no dense)
    "rsvd_native",
    "qtt_dot_native",
    "qtt_norm_native",
    "qtt_add_native",
    "qtt_sub_native",
    "qtt_hadamard_native",
    "qtt_round_native",
    "triton_matmul",
    "triton_gram",
    "adaptive_rank",
    "qtt_evaluate_at_index",
    "qtt_evaluate_at_indices",
    "HAS_TRITON",
    "HAS_CUDA",
]
