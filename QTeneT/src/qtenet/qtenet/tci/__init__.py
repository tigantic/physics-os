"""
QTeneT TCI — Black-Box Function to QTT Construction

TCI (Tensor Cross Interpolation) is THE critical algorithm for curse-breaking.
It builds QTT representations from black-box functions using only O(r² × log N)
function evaluations — never O(N).

This is how you compress YOUR functions to QTT format.

Example:
    >>> from qtenet.tci import from_function
    >>> import torch
    >>> 
    >>> # Define any function (can be expensive, e.g., physics simulation)
    >>> def my_function(indices: torch.Tensor) -> torch.Tensor:
    ...     # Evaluate at batch of indices
    ...     return torch.sin(indices.float() / 1000)
    >>> 
    >>> # Build QTT with O(log N) complexity
    >>> qtt_cores = from_function(
    ...     f=my_function,
    ...     n_qubits=30,      # 2^30 = 1 billion points
    ...     max_rank=64,
    ... )

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from qtenet.tci.from_function import (
    from_function,
    from_function_2d,
    from_function_nd,
    TCIConfig,
    TCIResult,
)
from qtenet.tci.from_samples import (
    from_samples,
    from_sparse_samples,
)

__all__ = [
    # Primary API
    "from_function",
    "from_function_2d",
    "from_function_nd",
    # Configuration
    "TCIConfig",
    "TCIResult",
    # Sample-based
    "from_samples",
    "from_sparse_samples",
]
