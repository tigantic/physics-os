"""
QTT-SDK: Billion-Point Tensor Compression for Big Data and Digital Twins

This library provides Quantized Tensor-Train (QTT) compression enabling
operations on billion-point datasets using kilobytes of memory.

Example:
    >>> from qtt_sdk import dense_to_qtt, qtt_norm
    >>> import torch
    >>> signal = torch.sin(torch.linspace(0, 10, 2**20))
    >>> qtt = dense_to_qtt(signal, max_bond=32)
    >>> print(f"Norm: {qtt_norm(qtt):.4f}")
"""

__version__ = "1.0.0"
__author__ = "HyperTensor Team"

from qtt_sdk.core import (
    QTTState,
    MPO,
    dense_to_qtt,
    qtt_to_dense,
)

from qtt_sdk.operations import (
    qtt_add,
    qtt_scale,
    qtt_inner_product,
    qtt_norm,
    truncate_qtt,
)

from qtt_sdk.operators import (
    identity_mpo,
    shift_mpo,
    derivative_mpo,
    laplacian_mpo,
    apply_mpo,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "QTTState",
    "MPO",
    # Conversion
    "dense_to_qtt",
    "qtt_to_dense",
    # Arithmetic
    "qtt_add",
    "qtt_scale",
    "qtt_inner_product",
    "qtt_norm",
    "truncate_qtt",
    # Operators
    "identity_mpo",
    "shift_mpo",
    "derivative_mpo",
    "laplacian_mpo",
    "apply_mpo",
]
