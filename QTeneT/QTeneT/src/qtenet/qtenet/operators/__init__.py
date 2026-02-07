"""
QTeneT Operators — Curse-Breaking N-Dimensional Operators

The shift_nd operator is THE "Master Key" that unlocks arbitrary-dimensional
QTT simulations. It enables O(log N) complexity for shifts in any dimension
using Morton (Z-curve) interleaving.

Example:
    >>> from qtenet.operators import shift_nd, laplacian_nd, gradient_nd
    >>> 
    >>> # 5D phase space (32^5 = 33M points, only 25 qubits)
    >>> shift_x = shift_nd(total_qubits=25, num_dims=5, axis=0, direction=+1)
    >>> shift_vx = shift_nd(total_qubits=25, num_dims=5, axis=3, direction=-1)
    >>> 
    >>> # 6D Vlasov-Maxwell (32^6 = 1 billion points, only 30 qubits)
    >>> laplacian = laplacian_nd(total_qubits=30, num_dims=6)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from qtenet.operators.shift_nd import (
    shift_nd,
    apply_shift,
    make_shift_operators,
    ShiftConfig,
)
from qtenet.operators.laplacian_nd import (
    laplacian_nd,
    apply_laplacian,
)
from qtenet.operators.gradient_nd import (
    gradient_nd,
    apply_gradient,
)

__all__ = [
    # Shift (The Master Key)
    "shift_nd",
    "apply_shift",
    "make_shift_operators",
    "ShiftConfig",
    # Laplacian
    "laplacian_nd",
    "apply_laplacian",
    # Gradient
    "gradient_nd",
    "apply_gradient",
]
