"""QTeneT Unified Public API.

This module defines the public interface contracts for QTT operations.
The stubs raise NotImplementedError by design - they serve as the unified
facade that downstream applications wire to specific implementations.

For direct QTT operations, use:
    - qtenet.tci.from_function() for compression
    - qtenet.operators.shift_nd() / apply_shift() for operations
    - qtenet.solvers for physics simulations

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class QTTMeta:
    dims: Sequence[int]
    ranks: Sequence[int] | None = None
    dtype: str | None = None
    layout: str | None = None


class QTTTensor(Protocol):
    """Protocol for a QTT tensor object."""

    meta: QTTMeta

    def round(self, *, eps: float, max_rank: int | None = None) -> "QTTTensor": ...

    def point_eval(self, index: Sequence[int]) -> Any: ...


def compress(obj: Any, **kwargs: Any) -> Any:
    """Compress array/stream into a QTT container.

    This is the unified facade entrypoint. For direct usage, see:
        qtenet.tci.from_function() - black-box function compression
        qtenet.tci.from_samples() - sample-based compression
    
    Raises:
        NotImplementedError: Facade stub - wire to implementation in application.
    """
    raise NotImplementedError("Use qtenet.tci.from_function() directly")


def query(container: Any, index: Sequence[int]) -> Any:
    """Query a single point from a compressed QTT container.
    
    Raises:
        NotImplementedError: Facade stub - wire to implementation in application.
    """
    raise NotImplementedError("Use tensornet.cfd.qtt_eval.qtt_evaluate() directly")


def reconstruct(container: Any, **kwargs: Any) -> Any:
    """Reconstruct dense output (escape hatch).
    
    Warning: This defeats the purpose of QTT compression. Only use for
    validation on small grids.
    
    Raises:
        NotImplementedError: Facade stub - wire to implementation in application.
    """
    raise NotImplementedError("Use tensornet.cfd.qtt_eval.qtt_to_dense() directly")
