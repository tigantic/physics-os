"""
QTeneT Demos — Runnable Demonstrations of Curse-Breaking

Ready-to-run demonstrations that prove QTT breaks the curse
of dimensionality on real physics problems.

The holy_grail_6d demo is the flagship — it runs full 6D Vlasov-Maxwell
with 1 billion grid points using only ~100KB of memory.

Example:
    >>> from qtenet.demos import holy_grail_6d
    >>> 
    >>> # Run 6D Vlasov-Maxwell demonstration
    >>> results = holy_grail_6d()
    >>> print(f"Compression: {results.compression_ratio:,.0f}×")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from qtenet.demos.holy_grail import (
    holy_grail_6d,
    holy_grail_5d,
    HolyGrailResult,
)
from qtenet.demos.two_stream import (
    two_stream_instability,
    TwoStreamResult,
)

__all__ = [
    "holy_grail_6d",
    "holy_grail_5d",
    "HolyGrailResult",
    "two_stream_instability",
    "TwoStreamResult",
]
