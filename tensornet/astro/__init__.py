"""
Astrophysics package: compact objects.

Domain: XII.2.
"""

from .compact_objects import (
    NeutronStarEOS,
    TOVSolver,
    KerrBlackHole,
    ShakuraSunyaevDisk,
)

__all__ = [
    "NeutronStarEOS",
    "TOVSolver",
    "KerrBlackHole",
    "ShakuraSunyaevDisk",
]
