"""
TensorNet Fusion Module - Magnetically Confined Plasma

Phase 9: The Tokamak Twin
- Toroidal magnetic geometry (the "donut")
- Boris particle pusher for ion dynamics
- Magnetic bottle confinement verification

The most complex physics problem in the universe,
now running on your GPU.

F = q(E + v × B) - The Lorentz Force that confines stars.
"""

from .tokamak import TokamakReactor, PlasmaState, ConfinementReport

__all__ = [
    "TokamakReactor",
    "PlasmaState",
    "ConfinementReport",
]
