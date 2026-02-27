"""Backward-compatibility shim — real module at tensornet.applied.robotics_physics.

This shim exists so that legacy imports like::

    from tensornet.robotics_physics import X
    from tensornet.robotics_physics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.robotics_physics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.robotics_physics")
_sys.modules[__name__] = _real
