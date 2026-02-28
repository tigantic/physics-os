"""Backward-compatibility shim — real module at ontic.applied.robotics_physics.

This shim exists so that legacy imports like::

    from ontic.robotics_physics import X
    from ontic.robotics_physics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.robotics_physics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.robotics_physics")
_sys.modules[__name__] = _real
