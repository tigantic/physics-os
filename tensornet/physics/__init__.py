"""Backward-compatibility shim — real module at tensornet.applied.physics.

This shim exists so that legacy imports like::

    from tensornet.physics import X
    from tensornet.physics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.physics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.physics")
_sys.modules[__name__] = _real
