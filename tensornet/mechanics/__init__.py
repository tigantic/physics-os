"""Backward-compatibility shim — real module at tensornet.materials.mechanics.

This shim exists so that legacy imports like::

    from tensornet.mechanics import X
    from tensornet.mechanics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.materials.mechanics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.materials.mechanics")
_sys.modules[__name__] = _real
