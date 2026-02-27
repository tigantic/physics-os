"""Backward-compatibility shim — real module at tensornet.applied.particle.

This shim exists so that legacy imports like::

    from tensornet.particle import X
    from tensornet.particle.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.particle``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.particle")
_sys.modules[__name__] = _real
