"""Backward-compatibility shim — real module at tensornet.applied.acoustics.

This shim exists so that legacy imports like::

    from tensornet.acoustics import X
    from tensornet.acoustics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.acoustics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.acoustics")
_sys.modules[__name__] = _real
