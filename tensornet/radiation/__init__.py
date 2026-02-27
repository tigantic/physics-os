"""Backward-compatibility shim — real module at tensornet.applied.radiation.

This shim exists so that legacy imports like::

    from tensornet.radiation import X
    from tensornet.radiation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.radiation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.radiation")
_sys.modules[__name__] = _real
