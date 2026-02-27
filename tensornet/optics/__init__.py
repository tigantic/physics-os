"""Backward-compatibility shim — real module at tensornet.applied.optics.

This shim exists so that legacy imports like::

    from tensornet.optics import X
    from tensornet.optics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.optics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.optics")
_sys.modules[__name__] = _real
