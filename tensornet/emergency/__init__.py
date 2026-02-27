"""Backward-compatibility shim — real module at tensornet.applied.emergency.

This shim exists so that legacy imports like::

    from tensornet.emergency import X
    from tensornet.emergency.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.emergency``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.emergency")
_sys.modules[__name__] = _real
