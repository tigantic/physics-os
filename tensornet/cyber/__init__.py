"""Backward-compatibility shim — real module at tensornet.applied.cyber.

This shim exists so that legacy imports like::

    from tensornet.cyber import X
    from tensornet.cyber.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.cyber``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.cyber")
_sys.modules[__name__] = _real
