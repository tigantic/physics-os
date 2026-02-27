"""Backward-compatibility shim — real module at tensornet.applied.medical.

This shim exists so that legacy imports like::

    from tensornet.medical import X
    from tensornet.medical.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.medical``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.medical")
_sys.modules[__name__] = _real
