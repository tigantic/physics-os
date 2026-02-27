"""Backward-compatibility shim — real module at tensornet.applied.special_applied.

This shim exists so that legacy imports like::

    from tensornet.special_applied import X
    from tensornet.special_applied.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.special_applied``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.special_applied")
_sys.modules[__name__] = _real
