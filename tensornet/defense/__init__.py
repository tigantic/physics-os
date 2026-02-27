"""Backward-compatibility shim — real module at tensornet.aerospace.defense.

This shim exists so that legacy imports like::

    from tensornet.defense import X
    from tensornet.defense.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.aerospace.defense``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.aerospace.defense")
_sys.modules[__name__] = _real
