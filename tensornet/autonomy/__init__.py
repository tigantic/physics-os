"""Backward-compatibility shim — real module at tensornet.aerospace.autonomy.

This shim exists so that legacy imports like::

    from tensornet.autonomy import X
    from tensornet.autonomy.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.aerospace.autonomy``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.aerospace.autonomy")
_sys.modules[__name__] = _real
