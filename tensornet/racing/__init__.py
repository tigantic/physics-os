"""Backward-compatibility shim — real module at tensornet.aerospace.racing.

This shim exists so that legacy imports like::

    from tensornet.racing import X
    from tensornet.racing.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.aerospace.racing``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.aerospace.racing")
_sys.modules[__name__] = _real
