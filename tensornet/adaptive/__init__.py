"""Backward-compatibility shim — real module at tensornet.engine.adaptive.

This shim exists so that legacy imports like::

    from tensornet.adaptive import X
    from tensornet.adaptive.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.adaptive``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.adaptive")
_sys.modules[__name__] = _real
