"""Backward-compatibility shim — real module at tensornet.engine.fuel.

This shim exists so that legacy imports like::

    from tensornet.fuel import X
    from tensornet.fuel.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.fuel``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.fuel")
_sys.modules[__name__] = _real
