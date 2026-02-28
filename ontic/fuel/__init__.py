"""Backward-compatibility shim — real module at ontic.engine.fuel.

This shim exists so that legacy imports like::

    from ontic.fuel import X
    from ontic.fuel.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.fuel``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.fuel")
_sys.modules[__name__] = _real
