"""Backward-compatibility shim — real module at ontic.engine.adaptive.

This shim exists so that legacy imports like::

    from ontic.adaptive import X
    from ontic.adaptive.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.adaptive``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.adaptive")
_sys.modules[__name__] = _real
