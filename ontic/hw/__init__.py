"""Backward-compatibility shim — real module at ontic.engine.hw.

This shim exists so that legacy imports like::

    from ontic.hw import X
    from ontic.hw.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.hw``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.hw")
_sys.modules[__name__] = _real
