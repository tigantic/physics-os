"""Backward-compatibility shim — real module at ontic.engine.gpu.

This shim exists so that legacy imports like::

    from ontic.gpu import X
    from ontic.gpu.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.gpu``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.gpu")
_sys.modules[__name__] = _real
