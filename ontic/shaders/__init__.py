"""Backward-compatibility shim — real module at ontic.applied.shaders.

This shim exists so that legacy imports like::

    from ontic.shaders import X
    from ontic.shaders.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.shaders``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.shaders")
_sys.modules[__name__] = _real
