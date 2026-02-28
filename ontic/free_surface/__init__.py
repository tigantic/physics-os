"""Backward-compatibility shim — real module at ontic.fluids.free_surface.

This shim exists so that legacy imports like::

    from ontic.free_surface import X
    from ontic.free_surface.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.free_surface``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.free_surface")
_sys.modules[__name__] = _real
