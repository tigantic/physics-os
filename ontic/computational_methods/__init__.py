"""Backward-compatibility shim — real module at ontic.fluids.computational_methods.

This shim exists so that legacy imports like::

    from ontic.computational_methods import X
    from ontic.computational_methods.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.computational_methods``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.computational_methods")
_sys.modules[__name__] = _real
