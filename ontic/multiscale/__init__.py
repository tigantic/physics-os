"""Backward-compatibility shim — real module at ontic.fluids.multiscale.

This shim exists so that legacy imports like::

    from ontic.multiscale import X
    from ontic.multiscale.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.multiscale``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.multiscale")
_sys.modules[__name__] = _real
