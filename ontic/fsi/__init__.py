"""Backward-compatibility shim — real module at ontic.fluids.fsi.

This shim exists so that legacy imports like::

    from ontic.fsi import X
    from ontic.fsi.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.fsi``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.fsi")
_sys.modules[__name__] = _real
