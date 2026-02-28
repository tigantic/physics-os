"""Backward-compatibility shim — real module at ontic.fluids.multiphase.

This shim exists so that legacy imports like::

    from ontic.multiphase import X
    from ontic.multiphase.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.fluids.multiphase``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.fluids.multiphase")
_sys.modules[__name__] = _real
