"""Backward-compatibility shim — real module at ontic.plasma_nuclear.fusion.

This shim exists so that legacy imports like::

    from ontic.fusion import X
    from ontic.fusion.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.plasma_nuclear.fusion``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.plasma_nuclear.fusion")
_sys.modules[__name__] = _real
