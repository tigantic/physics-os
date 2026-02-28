"""Backward-compatibility shim — real module at ontic.applied.acoustics.

This shim exists so that legacy imports like::

    from ontic.acoustics import X
    from ontic.acoustics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.acoustics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.acoustics")
_sys.modules[__name__] = _real
