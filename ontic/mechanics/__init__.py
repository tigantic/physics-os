"""Backward-compatibility shim — real module at ontic.materials.mechanics.

This shim exists so that legacy imports like::

    from ontic.mechanics import X
    from ontic.mechanics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.materials.mechanics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.materials.mechanics")
_sys.modules[__name__] = _real
