"""Backward-compatibility shim — real module at ontic.astro.geophysics.

This shim exists so that legacy imports like::

    from ontic.geophysics import X
    from ontic.geophysics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.astro.geophysics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.astro.geophysics")
_sys.modules[__name__] = _real
