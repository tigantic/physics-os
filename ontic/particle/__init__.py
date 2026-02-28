"""Backward-compatibility shim — real module at ontic.applied.particle.

This shim exists so that legacy imports like::

    from ontic.particle import X
    from ontic.particle.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.particle``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.particle")
_sys.modules[__name__] = _real
