"""Backward-compatibility shim — real module at ontic.applied.physics.

This shim exists so that legacy imports like::

    from ontic.physics import X
    from ontic.physics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.physics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.physics")
_sys.modules[__name__] = _real
