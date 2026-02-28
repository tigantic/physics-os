"""Backward-compatibility shim — real module at ontic.ml.ml_physics.

This shim exists so that legacy imports like::

    from ontic.ml_physics import X
    from ontic.ml_physics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.ml.ml_physics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.ml.ml_physics")
_sys.modules[__name__] = _real
