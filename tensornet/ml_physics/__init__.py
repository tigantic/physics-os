"""Backward-compatibility shim — real module at tensornet.ml.ml_physics.

This shim exists so that legacy imports like::

    from tensornet.ml_physics import X
    from tensornet.ml_physics.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.ml.ml_physics``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.ml.ml_physics")
_sys.modules[__name__] = _real
