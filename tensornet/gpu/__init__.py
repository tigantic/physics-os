"""Backward-compatibility shim — real module at tensornet.engine.gpu.

This shim exists so that legacy imports like::

    from tensornet.gpu import X
    from tensornet.gpu.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.gpu``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.gpu")
_sys.modules[__name__] = _real
