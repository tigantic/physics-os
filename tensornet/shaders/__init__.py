"""Backward-compatibility shim — real module at tensornet.applied.shaders.

This shim exists so that legacy imports like::

    from tensornet.shaders import X
    from tensornet.shaders.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.shaders``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.shaders")
_sys.modules[__name__] = _real
