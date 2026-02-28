"""Backward-compatibility shim — real module at ontic.applied.cyber.

This shim exists so that legacy imports like::

    from ontic.cyber import X
    from ontic.cyber.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.cyber``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.cyber")
_sys.modules[__name__] = _real
