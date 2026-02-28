"""Backward-compatibility shim — real module at ontic.engine.gateway.

This shim exists so that legacy imports like::

    from ontic.gateway import X
    from ontic.gateway.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.gateway``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.gateway")
_sys.modules[__name__] = _real
