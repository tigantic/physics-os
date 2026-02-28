"""Backward-compatibility shim — real module at ontic.engine.realtime.

This shim exists so that legacy imports like::

    from ontic.realtime import X
    from ontic.realtime.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.realtime``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.realtime")
_sys.modules[__name__] = _real
