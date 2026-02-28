"""Backward-compatibility shim — real module at ontic.applied.intent.

This shim exists so that legacy imports like::

    from ontic.intent import X
    from ontic.intent.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.intent``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.intent")
_sys.modules[__name__] = _real
