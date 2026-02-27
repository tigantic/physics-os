"""Backward-compatibility shim — real module at tensornet.applied.intent.

This shim exists so that legacy imports like::

    from tensornet.intent import X
    from tensornet.intent.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.applied.intent``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.applied.intent")
_sys.modules[__name__] = _real
