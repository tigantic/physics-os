"""Backward-compatibility shim — real module at tensornet.engine.realtime.

This shim exists so that legacy imports like::

    from tensornet.realtime import X
    from tensornet.realtime.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.realtime``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.realtime")
_sys.modules[__name__] = _real
