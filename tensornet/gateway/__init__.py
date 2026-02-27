"""Backward-compatibility shim — real module at tensornet.engine.gateway.

This shim exists so that legacy imports like::

    from tensornet.gateway import X
    from tensornet.gateway.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.gateway``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.gateway")
_sys.modules[__name__] = _real
