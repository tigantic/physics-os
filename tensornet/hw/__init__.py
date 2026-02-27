"""Backward-compatibility shim — real module at tensornet.engine.hw.

This shim exists so that legacy imports like::

    from tensornet.hw import X
    from tensornet.hw.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.hw``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.hw")
_sys.modules[__name__] = _real
