"""Backward-compatibility shim — real module at tensornet.engine.substrate.

This shim exists so that legacy imports like::

    from tensornet.substrate import X
    from tensornet.substrate.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.substrate``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.substrate")
_sys.modules[__name__] = _real
