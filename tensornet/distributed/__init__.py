"""Backward-compatibility shim — real module at tensornet.engine.distributed.

This shim exists so that legacy imports like::

    from tensornet.distributed import X
    from tensornet.distributed.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.distributed``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.distributed")
_sys.modules[__name__] = _real
