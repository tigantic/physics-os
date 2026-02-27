"""Backward-compatibility shim — real module at tensornet.engine.distributed_tn.

This shim exists so that legacy imports like::

    from tensornet.distributed_tn import X
    from tensornet.distributed_tn.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``tensornet.engine.distributed_tn``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("tensornet.engine.distributed_tn")
_sys.modules[__name__] = _real
