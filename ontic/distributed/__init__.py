"""Backward-compatibility shim — real module at ontic.engine.distributed.

This shim exists so that legacy imports like::

    from ontic.distributed import X
    from ontic.distributed.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.distributed``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.distributed")
_sys.modules[__name__] = _real
