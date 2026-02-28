"""Backward-compatibility shim — real module at ontic.engine.substrate.

This shim exists so that legacy imports like::

    from ontic.substrate import X
    from ontic.substrate.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.engine.substrate``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.engine.substrate")
_sys.modules[__name__] = _real
