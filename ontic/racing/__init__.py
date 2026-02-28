"""Backward-compatibility shim — real module at ontic.aerospace.racing.

This shim exists so that legacy imports like::

    from ontic.racing import X
    from ontic.racing.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.aerospace.racing``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.aerospace.racing")
_sys.modules[__name__] = _real
