"""Backward-compatibility shim — real module at ontic.applied.radiation.

This shim exists so that legacy imports like::

    from ontic.radiation import X
    from ontic.radiation.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.applied.radiation``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.applied.radiation")
_sys.modules[__name__] = _real
